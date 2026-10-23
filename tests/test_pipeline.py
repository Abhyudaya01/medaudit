"""End-to-end pipeline on mock providers: generate -> judge -> report -> human validation."""
import json

import pandas as pd

from medaudit.analysis import load_frames
from medaudit.config import load_config
from medaudit.report import build_report
from medaudit.runner import RunPaths, run_generation, run_judging
from medaudit.store import JsonlStore
from medaudit.validation import export_labeling_sheet, validate_judge

quiet = lambda *a, **k: None  # noqa: E731


def _run(cfg_path):
    cfg = load_config(cfg_path)
    run_generation(cfg, quiet)
    run_judging(cfg, quiet)
    return cfg, RunPaths(cfg)


def test_generation_records_cost_latency_tokens(mock_config):
    cfg, paths = _run(mock_config)
    recs = list(JsonlStore(paths.responses).load().values())
    assert len(recs) == 100 * len(cfg.models)
    assert all(r["status"] == "ok" for r in recs)
    assert all(r["input_tokens"] > 0 and r["output_tokens"] > 0 and r["cost_usd"] > 0 for r in recs)
    assert all("latency_s" in r for r in recs)
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["n_seeds"] == 100 and manifest["seed_set_hash"]


def test_resume_makes_no_new_calls(mock_config):
    cfg, paths = _run(mock_config)
    n_resp = sum(1 for _ in paths.responses.open())
    n_judg = sum(1 for _ in paths.judgments.open())
    run_generation(cfg, quiet)
    run_judging(cfg, quiet)
    assert sum(1 for _ in paths.responses.open()) == n_resp
    assert sum(1 for _ in paths.judgments.open()) == n_judg


def test_errored_responses_are_retried(mock_config):
    cfg, paths = _run(mock_config)
    store = JsonlStore(paths.responses)
    victim = next(iter(store.load().values()))
    store.append({**victim, "status": "error", "error": "simulated"})
    assert store.load()[victim["key"]]["status"] == "error"
    run_generation(cfg, quiet)
    assert store.load()[victim["key"]]["status"] == "ok"


def test_failure_rates_match_mock_behaviour(mock_config):
    cfg, paths = _run(mock_config)
    resp, _ = load_frames(cfg, paths)
    rates = resp.groupby("model")["fail"].mean()
    assert rates["Mock-Careful"] == 0.0
    assert rates["Mock-Reckless"] == 1.0
    assert 0.0 < rates["Mock-Mixed"] < 1.0
    assert resp.loc[resp.model == "Mock-Mixed", "over_refusal"].sum() > 0


def test_report_writes_tables_and_figures(mock_config):
    cfg, paths = _run(mock_config)
    summary = build_report(cfg, progress=quiet)
    for t in ["table1_failure_rates", "table2_model_summary", "table3_failure_by_tactic",
              "table4_pairwise_mcnemar", "table5_counterfactual_pairs", "table6_inter_judge_agreement"]:
        for ext in ("csv", "tex", "md"):
            assert (paths.results / f"{t}.{ext}").exists(), (t, ext)
    for f in ["fig1_failure_heatmap", "fig2_category_bars", "fig3_cost_vs_failure",
              "fig4_latency", "fig5_severity", "fig6_tactic_heatmap"]:
        assert (paths.figures / f"{f}.pdf").stat().st_size > 1000
        assert (paths.figures / f"{f}.png").exists()
    t1 = pd.read_csv(paths.results / "table1_failure_rates.csv")
    assert list(t1.columns)[-1] == "Overall" and len(t1) == len(cfg.models)
    assert summary["unjudged_ok_responses"] == 0
    tex = (paths.results / "table1_failure_rates.tex").read_text()
    assert r"\toprule" in tex and r"\textbf{" in tex


def test_human_validation_round_trip(mock_config):
    cfg, paths = _run(mock_config)
    resp, judg = load_frames(cfg, paths)
    sheet, key_path = export_labeling_sheet(resp, paths.human, 60, cfg.random_seed)
    labels = pd.read_csv(sheet)
    assert len(labels) == 60
    assert not {"model", "judge", "verdict"} & set(labels.columns)  # blinded
    key = json.loads(key_path.read_text())

    # Rater 1 agrees with the judge everywhere; rater 2 flips 6 items.
    truth = resp.set_index("key")["fail"]
    r1 = labels.copy()
    r1["human_verdict"] = [("fail" if truth[key[i]] == 1 else "pass") for i in r1.item_id]
    r1["rater_id"] = "clin1"
    r2 = r1.copy()
    r2.loc[:5, "human_verdict"] = r2.loc[:5, "human_verdict"].map({"fail": "pass", "pass": "fail"})
    r2["rater_id"] = "clin2"
    p1, p2 = paths.human / "r1.csv", paths.human / "r2.csv"
    r1.to_csv(p1, index=False)
    r2.to_csv(p2, index=False)

    v = validate_judge(resp, judg, key_path, [p1], [j.model.name for j in cfg.judges], iters=200)
    assert v["aggregate_judge"]["agreement"] == 1.0 and v["aggregate_judge"]["kappa"] == 1.0

    v2 = validate_judge(resp, judg, key_path, [p1, p2], [j.model.name for j in cfg.judges], iters=200)
    assert v2["inter_rater"] and v2["inter_rater"][0]["n"] == 60
    assert v2["inter_rater"][0]["agreement"] == 0.9
    assert set(v2["per_judge"]) == {j.model.name for j in cfg.judges}

    build_report(cfg, v2, progress=quiet)
    assert (paths.results / "table7_judge_validation.csv").exists()
    assert (paths.figures / "fig7_judge_confusion.pdf").exists()


def test_cli_smoke(mock_config, capsys):
    from medaudit.cli import main
    main(["list-seeds"])
    main(["estimate", "-c", str(mock_config)])
    main(["run", "-c", str(mock_config), "--limit", "5", "--run-name", "cli-smoke"])
    assert "cli-smoke" in capsys.readouterr().out
