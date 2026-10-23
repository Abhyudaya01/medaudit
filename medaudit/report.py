"""Write paper tables (CSV + LaTeX + Markdown) and figures for a run."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import analysis as A
from . import plots as P
from .config import RunConfig
from .runner import RunPaths
from .taxonomy import CATEGORIES, CATEGORY_ORDER


def _fmt_rate(r, lo, hi) -> str:
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "--"
    return f"{100 * r:.1f} ({100 * lo:.1f}-{100 * hi:.1f})"


def _latex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def to_latex(df: pd.DataFrame, caption: str, label: str, bold_min_cols: list[str] | None = None) -> str:
    """Minimal booktabs table; optionally bolds the lowest value per column."""
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(_latex_escape(str(c)) for c in cols) + r" \\", r"\midrule"]
    mins = {}
    for c in bold_min_cols or []:
        vals = df[c].map(lambda s: float(str(s).split()[0]) if str(s)[:1].isdigit() else math.inf)
        mins[c] = vals.min()
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = _latex_escape(str(row[c]))
            if c in mins and str(row[c])[:1].isdigit() and float(str(row[c]).split()[0]) == mins[c]:
                v = r"\textbf{" + v + "}"
            cells.append(v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              r"\end{table}"]
    return "\n".join(lines)


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def _write(df: pd.DataFrame, out: Path, name: str, caption: str, bold: list[str] | None = None):
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.tex").write_text(to_latex(df, caption, f"tab:{name}", bold))
    (out / f"{name}.md").write_text(to_markdown(df))


def build_report(cfg: RunConfig, validation: dict | None = None, progress=print) -> dict:
    paths = RunPaths(cfg)
    resp, judg = A.load_frames(cfg, paths)
    models = [m.name for m in cfg.models]
    out, figs = paths.results, paths.figures

    mc = A.failure_by_model_category(resp, cfg)
    mc.to_csv(out / "failure_by_model_category_long.csv", index=False)
    wide = pd.DataFrame({"Model": models})
    for c in CATEGORY_ORDER + ["__all__"]:
        label = CATEGORIES[c].short if c != "__all__" else "Overall"
        wide[label] = [_fmt_rate(*mc.loc[(mc.model == m) & (mc.category_key == c), ["rate", "ci_lo", "ci_hi"]]
                                 .iloc[0].tolist()) for m in models]
    ci_note = ("Wilson 95% CIs" if cfg.samples_per_prompt == 1
               else f"95% CIs from a prompt-level cluster bootstrap ({cfg.bootstrap_iters} resamples)")
    _write(wide, out, "table1_failure_rates",
           f"Failure rate (\\%) by model and category, with {ci_note}. Lower is better; best per column in bold.",
           bold=list(wide.columns[1:]))

    summ = A.model_summary(resp, cfg)
    summ.to_csv(out / "model_summary_full.csv", index=False)
    t2 = pd.DataFrame({
        "Model": summ.model,
        "Failure % (95% CI)": [_fmt_rate(r.failure_rate, r.ci_lo, r.ci_hi) for r in summ.itertuples()],
        "Severe %": (100 * summ.severe_failure_rate).round(1),
        "Over-refusal %": (100 * summ.over_refusal_rate).round(1),
        "Cost / 100 prompts ($)": summ.cost_per_100_prompts_usd.round(4),
        "Latency p50 (s)": summ.latency_p50_s.round(2),
        "Latency p95 (s)": summ.latency_p95_s.round(2),
        "Errors": summ.n_errors,
    })
    _write(t2, out, "table2_model_summary",
           "Per-model safety, cost, and latency. Cost covers the audited model only; judge cost is reported separately.")

    tt = A.failure_by_tactic(resp, cfg)
    if not tt.empty:
        tt.to_csv(out / "failure_by_tactic_long.csv", index=False)
        t3 = tt.assign(cell=[_fmt_rate(r.rate, r.ci_lo, r.ci_hi) for r in tt.itertuples()]) \
               .pivot(index="tactic", columns="model", values="cell").reset_index()
        _write(t3, out, "table3_failure_by_tactic", "Failure rate (\\%) by adversarial tactic.")

    pw = A.pairwise_model_tests(resp, cfg)
    if not pw.empty:
        pw.to_csv(out / "pairwise_mcnemar_raw.csv", index=False)  # unrounded p-values
        pfmt = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"  # noqa: E731
        pw = pw.assign(a_fail_rate=pw.a_fail_rate.round(3), b_fail_rate=pw.b_fail_rate.round(3),
                       p_mcnemar=pw.p_mcnemar.map(pfmt), p_holm=pw.p_holm.map(pfmt))
        _write(pw, out, "table4_pairwise_mcnemar",
               "Pairwise model comparison on per-prompt verdicts: exact McNemar test, Holm-adjusted.")

    cf = A.counterfactual_pairs(resp, cfg)
    if not cf.empty:
        _write(cf, out, "table5_counterfactual_pairs",
               "Counterfactual pairs: fail rate per variant and verdict discordance.")

    ij = A.inter_judge_agreement(judg, cfg)
    if not ij.empty:
        _write(ij.round(3), out, "table6_inter_judge_agreement", "Agreement between judge models.")
    jh = A.judge_health(judg)
    if not jh.empty:
        jh.to_csv(out / "judge_health.csv", index=False)

    written = []
    written += P.fig_model_category_heatmap(mc, models, figs)
    written += P.fig_category_bars(mc, models, figs)
    written += P.fig_cost_vs_failure(summ, models, figs) or []
    written += P.fig_latency(resp, models, figs) or []
    written += P.fig_severity(resp, models, figs) or []
    written += P.fig_tactic_heatmap(tt, models, figs) or []

    if validation:
        v = {k: val for k, val in validation.items() if not k.startswith("_")}
        (out / "judge_validation.json").write_text(json.dumps(v, indent=2, default=float))
        a = validation["aggregate_judge"]
        rows = [{"Judge": "Aggregate", "n": a["n"], "Agreement %": round(100 * a["agreement"], 1),
                 "Cohen's kappa (95% CI)": f"{a['kappa']:.2f} ({a['kappa_ci'][0]:.2f}-{a['kappa_ci'][1]:.2f})",
                 "Sensitivity": round(a["sensitivity"], 3), "Specificity": round(a["specificity"], 3)}]
        for name, b in validation["per_judge"].items():
            rows.append({"Judge": name, "n": b["n"], "Agreement %": round(100 * b["agreement"], 1),
                         "Cohen's kappa (95% CI)": f"{b['kappa']:.2f} ({b['kappa_ci'][0]:.2f}-{b['kappa_ci'][1]:.2f})",
                         "Sensitivity": round(b["sensitivity"], 3), "Specificity": round(b["specificity"], 3)})
        _write(pd.DataFrame(rows), out, "table7_judge_validation",
               "LLM judge vs. clinician labels ('fail' = positive class).")
        written += P.fig_judge_confusion(validation["_merged"], figs)

    unjudged = int(resp.loc[resp.status == "ok", "fail"].isna().sum())
    summary = {"run": cfg.run_name, "responses": int(len(resp)), "unjudged_ok_responses": unjudged,
               "tables": sorted(p.name for p in out.glob("table*.csv")),
               "figures": sorted(p.name for p in written if p.suffix == ".pdf")}
    (out / "report_summary.json").write_text(json.dumps(summary, indent=2))
    progress(f"tables -> {out}\nfigures -> {figs}")
    if unjudged:
        progress(f"warning: {unjudged} successful responses have no valid judgment; re-run `medaudit judge`")
    return summary
