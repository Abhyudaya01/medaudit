"""Turn raw JSONL into tidy frames and paper tables."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import RunConfig
from .judge import aggregate
from .metrics import cluster_bootstrap_ci, cohen_kappa, holm, mcnemar_exact, percentile, wilson_ci
from .runner import RunPaths
from .seeds import load_seeds
from .store import JsonlStore
from .taxonomy import CATEGORIES, CATEGORY_ORDER, TACTICS


def load_frames(cfg: RunConfig, paths: RunPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (responses, judgments). `responses` has one row per generation with the
    aggregated judge verdict attached (fail=1/0, NaN if unjudged)."""
    resp = pd.DataFrame(JsonlStore(paths.responses).load().values())
    judg = pd.DataFrame(JsonlStore(paths.judgments).load().values())
    if resp.empty:
        raise RuntimeError(f"no responses in {paths.responses}; run `medaudit generate` first")
    resp = resp[resp["model"].isin([m.name for m in cfg.models])].copy()
    judge_names = [j.model.name for j in cfg.judges]

    agg_rows = []
    if not judg.empty:
        judg = judg[judg["judge"].isin(judge_names)].copy()
        for rk, grp in judg.groupby("response_key"):
            a = aggregate({r.judge: r.parsed for r in grp.itertuples() if isinstance(r.parsed, dict)})
            agg_rows.append({
                "key": rk,
                "fail": None if a is None else int(a.verdict == "fail"),
                "severity": None if a is None else a.severity,
                "over_refusal": None if a is None else int(a.over_refusal),
                "judges_unanimous": None if a is None else a.unanimous,
                "n_judges": 0 if a is None else a.n_judges,
                "judge_cost_usd": grp["cost_usd"].sum(min_count=1) if "cost_usd" in grp else None,
            })
    agg = pd.DataFrame(agg_rows, columns=["key", "fail", "severity", "over_refusal",
                                          "judges_unanimous", "n_judges", "judge_cost_usd"])
    resp = resp.merge(agg, on="key", how="left")
    seeds = {s.id: s for s in load_seeds()}
    resp["pair_id"] = resp["seed_id"].map(lambda i: seeds[i].pair_id if i in seeds else None)
    resp["severity_if_failed"] = resp["seed_id"].map(lambda i: seeds[i].severity_if_failed if i in seeds else None)
    return resp, judg


def _rate_row(sub: pd.DataFrame, samples: int, iters: int, seed: int) -> dict:
    sub = sub[sub["fail"].notna()]
    n, k = len(sub), int(sub["fail"].sum())
    lo, hi = wilson_ci(k, n)
    if samples > 1:  # correlated samples per prompt -> cluster bootstrap over prompts
        lo, hi = cluster_bootstrap_ci([g["fail"].tolist() for _, g in sub.groupby("seed_id")], iters, seed=seed)
    return {"n": n, "fails": k, "rate": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi}


def failure_by_model_category(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    rows = []
    for model in [m.name for m in cfg.models]:
        for cat in CATEGORY_ORDER + ["__all__"]:
            sub = df[df["model"] == model] if cat == "__all__" else df[(df["model"] == model) & (df["category"] == cat)]
            r = _rate_row(sub, cfg.samples_per_prompt, cfg.bootstrap_iters, cfg.random_seed)
            rows.append({"model": model, "category": "Overall" if cat == "__all__" else CATEGORIES[cat].name,
                         "category_key": cat, **r})
    return pd.DataFrame(rows)


def failure_by_tactic(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    rows = []
    for model in [m.name for m in cfg.models]:
        for t in TACTICS:
            sub = df[(df["model"] == model) & (df["tactic"] == t)]
            if sub["fail"].notna().any():
                rows.append({"model": model, "tactic": TACTICS[t].name, "tactic_key": t,
                             **_rate_row(sub, cfg.samples_per_prompt, cfg.bootstrap_iters, cfg.random_seed)})
    return pd.DataFrame(rows)


def model_summary(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    rows = []
    for spec in cfg.models:
        sub = df[df["model"] == spec.name]
        ok = sub[sub["status"] == "ok"]
        judged = ok[ok["fail"].notna()]
        r = _rate_row(judged, cfg.samples_per_prompt, cfg.bootstrap_iters, cfg.random_seed)
        fails = judged[judged["fail"] == 1]
        cost = ok["cost_usd"].sum(min_count=1) if "cost_usd" in ok else np.nan
        lat = ok["latency_s"].tolist() if "latency_s" in ok else []
        rows.append({
            "model": spec.name, "provider": spec.provider, "model_id": spec.model_id,
            "n_requests": len(sub), "n_errors": int((sub["status"] == "error").sum()),
            "n_judged": r["n"], "failure_rate": r["rate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            "severe_failure_rate": (fails["severity"] == 3).sum() / r["n"] if r["n"] else np.nan,
            "mean_severity_given_fail": fails["severity"].mean() if len(fails) else np.nan,
            "over_refusal_rate": judged["over_refusal"].mean() if r["n"] else np.nan,
            "input_tokens": int(ok["input_tokens"].sum()) if len(ok) else 0,
            "output_tokens": int(ok["output_tokens"].sum()) if len(ok) else 0,
            "mean_output_tokens": ok["output_tokens"].mean() if len(ok) else np.nan,
            "cost_usd": cost,
            "cost_per_100_prompts_usd": cost / len(ok) * 100 if len(ok) and pd.notna(cost) else np.nan,
            "judge_cost_usd": judged["judge_cost_usd"].sum(min_count=1),
            "latency_mean_s": float(np.mean(lat)) if lat else np.nan,
            "latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
        })
    return pd.DataFrame(rows)


def pairwise_model_tests(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Exact McNemar on per-prompt verdicts (majority over samples), Holm-adjusted."""
    per = (df[df["fail"].notna()].groupby(["model", "seed_id"])["fail"].mean() >= 0.5).astype(int).unstack(0)
    rows = []
    for a, b in combinations([m.name for m in cfg.models], 2):
        if a not in per or b not in per:
            continue
        both = per[[a, b]].dropna()
        b_only_a = int(((both[a] == 1) & (both[b] == 0)).sum())
        c_only_b = int(((both[a] == 0) & (both[b] == 1)).sum())
        rows.append({"model_a": a, "model_b": b, "n_prompts": len(both),
                     "a_fail_rate": both[a].mean(), "b_fail_rate": both[b].mean(),
                     "a_only_fails": b_only_a, "b_only_fails": c_only_b,
                     "p_mcnemar": mcnemar_exact(b_only_a, c_only_b)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_mcnemar"].tolist())
    return out


def counterfactual_pairs(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Per model and pair: fail rate of each variant and whether verdicts disagree."""
    rows = []
    sub = df[df["pair_id"].notna() & df["fail"].notna()]
    seeds = {s.id: s for s in load_seeds()}
    for (model, pid), g in sub.groupby(["model", "pair_id"]):
        rates = g.groupby("seed_id")["fail"].mean()
        rec = {"model": model, "pair_id": pid,
               "variants": "; ".join(f"{seeds[s].pair_attribute}={rates[s]:.2f}" for s in sorted(rates.index)),
               "max_rate_gap": float(rates.max() - rates.min()),
               "discordant": bool((rates >= 0.5).nunique() > 1)}
        rows.append(rec)
    return pd.DataFrame(rows)


def inter_judge_agreement(judg: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    names = [j.model.name for j in cfg.judges]
    if len(names) < 2 or judg.empty:
        return pd.DataFrame()
    v = judg[judg["parsed"].apply(lambda p: isinstance(p, dict))].copy()
    v["fail"] = v["parsed"].apply(lambda p: int(p["verdict"] == "fail"))
    wide = v.pivot_table(index="response_key", columns="judge", values="fail")
    rows = []
    for a, b in combinations(names, 2):
        if a in wide and b in wide:
            both = wide[[a, b]].dropna()
            if len(both):
                rows.append({"judge_a": a, "judge_b": b, "n": len(both),
                             "agreement": float((both[a] == both[b]).mean()),
                             "kappa": cohen_kappa(both[a].tolist(), both[b].tolist())})
    return pd.DataFrame(rows)


def judge_health(judg: pd.DataFrame) -> pd.DataFrame:
    if judg.empty:
        return pd.DataFrame()
    g = judg.groupby("judge")
    return pd.DataFrame({
        "n": g.size(),
        "parse_failures": g["parsed"].apply(lambda s: int(sum(not isinstance(p, dict) for p in s))),
        "repaired": g["repaired"].apply(lambda s: int(s.fillna(False).sum())) if "repaired" in judg else 0,
        "cost_usd": g["cost_usd"].sum(min_count=1) if "cost_usd" in judg else np.nan,
    }).reset_index()
