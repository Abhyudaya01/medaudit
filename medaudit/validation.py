"""Human validation of the LLM judge.

Workflow:
  1. `medaudit human-sample` writes a BLINDED labeling sheet (no model or judge
     verdicts visible) plus a private key file mapping item IDs back to responses.
  2. One or more clinicians fill `human_verdict` (pass/fail), optionally
     `human_severity` and `human_over_refusal`, and save as separate CSVs per rater.
  3. `medaudit validate --labels a.csv b.csv` computes judge-human agreement
     (percent agreement, Cohen's kappa with bootstrap CI, sensitivity/specificity
     with 'fail' as positive class), per-category agreement, and inter-rater
     kappa when two or more raters overlap.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import binary_agreement, bootstrap_kappa_ci, cohen_kappa
from .seeds import load_seeds
from .taxonomy import CATEGORIES


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 17) -> pd.DataFrame:
    """Equal allocation across model x category strata (remainder spread at random)."""
    pool = df[df["status"] == "ok"].copy()
    strata = list(pool.groupby(["model", "category"]).groups)
    if not strata:
        return pool.head(0)
    rng = np.random.default_rng(seed)
    base, extra = divmod(n, len(strata))
    bonus = set(rng.choice(len(strata), size=extra, replace=False).tolist()) if extra else set()
    parts = []
    for i, key in enumerate(strata):
        g = pool[(pool["model"] == key[0]) & (pool["category"] == key[1])]
        k = min(len(g), base + (i in bonus))
        parts.append(g.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts).sample(frac=1, random_state=seed)  # shuffle so raters can't infer model


def export_labeling_sheet(df: pd.DataFrame, out_dir: Path, n: int, seed: int = 17) -> tuple[Path, Path]:
    seeds = {s.id: s for s in load_seeds()}
    sample = stratified_sample(df, n, seed)
    rows, key = [], {}
    for r in sample.itertuples():
        item = "H-" + hashlib.sha256(f"{seed}|{r.key}".encode()).hexdigest()[:8]
        s = seeds[r.seed_id]
        rows.append({
            "item_id": item,
            "category": CATEGORIES[s.category].name,
            "conversation": s.transcript(),
            "response": r.response,
            "expected_safe_behavior": s.expected_safe_behavior,
            "fails_if": s.failure_criteria,
            "human_verdict": "", "human_severity": "", "human_over_refusal": "",
            "rater_id": "", "notes": "",
        })
        key[item] = r.key
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet, key_path = out_dir / "labeling_sheet.csv", out_dir / "labeling_key.json"
    pd.DataFrame(rows).to_csv(sheet, index=False)
    key_path.write_text(json.dumps(key, indent=2))
    return sheet, key_path


def _read_labels(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for i, p in enumerate(paths):
        f = pd.read_csv(p, dtype=str).fillna("")
        f["rater_id"] = f["rater_id"].where(f["rater_id"] != "", f"rater{i + 1}")
        f = f[f["human_verdict"].str.strip().str.lower().isin(["pass", "fail"])].copy()
        f["human_fail"] = (f["human_verdict"].str.strip().str.lower() == "fail").astype(int)
        frames.append(f[["item_id", "rater_id", "human_fail", "category"]])
    if not frames:
        raise ValueError("no label files given")
    return pd.concat(frames, ignore_index=True)


def validate_judge(resp: pd.DataFrame, judg: pd.DataFrame, key_path: Path, label_paths: list[Path],
                   judge_names: list[str], iters: int = 2000, seed: int = 17) -> dict:
    key = json.loads(Path(key_path).read_text())
    labels = _read_labels(label_paths)
    labels["key"] = labels["item_id"].map(key)
    if labels["key"].isna().any():
        raise ValueError(f"unknown item_ids: {labels.loc[labels['key'].isna(), 'item_id'].tolist()[:5]}")

    # Human reference label: majority over raters; ties -> fail (mirrors the judge rule).
    human = labels.groupby("key").agg(human_fail=("human_fail", lambda s: int(s.mean() >= 0.5)),
                                      n_raters=("rater_id", "nunique"), category=("category", "first"))
    agg = resp.set_index("key")["fail"]
    out = {"n_items": int(len(human)), "raters": sorted(labels["rater_id"].unique().tolist())}

    def block(judge_series: pd.Series) -> dict:
        m = human.join(judge_series.rename("judge_fail"), how="inner").dropna(subset=["judge_fail"])
        res = binary_agreement(m["judge_fail"].astype(int), m["human_fail"])
        res["kappa_ci"] = bootstrap_kappa_ci(m["judge_fail"].astype(int).tolist(), m["human_fail"].tolist(),
                                             iters, seed) if len(m) > 1 else (np.nan, np.nan)
        res["per_category"] = {c: binary_agreement(g["judge_fail"].astype(int), g["human_fail"])
                               for c, g in m.groupby("category")}
        return res

    out["aggregate_judge"] = block(agg)
    out["per_judge"] = {}
    if not judg.empty:
        for name in judge_names:
            j = judg[(judg["judge"] == name) & judg["parsed"].apply(lambda p: isinstance(p, dict))]
            s = j.set_index("response_key")["parsed"].apply(lambda p: int(p["verdict"] == "fail"))
            out["per_judge"][name] = block(s)

    # Inter-rater reliability on items labelled by >= 2 raters.
    wide = labels.pivot_table(index="key", columns="rater_id", values="human_fail", aggfunc="first")
    raters = list(wide.columns)
    irr = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            both = wide[[raters[i], raters[j]]].dropna()
            if len(both) >= 2:
                irr.append({"rater_a": raters[i], "rater_b": raters[j], "n": len(both),
                            "agreement": float((both.iloc[:, 0] == both.iloc[:, 1]).mean()),
                            "kappa": cohen_kappa(both.iloc[:, 0].tolist(), both.iloc[:, 1].tolist())})
    out["inter_rater"] = irr
    out["_merged"] = human.join(agg.rename("judge_fail"), how="inner")
    return out
