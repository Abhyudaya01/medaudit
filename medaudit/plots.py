"""Paper figures (matplotlib only). Each figure is written as PDF (vector) and PNG.

Color rules: models take categorical slots in fixed order (never cycled); rates use
a single-hue sequential ramp; severity uses an ordinal ramp of the same hue. Text
stays in neutral ink. Every multi-series figure has a legend or direct labels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

from .taxonomy import CATEGORIES, CATEGORY_ORDER  # noqa: E402

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OVERFLOW = "#8a8984"  # models beyond 8 share a neutral and rely on direct labels
SEQ = LinearSegmentedColormap.from_list(
    "medaudit_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEVERITY_COLORS = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def model_colors(models: list[str]) -> dict[str, str]:
    return {m: CATEGORICAL[i] if i < len(CATEGORICAL) else OVERFLOW for i, m in enumerate(models)}


def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def _ink_for(rgba) -> str:
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.5 else INK


def _heatmap(ax, mat: np.ndarray, rows: list[str], cols: list[str], vmax: float):
    im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color=_ink_for(SEQ(v / vmax if vmax else 0)))
    return im


def fig_model_category_heatmap(tbl: pd.DataFrame, models: list[str], out: Path) -> list[Path]:
    cols = CATEGORY_ORDER + ["__all__"]
    labels = [CATEGORIES[c].short for c in CATEGORY_ORDER] + ["Overall"]
    mat = np.array([[tbl.loc[(tbl.model == m) & (tbl.category_key == c), "rate"].squeeze()
                     if ((tbl.model == m) & (tbl.category_key == c)).any() else np.nan
                     for c in cols] for m in models], dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(cols), 0.9 + 0.42 * len(models)))
    vmax = max(0.05, np.nanmax(mat) if np.isfinite(mat).any() else 1)
    im = _heatmap(ax, mat, models, labels, vmax)
    ax.axvline(len(cols) - 1.5, color="white", linewidth=4)  # separate the Overall column
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    ax.set_title("Failure rate by model and category", loc="left", color=INK)
    return _save(fig, out, "fig1_failure_heatmap")


def fig_category_bars(tbl: pd.DataFrame, models: list[str], out: Path) -> list[Path]:
    colors = model_colors(models)
    cats = CATEGORY_ORDER
    x = np.arange(len(cats))
    w = min(0.8 / max(1, len(models)), 0.22)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, m in enumerate(models):
        sub = tbl[(tbl.model == m)].set_index("category_key").reindex(cats)
        pos = x + (i - (len(models) - 1) / 2) * w
        ax.bar(pos, sub["rate"], width=w * 0.92, color=colors[m], label=m, edgecolor="white", linewidth=1)
        zero = (sub["rate"] == 0).to_numpy()  # a 0% bar is invisible; draw a stub so it reads as "zero"
        if zero.any():
            ax.hlines(np.full(zero.sum(), 0.004), pos[zero] - w * 0.46, pos[zero] + w * 0.46,
                      colors=colors[m], linewidth=3, zorder=3)
        err = np.vstack([sub["rate"] - sub["ci_lo"], sub["ci_hi"] - sub["rate"]]).clip(min=0)
        ax.errorbar(pos, sub["rate"], yerr=err, fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=1.5)
    ax.set_xticks(x, [CATEGORIES[c].short for c in cats])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Failure rate (95% CI)")
    ax.set_ylim(0, min(1.0, max(0.1, np.nanmax(tbl.loc[tbl.category_key != "__all__", "ci_hi"]) * 1.1)))
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=min(4, len(models)), loc="upper left", bbox_to_anchor=(0, 1.18), fontsize=8)
    return _save(fig, out, "fig2_category_bars")


def fig_cost_vs_failure(summary: pd.DataFrame, models: list[str], out: Path) -> list[Path] | None:
    s = summary.dropna(subset=["cost_per_100_prompts_usd", "failure_rate"])
    if s.empty:
        return None
    colors = model_colors(models)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for r in s.itertuples():
        ax.errorbar(r.cost_per_100_prompts_usd, r.failure_rate,
                    yerr=[[r.failure_rate - r.ci_lo], [r.ci_hi - r.failure_rate]],
                    fmt="o", ms=7, color=colors[r.model], ecolor=INK_2, elinewidth=0.8, capsize=2,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(r.model, (r.cost_per_100_prompts_usd, r.failure_rate), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 100 prompts (USD, log scale)")
    ax.set_ylabel("Overall failure rate (95% CI)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Safety vs. cost", loc="left", color=INK)
    return _save(fig, out, "fig3_cost_vs_failure")


def fig_latency(df: pd.DataFrame, models: list[str], out: Path) -> list[Path] | None:
    data = [df.loc[(df.model == m) & (df.status == "ok"), "latency_s"].dropna().values for m in models]
    if not any(len(d) for d in data):
        return None
    colors = model_colors(models)
    fig, ax = plt.subplots(figsize=(5.0, 0.6 + 0.4 * len(models)))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK_2,
                                "markeredgecolor": "none", "alpha": 0.6},
                    whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.85); patch.set_edgecolor(colors[m])
    ax.set_yticks(range(1, len(models) + 1), models)
    ax.invert_yaxis()
    ax.set_xlabel("Response latency (s, wall-clock)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return _save(fig, out, "fig4_latency")


def fig_severity(df: pd.DataFrame, models: list[str], out: Path) -> list[Path] | None:
    j = df[df.fail.notna()]
    if j.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 0.7 + 0.4 * len(models)))
    for i, m in enumerate(models):
        sub = j[j.model == m]
        left = 0.0
        for sev in (3, 2, 1):  # most severe anchored to the baseline
            share = ((sub.fail == 1) & (sub.severity == sev)).mean() if len(sub) else 0
            ax.barh(i, share, left=left, color=SEVERITY_COLORS[sev], edgecolor="white", linewidth=2,
                    height=0.6, label={3: "Severe", 2: "Moderate", 1: "Minor"}[sev] if i == 0 else None)
            left += share
        ax.text(left + 0.005, i, f"{left:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of responses judged unsafe, by severity")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return _save(fig, out, "fig5_severity")


def fig_tactic_heatmap(tt: pd.DataFrame, models: list[str], out: Path) -> list[Path] | None:
    if tt.empty:
        return None
    tactics = list(dict.fromkeys(tt["tactic"]))
    mat = np.array([[tt.loc[(tt.model == m) & (tt.tactic == t), "rate"].squeeze()
                     if ((tt.model == m) & (tt.tactic == t)).any() else np.nan for t in tactics]
                    for m in models], dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(tactics), 0.9 + 0.42 * len(models)))
    _heatmap(ax, mat, models, tactics, max(0.05, np.nanmax(mat)))
    ax.set_title("Failure rate by adversarial tactic", loc="left", color=INK)
    return _save(fig, out, "fig6_tactic_heatmap")


def fig_judge_confusion(merged: pd.DataFrame, out: Path) -> list[Path]:
    m = merged.dropna(subset=["judge_fail"])
    cm = np.array([[((m.human_fail == h) & (m.judge_fail == j)).sum() for j in (0, 1)] for h in (0, 1)])
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.imshow(cm, cmap=SEQ, vmin=0, vmax=max(1, cm.max()))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                    color=_ink_for(SEQ(cm[i, j] / max(1, cm.max()))))
    ax.set_xticks([0, 1], ["Pass", "Fail"]); ax.set_yticks([0, 1], ["Pass", "Fail"])
    ax.set_xlabel("LLM judge"); ax.set_ylabel("Human")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Judge vs. human labels", loc="left", color=INK)
    return _save(fig, out, "fig7_judge_confusion")
