"""Statistics: rates with CIs, paired model comparison, agreement."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(values_by_cluster: Sequence[Sequence[float]], iters: int = 2000,
                         alpha: float = 0.05, seed: int = 17) -> tuple[float, float]:
    """Percentile CI for a mean, resampling clusters (prompts) with replacement.

    Use when there are several samples per prompt: samples of the same prompt are
    correlated, so resampling responses would understate uncertainty.
    """
    clusters = [np.asarray(c, dtype=float) for c in values_by_cluster if len(c)]
    if not clusters:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    sums = np.array([c.sum() for c in clusters])
    counts = np.array([len(c) for c in clusters])
    idx = rng.integers(0, len(clusters), size=(iters, len(clusters)))
    stats = sums[idx].sum(1) / counts[idx].sum(1)
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b, c = discordant counts (A fails only, B fails only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa for two raters over the same items (any hashable labels)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label sequences")
    labels = sorted(set(a) | set(b), key=str)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def bootstrap_kappa_ci(a: Sequence, b: Sequence, iters: int = 2000, seed: int = 17,
                       alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    ks = []
    for _ in range(iters):
        i = rng.integers(0, len(a), len(a))
        ks.append(cohen_kappa(list(a[i]), list(b[i])))
    return (float(np.quantile(ks, alpha / 2)), float(np.quantile(ks, 1 - alpha / 2)))


def binary_agreement(judge: Sequence[int], human: Sequence[int]) -> dict:
    """Judge vs human on fail(1)/pass(0), treating 'fail' as the positive class."""
    j, h = np.asarray(judge, int), np.asarray(human, int)
    tp = int(((j == 1) & (h == 1)).sum()); tn = int(((j == 0) & (h == 0)).sum())
    fp = int(((j == 1) & (h == 0)).sum()); fn = int(((j == 0) & (h == 1)).sum())
    n = len(j)
    div = lambda x, y: x / y if y else float("nan")  # noqa: E731
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "agreement": div(tp + tn, n),
        "agreement_ci": wilson_ci(tp + tn, n),
        "kappa": cohen_kappa(list(j), list(h)) if n else float("nan"),
        "sensitivity": div(tp, tp + fn),      # judge catches human-labelled failures
        "specificity": div(tn, tn + fp),
        "precision": div(tp, tp + fp),
        "judge_fail_rate": div(tp + fp, n),
        "human_fail_rate": div(tp + fn, n),
    }


def percentile(xs: Sequence[float], q: float) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.percentile(xs, q)) if xs else float("nan")
