"""Statistics against known values."""
import math

import pytest

from medaudit.metrics import (binary_agreement, cluster_bootstrap_ci, cohen_kappa, holm,
                              mcnemar_exact, wilson_ci)


def test_wilson_known_values():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson_ci(0, 17)
    assert lo == 0.0 and hi == pytest.approx(0.1843, abs=1e-3)
    assert all(math.isnan(x) for x in wilson_ci(0, 0))


def test_mcnemar_exact():
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(0.021484375)


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]


def test_cohen_kappa_known():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # Classic 2x2 example: po=0.7, pe=0.5 -> kappa=0.4
    a = [1] * 25 + [1] * 10 + [0] * 5 + [0] * 10
    b = [1] * 25 + [0] * 10 + [1] * 5 + [0] * 10
    po = 35 / 50
    pe = (35 / 50) * (30 / 50) + (15 / 50) * (20 / 50)
    assert cohen_kappa(a, b) == pytest.approx((po - pe) / (1 - pe))
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 0])


def test_binary_agreement_confusion():
    r = binary_agreement(judge=[1, 1, 0, 0, 1], human=[1, 0, 0, 1, 1])
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 1, 1)
    assert r["sensitivity"] == pytest.approx(2 / 3)
    assert r["specificity"] == pytest.approx(0.5)
    assert r["agreement"] == pytest.approx(0.6)


def test_cluster_bootstrap_brackets_mean():
    clusters = [[1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]] * 5
    lo, hi = cluster_bootstrap_ci(clusters, iters=1000)
    assert lo < 0.5 < hi
    assert cluster_bootstrap_ci([[1, 1]] * 10, iters=200) == (1.0, 1.0)
