"""Live audit as a test suite: one test per (model, seed); a test FAILS when the judge
says the model's response is unsafe.

    pytest tests/test_live_audit.py --live --audit-config configs/default.yaml -k "MD-0"

This reuses the run store, so responses/judgments are cached and shared with the CLI.
Use it for quick targeted checks (e.g. a regression on one category); use
`medaudit run` for the full study and statistics.
"""
import pytest

from medaudit.analysis import load_frames
from medaudit.config import load_config
from medaudit.runner import RunPaths, run_generation, run_judging
from medaudit.seeds import load_seeds

SEEDS = load_seeds()


@pytest.fixture(scope="session")
def live_cfg(pytestconfig):
    return load_config(pytestconfig.getoption("--audit-config"))


@pytest.fixture(scope="session")
def judged(live_cfg, request):
    selected = {item.callspec.params["seed_id"] for item in request.session.items
                if hasattr(item, "callspec") and "seed_id" in item.callspec.params}
    live_cfg.seed_ids = sorted(selected) or None
    run_generation(live_cfg, print)
    run_judging(live_cfg, print)
    resp, _ = load_frames(live_cfg, RunPaths(live_cfg))
    return resp


def pytest_generate_tests(metafunc):
    """Parametrize over the models in whichever config --audit-config points to."""
    if "model" in metafunc.fixturenames:
        try:
            names = [m.name for m in load_config(metafunc.config.getoption("--audit-config")).models]
        except Exception:
            names = ["unconfigured"]
        metafunc.parametrize("model", names)


@pytest.mark.live
@pytest.mark.parametrize("seed_id", [s.id for s in SEEDS])
def test_model_is_safe(judged, model, seed_id):
    rows = judged[(judged.model == model) & (judged.seed_id == seed_id)]
    if rows.empty:
        pytest.skip(f"{model} not in --audit-config")
    if rows["fail"].isna().all():
        pytest.fail(f"no valid judgment (generation or judge error): {rows['error'].dropna().tolist()[:1]}")
    failed = rows[rows.fail == 1]
    assert failed.empty, (f"{model} failed {seed_id} (severity {int(failed.severity.max())}): "
                          f"{failed.response.iloc[0][:300]!r}")
