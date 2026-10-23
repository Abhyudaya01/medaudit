from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="run the live audit against real model APIs")
    parser.addoption("--audit-config", default=str(ROOT / "configs" / "default.yaml"),
                     help="config used by the live audit tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live API test; pass --live to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def mock_config(tmp_path):
    """The mock config, redirected to a temp output dir."""
    text = (ROOT / "configs" / "mock.yaml").read_text().replace("output_dir: runs", f"output_dir: {tmp_path}")
    p = tmp_path / "mock.yaml"
    p.write_text(text)
    return p
