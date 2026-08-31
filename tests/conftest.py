"""Shared fixtures.

The suite is split in two tiers:

* tests that only need the standard library (plus pytest and PyYAML). They check
  packaging, Hydra configs and the consistency between the README and the code.
  They run anywhere, including in CI, in a couple of seconds.
* tests marked `torch` that need the real environment from `environment.yaml`.
  They are skipped, not failed, when torch / PyG are missing.
"""

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are not python packages of this project.
NON_PACKAGE_DIRS = {
    ".git", ".github", "__pycache__", "docs", "data", "outputs", "predictions",
    "checkpoints", "tests", "Data_General", "models-arxiv", "build", "dist",
}


@lru_cache(maxsize=1)
def tracked_files():
    """Files under version control, as absolute paths.

    Everything else is either local scratch (`configs/dev-experiments/`, run
    outputs) or macOS resource forks. CI only ever sees tracked files, so the
    tests look at exactly the same set locally.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return tuple(REPO_ROOT / name for name in out.stdout.split("\0") if name)


def tracked_matching(pattern: str):
    """Tracked files whose repo-relative path matches a glob."""
    return tuple(sorted(p for p in tracked_files() if p.match(pattern) and p.is_file()))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: needs the full torch / PyG environment")
    # Let the tests import the project even when `pip install -e .` was not run.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
