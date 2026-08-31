"""Checks that `pip install -e .` really exposes the project's packages.

The repository uses a flat layout and the scripts import each other through
top-level names (`configs.*`, `utils.*`, `dataset_tools.*`). Getting that wrong
does not fail at packaging time, it fails much later with an ImportError in the
middle of a run, so it is worth pinning down.
"""

import importlib.util
from pathlib import Path

import pytest

try:  # python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - python 3.8 .. 3.10
    tomllib = pytest.importorskip(
        "tomli", reason="needs tomllib (python>=3.11) or the tomli backport"
    )

from conftest import NON_PACKAGE_DIRS, REPO_ROOT

DIST_NAME = "wsi-graph-formers"


@pytest.fixture(scope="module")
def pyproject():
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def declared_packages(pyproject):
    return pyproject["tool"]["setuptools"]["packages"]


def test_pyproject_has_the_expected_metadata(pyproject):
    project = pyproject["project"]
    assert project["name"] == DIST_NAME
    assert project["version"]
    assert project["requires-python"] == ">=3.8", (
        "environment.yaml pins python 3.8, keep the two in sync"
    )


def test_dependencies_are_left_to_environment_yaml(pyproject):
    """torch 1.9.0+cu111 and friends only exist on the PyTorch / PyG indexes.

    Listing them here would make `pip install -e .` try to re-resolve them from
    PyPI and break an already working conda environment.
    """
    assert pyproject["project"]["dependencies"] == []


def test_every_declared_package_exists_on_disk(declared_packages):
    for pkg in declared_packages:
        path = REPO_ROOT.joinpath(*pkg.split("."))
        assert path.is_dir(), f"{pkg} is declared in pyproject.toml but {path} is missing"


def test_no_python_package_is_left_undeclared(declared_packages):
    """A new folder of scripts that nobody added to pyproject.toml."""
    declared = set(declared_packages)
    for path in sorted(REPO_ROOT.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if path.name in NON_PACKAGE_DIRS:
            continue
        if not any(path.glob("*.py")):
            continue
        assert path.name in declared, (
            f"{path.name}/ contains python modules but is not listed in "
            f"[tool.setuptools].packages, so `pip install -e .` will not expose it"
        )


def test_config_files_are_shipped(pyproject):
    """A non-editable `pip install .` must still carry the Hydra yamls."""
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert any("*.yaml" in patterns for patterns in package_data.values())


# --- the install itself -----------------------------------------------------

def _is_installed() -> bool:
    from importlib.metadata import PackageNotFoundError, distribution
    try:
        distribution(DIST_NAME)
    except PackageNotFoundError:
        return False
    return True


requires_install = pytest.mark.skipif(
    not _is_installed(), reason=f"{DIST_NAME} is not installed, run `pip install -e .`"
)


@requires_install
def test_declared_packages_are_importable(declared_packages):
    for pkg in declared_packages:
        assert importlib.util.find_spec(pkg) is not None, (
            f"cannot import {pkg} after installation"
        )


@requires_install
def test_editable_install_points_at_this_checkout(declared_packages):
    for pkg in declared_packages:
        spec = importlib.util.find_spec(pkg)
        locations = list(spec.submodule_search_locations or [])
        if spec.origin and spec.origin != "namespace":
            locations.append(spec.origin)
        assert locations, f"{pkg} resolved to nothing"
        for location in locations:
            resolved = Path(location).resolve()
            assert REPO_ROOT in resolved.parents or resolved == REPO_ROOT, (
                f"{pkg} resolves to {location}, outside this checkout"
            )
