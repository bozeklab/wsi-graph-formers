"""Consistency of the Hydra configuration tree.

Hydra resolves most of this lazily, at run time: a typo in an experiment file
surfaces as an unrelated crash after the dataset has been loaded, or worse, as a
silently ignored override. These tests make that a two-second failure instead.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from conftest import REPO_ROOT, tracked_matching  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
GT_CONFIGS = CONFIGS / "Graph_Transformers"
EXPERIMENTS = GT_CONFIGS / "experiments"

# Only versioned configs: `configs/dev/` and `configs/Graph_Transformers/dev-experiments/`
# are gitignored scratch space and are none of the test suite's business.
ALL_YAML = tracked_matching("configs/*.yaml") + tracked_matching("configs/**/*.yaml")
EXPERIMENT_YAML = tracked_matching("configs/Graph_Transformers/experiments/*.yaml")


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def test_configs_are_present():
    assert (CONFIGS / "config.yaml").is_file()
    assert (GT_CONFIGS / "config_largewsi.yaml").is_file()
    assert EXPERIMENT_YAML, "no experiment config found"


@pytest.mark.parametrize("path", ALL_YAML, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_yaml_parses(path):
    assert isinstance(_load(path), dict)


@pytest.mark.parametrize("path", EXPERIMENT_YAML, ids=lambda p: p.stem)
def test_experiment_only_overrides_known_keys(path):
    """Experiment files are `# @package _global_`, so every key they set must
    already exist in config_largewsi.yaml, otherwise the override is a typo."""
    base = _load(GT_CONFIGS / "config_largewsi.yaml")
    unknown = sorted(set(_load(path)) - set(base) - {"defaults"})
    assert not unknown, (
        f"{path.name} sets keys absent from config_largewsi.yaml: {unknown}"
    )


@pytest.mark.parametrize("path", EXPERIMENT_YAML, ids=lambda p: p.stem)
def test_experiment_is_package_global(path):
    head = path.read_text(encoding="utf-8").lstrip().splitlines()[0].strip()
    assert head.startswith("# @package _global_"), (
        f"{path.name} misses the `# @package _global_` header, its keys would be "
        f"nested under `experiments` instead of overriding the main config"
    )


@pytest.mark.parametrize("path", EXPERIMENT_YAML, ids=lambda p: p.stem)
def test_referenced_order_file_exists(path):
    order_file = _load(path).get("order_file")
    if not order_file:
        return
    assert (REPO_ROOT / order_file).is_file(), (
        f"{path.name} points at a missing order file: {order_file}"
    )


def _methods_known_to_parse_method() -> set:
    """Read models/parse.py as text; importing it would pull in torch."""
    source = (REPO_ROOT / "models" / "parse.py").read_text(encoding="utf-8")
    known = set(re.findall(r"cfg\.method\s*==\s*['\"]([\w-]+)['\"]", source))
    for group in re.findall(r"cfg\.method\s+in\s*\[([^\]]*)\]", source):
        known.update(re.findall(r"['\"]([\w-]+)['\"]", group))
    return known


def test_parse_method_exposes_the_documented_models():
    known = _methods_known_to_parse_method()
    assert {"sgformer", "nodeformer", "difformer"} <= known
    # the binary variants are the ones actually used by the paper experiments
    assert {"gcnbin", "gatbin", "sgcbin", "sgc2bin", "signbin"} <= known


@pytest.mark.parametrize("path", ALL_YAML, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_configured_method_is_implemented(path):
    method = _load(path).get("method")
    if not method:  # generalgnn experiments take it from the command line
        return
    assert method in _methods_known_to_parse_method(), (
        f"{path.name} sets method={method}, which models/parse.py does not handle"
    )


# --- the README must not send people to configs that do not exist -----------

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_experiments_quoted_in_readme_exist():
    quoted = set(re.findall(r"experiments=([\w.-]+)", README)) - {"<chosen_experiment>"}
    missing = sorted(n for n in quoted if not (EXPERIMENTS / f"{n}.yaml").is_file())
    assert not missing, f"README documents experiments that do not exist: {missing}"


def test_methods_quoted_in_readme_exist():
    known = _methods_known_to_parse_method()
    quoted = set(re.findall(r"method=([\w-]+)", README)) - {"chosen_method"}
    missing = sorted(m for m in quoted if m not in known)
    assert not missing, f"README documents methods parse.py does not handle: {missing}"
