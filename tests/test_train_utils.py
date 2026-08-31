"""Fold construction and graph-id parsing.

Marked `torch` because `utils.train_utils` imports torch at module level, but
the logic under test is plain numpy / string handling.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="needs the wsi-graph-formers environment")
pytest.importorskip("pandas")

pytestmark = pytest.mark.torch

from utils.train_utils import (  # noqa: E402
    cv_train_test_indices,
    fix_seed,
    get_graph_id,
    get_patchgraph_id,
    has_epithelial_nodes,
    normalize_wsi,
)

# --- cross-validation folds -------------------------------------------------

def test_train_and_test_partition_the_dataset():
    train, test = cv_train_test_indices(37, k_folds=3, testfold=0)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(37))
    assert len(train) + len(test) == 37


@pytest.mark.parametrize("k_folds", [2, 3, 5])
def test_the_test_folds_tile_the_dataset_exactly_once(k_folds):
    """No sample may be tested twice, and none may be left untested."""
    n = 41
    seen = []
    for fold in range(k_folds):
        _, test = cv_train_test_indices(n, k_folds=k_folds, testfold=fold)
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(n))


def test_folds_are_reproducible_for_a_given_seed():
    a = cv_train_test_indices(50, k_folds=3, testfold=1, seed=123)
    b = cv_train_test_indices(50, k_folds=3, testfold=1, seed=123)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_a_different_seed_gives_a_different_split():
    a = cv_train_test_indices(50, k_folds=3, testfold=0, seed=1)[1]
    b = cv_train_test_indices(50, k_folds=3, testfold=0, seed=2)[1]
    assert set(a) != set(b)


@pytest.mark.parametrize("testfold", [-1, 3, 99])
def test_an_out_of_range_fold_is_rejected(testfold):
    with pytest.raises(ValueError):
        cv_train_test_indices(30, k_folds=3, testfold=testfold)


# --- graph identifiers, used to keep a patient inside a single fold ---------

@pytest.mark.parametrize(
    "filename, expected",
    [
        ("wsigraph_93_4.pt", "wsigraph_93"),
        ("wsigraph_7_0.pt", "wsigraph_7"),
        ("subgraph_graph_r50_12_99.pt", "subgraph_graph_r50_12"),
    ],
)
def test_get_graph_id_drops_the_subgraph_suffix(filename, expected):
    assert get_graph_id(SimpleNamespace(name=filename)) == expected


def test_get_patchgraph_id_drops_the_subgraph_suffix():
    assert get_patchgraph_id(SimpleNamespace(name="tilegraph_93_4.pt")) == "tilegraph_93"


def test_subgraphs_of_one_slide_share_an_id():
    ids = {get_graph_id(SimpleNamespace(name=f"wsigraph_12_{i}.pt")) for i in range(100)}
    assert ids == {"wsigraph_12"}, "subgraphs of a slide must not straddle two folds"


def test_normalize_wsi_collapses_whitespace():
    assert normalize_wsi("  slide   A \t B ") == "slide A B"


# --- epithelial nodes -------------------------------------------------------

@pytest.mark.parametrize(
    "labels, expected",
    [([1, 2, 3], False), ([1, 4], True), ([5], True), ([4, 5], True), ([], False)],
)
def test_has_epithelial_nodes_looks_for_classes_4_and_5(labels, expected):
    graph = SimpleNamespace(label=torch.tensor(labels, dtype=torch.long))
    assert has_epithelial_nodes(graph) is expected


def test_has_epithelial_nodes_accepts_a_plain_list():
    assert has_epithelial_nodes(SimpleNamespace(label=[1, 5])) is True


# --- seeding ----------------------------------------------------------------

def test_fix_seed_makes_torch_reproducible():
    fix_seed(7)
    first = torch.rand(5)
    fix_seed(7)
    assert torch.equal(first, torch.rand(5))
