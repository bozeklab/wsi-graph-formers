"""Feature normalization and cell-type masking.

`celltype_asfeature` and `zscore_normalization` are the two switches of the
feature ablation study, so their behaviour is worth pinning down: a z-scoring
fitted on the wrong data, or a mask that misses a column, leaks the label the
model is supposed to predict.
"""

import pytest

torch = pytest.importorskip("torch", reason="needs the wsi-graph-formers environment")
pytest.importorskip("torch_geometric")

pytestmark = pytest.mark.torch

from torch_geometric.data import Data  # noqa: E402

from utils.graph_utils import (  # noqa: E402
    fit_zscore_stats_pyg,
    mask_celltype_onehot_cols,
    mask_on_graph_list,
    normalize_zscore_pyg,
)

CONT_IDX = [0, 1, 2, 3]


def _graph(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.rand(n, 4, generator=g) * 100 + 10,
        centroid=torch.rand(n, 2, generator=g) * 1000,
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )


# --- z-scoring --------------------------------------------------------------

def test_stats_cover_the_centroid_and_the_continuous_columns():
    stats = fit_zscore_stats_pyg([_graph()], cont_idx=CONT_IDX)
    assert stats["mu"].shape == (6,)  # 2 centroid + 4 continuous
    assert stats["sd"].shape == (6,)


def test_normalized_features_are_centred_and_scaled():
    graph = _graph(n=64)
    stats = fit_zscore_stats_pyg([graph], cont_idx=CONT_IDX)
    out = normalize_zscore_pyg(graph, stats, cont_idx=CONT_IDX)

    assert torch.allclose(out.x.mean(0), torch.zeros(6), atol=1e-5)
    assert torch.allclose(out.x.std(0), torch.ones(6), atol=1e-5)


def test_a_constant_column_does_not_divide_by_zero():
    graph = _graph(n=16)
    graph.x[:, 2] = 3.0
    stats = fit_zscore_stats_pyg([graph], cont_idx=CONT_IDX)
    out = normalize_zscore_pyg(graph, stats, cont_idx=CONT_IDX)
    assert torch.isfinite(out.x).all()


def test_raw_features_and_statistics_are_kept_for_reproducibility():
    graph = _graph()
    raw = graph.x.clone()
    stats = fit_zscore_stats_pyg([graph], cont_idx=CONT_IDX)
    out = normalize_zscore_pyg(graph, stats, cont_idx=CONT_IDX)

    assert torch.equal(out.x_raw, raw)
    assert out.transform_version == "zscore_v1"
    assert torch.allclose(out.transform_mu, stats["mu"])
    assert out.feature_names[:2] == ["centroid_x", "centroid_y"]


def test_statistics_come_from_the_training_graphs_only():
    """Fitting on train and applying to test must not recentre the test graph."""
    train, test = _graph(n=32, seed=1), _graph(n=32, seed=2)
    test.x += 500.0  # a shifted, unseen slide
    stats = fit_zscore_stats_pyg([train], cont_idx=CONT_IDX)
    out = normalize_zscore_pyg(test, stats, cont_idx=CONT_IDX)
    assert out.x.mean().abs() > 1.0, "test graph was normalized with its own statistics"


def test_centroids_can_be_normalized_by_slide_size():
    graph = _graph(n=16)
    scaled = fit_zscore_stats_pyg(
        [graph], cont_idx=CONT_IDX, normalize_centroid=(1000, 1000)
    )
    plain = fit_zscore_stats_pyg([graph], cont_idx=CONT_IDX)
    assert not torch.allclose(scaled["mu"][:2], plain["mu"][:2])


# --- cell-type masking ------------------------------------------------------

def _onehot_graph():
    data = _graph(n=6)
    data.x = torch.ones(6, 5)
    data.feature_names = ["cont_0", "cell_3", "cell_4", "cell_5", "cont_1"]
    return data


def test_masking_zeroes_only_the_requested_cell_columns():
    data = _onehot_graph()
    mask_celltype_onehot_cols(data, classes=(4, 5))

    assert (data.x[:, 2] == 0).all() and (data.x[:, 3] == 0).all()
    assert (data.x[:, 0] == 1).all() and (data.x[:, 1] == 1).all()
    assert (data.x[:, 4] == 1).all()


def test_masking_is_a_no_op_without_feature_names():
    data = _graph()
    before = data.x.clone()
    mask_celltype_onehot_cols(data, classes=(4, 5))
    assert torch.equal(data.x, before)


def test_masking_ignores_classes_that_have_no_column():
    data = _onehot_graph()
    mask_celltype_onehot_cols(data, classes=(99,))
    assert (data.x == 1).all()


def test_masking_applies_to_every_graph_of_a_list():
    graphs = [_onehot_graph() for _ in range(3)]
    mask_on_graph_list(graphs, classes=(4, 5))
    assert all((g.x[:, 2] == 0).all() for g in graphs)


def test_a_non_pyg_item_is_rejected_in_strict_mode():
    with pytest.raises(TypeError):
        mask_on_graph_list([_onehot_graph(), "not a graph"], classes=(4,))
