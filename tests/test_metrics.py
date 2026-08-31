"""Evaluation metrics.

`metric: bacc` in the configs means every number in the paper goes through
`eval_binary_bacc`. Balanced accuracy on an imbalanced problem is exactly where
a silent argmax or axis mistake would flatter the results.
"""

import pytest

torch = pytest.importorskip("torch", reason="needs the wsi-graph-formers environment")
pytest.importorskip("sklearn")
# models.data_utils imports torch_sparse at module level
pytest.importorskip("torch_sparse")

pytestmark = pytest.mark.torch

from models.data_utils import (  # noqa: E402
    eval_binary_acc,
    eval_binary_bacc,
    eval_binary_f1,
)


def test_perfect_predictions_score_one():
    y = torch.tensor([0, 1, 1, 0, 1])
    logits = torch.nn.functional.one_hot(y, num_classes=2).float()
    assert eval_binary_acc(y, logits) == pytest.approx(1.0)
    assert eval_binary_bacc(y, logits) == pytest.approx(1.0)


def test_inverted_predictions_score_zero():
    y = torch.tensor([0, 1, 1, 0])
    logits = torch.nn.functional.one_hot(1 - y, num_classes=2).float()
    assert eval_binary_acc(y, logits) == pytest.approx(0.0)


def test_accuracy_accepts_hard_labels_as_well_as_logits():
    y = torch.tensor([0, 1, 1, 0])
    hard = torch.tensor([0, 1, 0, 0])
    logits = torch.nn.functional.one_hot(hard, num_classes=2).float()
    assert eval_binary_acc(y, hard) == pytest.approx(0.75)
    assert eval_binary_acc(y, logits) == pytest.approx(0.75)


def test_balanced_accuracy_punishes_the_majority_class_shortcut():
    """9 negatives, 1 positive, always predict negative.

    Plain accuracy says 0.9 and balanced accuracy says 0.5. Only the second one
    is honest about a model that never finds a tumor cell.
    """
    y = torch.tensor([0] * 9 + [1])
    always_negative = torch.zeros(10, dtype=torch.long)
    assert eval_binary_acc(y, always_negative) == pytest.approx(0.9)
    assert eval_binary_bacc(y, always_negative) == pytest.approx(0.5)


def test_balanced_accuracy_is_the_mean_of_the_per_class_recalls():
    y = torch.tensor([0, 0, 0, 0, 1, 1])
    pred = torch.tensor([0, 0, 0, 1, 1, 0])
    # recall(0) = 3/4, recall(1) = 1/2
    assert eval_binary_bacc(y, pred) == pytest.approx((0.75 + 0.5) / 2)


def test_f1_is_computed_on_the_positive_class():
    y = torch.tensor([0, 1, 1, 1])
    pred = torch.tensor([0, 1, 1, 0])
    # precision = 1.0, recall = 2/3
    assert eval_binary_f1(y, pred) == pytest.approx(2 * (1.0 * 2 / 3) / (1.0 + 2 / 3))
