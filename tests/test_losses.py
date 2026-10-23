import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toxicity.losses import FocalLossWithLogits, build_loss, pos_weight_from_labels  # noqa: E402


def test_pos_weight_is_negatives_over_positives():
    y = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]])
    assert pos_weight_from_labels(y).tolist() == pytest.approx([3.0, 1.0, 1.0])
    assert pos_weight_from_labels(y, power=0.5).tolist() == pytest.approx([3**0.5, 1.0, 1.0])
    assert pos_weight_from_labels(y, max_weight=2.0).tolist() == pytest.approx([2.0, 1.0, 1.0])


def test_focal_with_gamma_zero_is_plain_bce():
    logits = torch.tensor([[2.0, -1.0], [0.3, 0.7]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bce = torch.nn.BCEWithLogitsLoss()(logits, targets)
    assert FocalLossWithLogits(gamma=0.0)(logits, targets).item() == pytest.approx(bce.item())


def test_focal_shrinks_easy_examples_more_than_hard_ones():
    target = torch.tensor([[1.0]])
    easy, hard = torch.tensor([[4.0]]), torch.tensor([[-4.0]])
    bce, focal = torch.nn.BCEWithLogitsLoss(), FocalLossWithLogits(gamma=2.0)
    assert focal(easy, target) / bce(easy, target) < 0.01
    assert focal(hard, target) / bce(hard, target) > 0.9


def test_build_loss_by_name():
    y = np.array([[1, 0], [0, 0], [0, 0], [0, 1]], dtype=np.float32)
    weighted = build_loss("weighted_bce", y, {"pos_weight_power": 1.0})
    assert weighted.pos_weight.tolist() == pytest.approx([3.0, 3.0])
    assert isinstance(build_loss("focal", y, {}), FocalLossWithLogits)
    assert isinstance(build_loss("bce", y, {}), torch.nn.BCEWithLogitsLoss)
    with pytest.raises(ValueError):
        build_loss("hinge", y, {})
