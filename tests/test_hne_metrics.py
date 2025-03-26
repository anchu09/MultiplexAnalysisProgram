"""Unit tests for hne/analysis/metrics.py."""

import pytest

pytest.importorskip("sklearn", reason="requires uv sync --group ml")

import numpy as np
import pytest
from multiplex_pipeline.hne.analysis.metrics import (
    compute_classification_report,
    compute_confusion_matrix,
    compute_iou_dice,
    compute_overall_metrics,
    compute_per_patch_iou,
    compute_roc_auc,
)
from multiplex_pipeline.hne.config import CLASS_LABELS, IGNORE_INDEX, N_CLASSES


def _perfect_labels(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth == prediction, n pixels per class."""
    y = np.repeat(np.arange(N_CLASSES), n)
    return y, y.copy()


def _uniform_probs(y_true: np.ndarray) -> np.ndarray:
    """Uninformative class probabilities (1/N_CLASSES for every pixel)."""
    return np.full((len(y_true), N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32)


def _perfect_probs(y_true: np.ndarray) -> np.ndarray:
    """One-hot probability vectors matching y_true."""
    P = np.zeros((len(y_true), N_CLASSES), dtype=np.float32)
    P[np.arange(len(y_true)), y_true] = 1.0
    return P


class TestComputeConfusionMatrix:
    def test_shape(self):
        y_true, y_pred = _perfect_labels()
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (N_CLASSES, N_CLASSES)

    def test_perfect_prediction_diagonal(self):
        y_true, y_pred = _perfect_labels(100)
        cm = compute_confusion_matrix(y_true, y_pred)
        assert (np.diag(cm) == 100).all()
        assert (cm - np.diag(np.diag(cm)) == 0).all()

    def test_ignore_index_excluded(self):
        y_true = np.array([0, 1, 2, IGNORE_INDEX, 0])
        y_pred = np.array([0, 1, 2, 0, 1])
        cm = compute_confusion_matrix(y_true, y_pred)
        # Total counted pixels = 4 (the IGNORE_INDEX pixel is excluded)
        assert cm.sum() == 4

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 2, 0, 2, 0, 1])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert np.diag(cm).sum() == 0


class TestComputeIouDice:
    def test_perfect_scores(self):
        y_true, y_pred = _perfect_labels(200)
        result = compute_iou_dice(y_true, y_pred)
        for name in CLASS_LABELS.values():
            assert result["iou"][name] == pytest.approx(1.0)
            assert result["dice"][name] == pytest.approx(1.0)

    def test_keys_present(self):
        y_true, y_pred = _perfect_labels()
        result = compute_iou_dice(y_true, y_pred)
        assert set(result.keys()) == {"iou", "dice"}
        assert set(result["iou"].keys()) == set(CLASS_LABELS.values())

    def test_scores_bounded(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, N_CLASSES, 1000)
        y_pred = rng.integers(0, N_CLASSES, 1000)
        result = compute_iou_dice(y_true, y_pred)
        for score in list(result["iou"].values()) + list(result["dice"].values()):
            assert 0.0 <= score <= 1.0


class TestComputeClassificationReport:
    def test_perfect_f1(self):
        y_true, y_pred = _perfect_labels(50)
        report = compute_classification_report(y_true, y_pred)
        for name in CLASS_LABELS.values():
            assert report["f1"][name] == pytest.approx(1.0)
            assert report["precision"][name] == pytest.approx(1.0)
            assert report["recall"][name] == pytest.approx(1.0)

    def test_metric_keys(self):
        y_true, y_pred = _perfect_labels()
        report = compute_classification_report(y_true, y_pred)
        assert set(report.keys()) == {"precision", "recall", "f1", "accuracy"}


class TestComputeOverallMetrics:
    def test_perfect_accuracy(self):
        y_true, y_pred = _perfect_labels(100)
        result = compute_overall_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_output_keys(self):
        y_true, y_pred = _perfect_labels()
        result = compute_overall_metrics(y_true, y_pred)
        assert set(result.keys()) == {"accuracy", "precision", "recall", "f1"}


class TestComputeRocAuc:
    def test_perfect_auc(self):
        y_true, _ = _perfect_labels(100)
        y_prob = _perfect_probs(y_true)
        result = compute_roc_auc(y_true, y_prob)
        for name, auc_val in result.items():
            assert auc_val == pytest.approx(1.0), f"AUC != 1.0 for class {name}"

    def test_random_auc_near_half(self):
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, N_CLASSES, 10_000)
        y_prob = _uniform_probs(y_true)
        result = compute_roc_auc(y_true, y_prob)
        for name, auc_val in result.items():
            assert 0.4 < auc_val < 0.6, f"Uniform AUC far from 0.5 for {name}: {auc_val:.3f}"

    def test_output_keys(self):
        y_true, _ = _perfect_labels(50)
        result = compute_roc_auc(y_true, _perfect_probs(y_true))
        assert set(result.keys()) == set(CLASS_LABELS.values())

    def test_subsampling(self):
        y_true = np.repeat(np.arange(N_CLASSES), 1000)
        y_prob = _perfect_probs(y_true)
        result = compute_roc_auc(y_true, y_prob, max_pixels=100)
        assert len(result) == N_CLASSES


class TestComputePerPatchIou:
    def test_output_shape(self):
        patches = [np.repeat(np.arange(N_CLASSES), 50)] * 5
        result = compute_per_patch_iou(patches, patches)
        assert result.shape == (5, N_CLASSES)

    def test_perfect_iou(self):
        p = np.repeat(np.arange(N_CLASSES), 100)
        result = compute_per_patch_iou([p] * 3, [p] * 3)
        assert np.allclose(result, 1.0)

    def test_absent_class_gives_zero(self):
        """A class absent from both true and pred should give IoU = 0."""
        y = np.zeros(300, dtype=int)  # only class 0
        result = compute_per_patch_iou([y], [y])
        # Class 0 is perfect, classes 1 and 2 have denom=0 → IoU=0
        assert result[0, 1] == 0.0
        assert result[0, 2] == 0.0
