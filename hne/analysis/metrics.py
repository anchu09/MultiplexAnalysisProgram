"""Segmentation evaluation metrics for the H&E invasion-front detection model.

All functions operate on flat 1-D arrays of integer class labels and return
plain Python dicts / numpy arrays so they can be used independently of any
visualization framework.
"""

from __future__ import annotations

import numpy as np
from multiplex_pipeline.hne.config import CLASS_LABELS, IGNORE_INDEX, N_CLASSES, RANDOM_SEED
from sklearn.metrics import (
    auc,
    roc_curve,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute the confusion matrix, excluding IGNORE_INDEX pixels.

    Args:
        y_true: Flat integer array of true labels.
        y_pred: Flat integer array of predicted labels.

    Returns:
        np.ndarray of shape (n_classes, n_classes).
    """
    valid = y_true != IGNORE_INDEX
    return sk_confusion_matrix(y_true[valid], y_pred[valid], labels=list(range(N_CLASSES)))


def compute_iou_dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-class IoU and Dice scores from flat label arrays.

    Args:
        y_true: Flat integer array of true labels (IGNORE_INDEX pixels excluded).
        y_pred: Flat integer array of predicted labels.

    Returns:
        Dict mapping "iou" and "dice" to per-class dicts {class_name: score}.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom_iou > 0, tp / denom_iou, 0.0)
        dice = np.where(denom_dice > 0, 2 * tp / denom_dice, 0.0)

    names = list(CLASS_LABELS.values())
    return {
        "iou": {n: float(v) for n, v in zip(names, iou, strict=False)},
        "dice": {n: float(v) for n, v in zip(names, dice, strict=False)},
    }


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, F1-score, and accuracy.

    Args:
        y_true: Flat integer array of true labels.
        y_pred: Flat integer array of predicted labels.

    Returns:
        Dict mapping "precision", "recall", "f1", "accuracy" to per-class dicts.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    total = cm.sum()
    tn = total - (tp + fp + fn)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
        accuracy = (tp + tn) / total

    names = list(CLASS_LABELS.values())
    return {
        "precision": {n: float(v) for n, v in zip(names, precision, strict=False)},
        "recall": {n: float(v) for n, v in zip(names, recall, strict=False)},
        "f1": {n: float(v) for n, v in zip(names, f1, strict=False)},
        "accuracy": {n: float(v) for n, v in zip(names, accuracy, strict=False)},
    }


def compute_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute global accuracy and macro-averaged precision, recall, F1.

    Args:
        y_true: Flat integer array of true labels.
        y_pred: Flat integer array of predicted labels.

    Returns:
        Dict with keys "accuracy", "precision", "recall", "f1".
    """
    report = compute_classification_report(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    accuracy = float(tp.sum() / cm.sum())
    return {
        "accuracy": accuracy,
        "precision": float(np.mean(list(report["precision"].values()))),
        "recall": float(np.mean(list(report["recall"].values()))),
        "f1": float(np.mean(list(report["f1"].values()))),
    }


def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_pixels: int = 200_000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Compute per-class ROC AUC, optionally sub-sampling for speed.

    Args:
        y_true: Flat integer array of ground-truth labels.
        y_score: Float array of shape (N_pixels, n_classes) with class probabilities.
        max_pixels: Maximum pixels to use; if N > max_pixels a random subset is drawn.
        seed: Random seed for sub-sampling reproducibility.

    Returns:
        Dict mapping each class name to its AUC value.
    """
    n = y_true.shape[0]
    if n > max_pixels:
        idx = np.random.default_rng(seed).choice(n, size=max_pixels, replace=False)
        y_true_s = y_true[idx]
        y_score_s = y_score[idx]
    else:
        y_true_s, y_score_s = y_true, y_score

    result: dict[str, float] = {}
    for cls in range(N_CLASSES):
        yt = (y_true_s == cls).astype(int)
        fpr, tpr, _ = roc_curve(yt, y_score_s[:, cls])
        result[CLASS_LABELS[cls]] = float(auc(fpr, tpr))
    return result


def compute_per_patch_iou(
    y_true_patches: list[np.ndarray],
    y_pred_patches: list[np.ndarray],
) -> np.ndarray:
    """Compute per-patch IoU for classes 1 (invasion front) and 2 (stroma).

    Args:
        y_true_patches: List of flat 1-D arrays of true labels, one per patch.
        y_pred_patches: List of flat 1-D arrays of predicted labels.

    Returns:
        np.ndarray of shape (n_patches, n_classes). Zero where the class is
        absent in both arrays.
    """
    ious = []
    for t, p in zip(y_true_patches, y_pred_patches, strict=False):
        cm = sk_confusion_matrix(t, p, labels=list(range(N_CLASSES)))
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom > 0, tp / denom, 0.0)
        ious.append(iou)
    return np.vstack(ious)
