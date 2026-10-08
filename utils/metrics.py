"""Anomaly-detection metrics.

This module implements the two evaluation protocols the project cares about:

* **Pointwise** -- every timestep is judged independently. This is the *primary*
  target of the project because it is much harder to game than the segment-level
  protocol and better reflects real localisation quality.
* **Point-adjusted (PA)** -- the widely used (but optimistic) protocol where a
  whole ground-truth anomaly segment counts as detected if *any* point inside it
  is flagged. Reported for comparability with prior work.

It also provides threshold-free ranking metrics (AUROC / AUPRC) and several
threshold-selection strategies, including a label-free Peaks-Over-Threshold
(POT) estimator so that the test labels are never required to pick a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

try:  # sklearn is optional but used when available for robust ranking metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - fallback path
    _HAS_SKLEARN = False

try:
    from vus.metrics import get_metrics

    _HAS_VUS = True
except ImportError:
    try:
        from metrics.vus.metrics import get_metrics

        _HAS_VUS = True
    except ImportError:
        get_metrics = None
        _HAS_VUS = False


@dataclass
class PRF:
    """Container for a precision/recall/F1 result at a fixed threshold."""

    precision: float
    recall: float
    f1: float
    threshold: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def point_adjust(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Apply the standard point-adjustment to a binary prediction vector.

    For each contiguous ground-truth anomaly segment, if at least one timestep
    inside the segment is predicted positive, the entire segment is marked
    positive. Predictions outside anomaly segments are left untouched.
    """
    pred = pred.astype(bool).copy()
    label = label.astype(bool)

    n = len(label)
    i = 0
    while i < n:
        if label[i]:
            j = i
            while j < n and label[j]:
                j += 1
            if pred[i:j].any():
                pred[i:j] = True
            i = j
        else:
            i += 1
    return pred.astype(int)

def anomaly_detection_delay(pred, labels):
    pred = np.asarray(pred).astype(int).flatten()
    labels = np.asarray(labels).astype(int).flatten()

    if pred.shape != labels.shape:
        raise ValueError(
            f"Prediction and label shapes differ: "
            f"{pred.shape} versus {labels.shape}"
        )

    transitions = np.diff(
        np.concatenate(([0], labels, [0]))
    )

    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    if len(starts) == 0:
        return 0.0

    delays = []

    for start, end in zip(starts, ends):
        hits = np.flatnonzero(pred[start:end] == 1)

        if hits.size > 0:
            delays.append(float(hits[0]))
        else:
            # Same general behavior as IMDiffusion:
            # a missed anomaly receives its full segment length.
            delays.append(float(end - start))

    return float(np.mean(delays))

def add_values_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Return raw and point-adjusted ADD at one fixed threshold."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)

    raw_pred = (scores > threshold).astype(int)
    adjusted_pred = point_adjust(raw_pred.copy(), labels)

    return {
        "raw_add": anomaly_detection_delay(raw_pred, labels),
        "adjusted_add": anomaly_detection_delay(adjusted_pred, labels),
    }

def debug_anomaly_detection_delay(pred, labels):
    pred = np.asarray(pred, dtype=int).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)

    if pred.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred.shape}, labels={labels.shape}"
        )

    diff = np.diff(np.concatenate([[0], labels, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    delays = []

    print("\n========== ADD DEBUG ==========")

    for i, (start, end) in enumerate(zip(starts, ends), 1):
        segment_pred = pred[start:end]
        hits = np.flatnonzero(segment_pred == 1)

        segment_length = end - start
        detected_points = int(segment_pred.sum())
        missed_points = segment_length - detected_points

        if hits.size > 0:
            delay = int(hits[0])
            status = "DETECTED"
        else:
            delay = int(segment_length)
            status = "MISSED"

        delays.append(delay)

        print(
            f"Segment {i:3d}: "
            f"start={start:6d}, "
            f"end={end:6d}, "
            f"length={segment_length:4d}, "
            f"hits={detected_points:3d}, "
            f"missed_points={missed_points:3d}, "
            f"delay={delay:3d}, "
            f"{status}"
        )

    print("--------------------------------")
    print(f"Detected segments : {sum(d < (e-s) for d, s, e in zip(delays, starts, ends))}")
    print(f"Total segments    : {len(starts)}")
    print(f"Delays            : {delays}")

    if delays:
        print(f"Mean ADD          : {np.mean(delays):.6f}")
    else:
        print("No ground-truth anomaly segments")

    print("================================\n")

def add_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    adjust: bool = False,
) -> float:
    pred = (scores > threshold).astype(int)

    if adjust:
        pred = point_adjust(pred.copy(), labels)

    return anomaly_detection_delay(pred, labels)

def debug_adjusted_evaluation(scores, labels):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)

    best_adj = best_f1_search(
        scores,
        labels,
        adjust=True,
    )

    raw_pred = (scores > best_adj.threshold).astype(int)
    adjusted_pred = point_adjust(raw_pred.copy(), labels)

    raw_add = anomaly_detection_delay(raw_pred, labels)
    adjusted_add = anomaly_detection_delay(adjusted_pred, labels)

    print("\n========== ADJUSTED EVALUATION DEBUG ==========")
    print(f"Adjusted-optimal threshold : {best_adj.threshold:.10f}")
    print(f"Adjusted precision         : {best_adj.precision:.6f}")
    print(f"Adjusted recall            : {best_adj.recall:.6f}")
    print(f"Adjusted F1                : {best_adj.f1:.6f}")
    print(f"Raw positives              : {raw_pred.sum()}")
    print(f"Adjusted positives         : {adjusted_pred.sum()}")
    print(f"Raw ADD at PA threshold    : {raw_add:.6f}")
    print(f"ADD after point adjustment : {adjusted_add:.6f}")
    print(
        "Predictions removed by adjustment:",
        int(((raw_pred == 1) & (adjusted_pred == 0)).sum()),
    )
    print(
        "Predictions added by adjustment:",
        int(((raw_pred == 0) & (adjusted_pred == 1)).sum()),
    )
    print("===============================================\n")

    print("RAW PREDICTIONS:")
    debug_anomaly_detection_delay(raw_pred, labels)

    print("ADJUSTED PREDICTIONS:")
    debug_anomaly_detection_delay(adjusted_pred, labels)

def compute_range_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """Compute Range-AUC and VUS metrics.

    Returns NaN when the external VUS implementation fails so that failures
    cannot be mistaken for valid zero-valued metrics.
    """
    failed = {
        "rauc_roc": float("nan"),
        "rauc_pr": float("nan"),
        "vus_roc": float("nan"),
        "vus_pr": float("nan"),
    }

    if not _HAS_VUS or get_metrics is None:
        print(
            "VUS metrics unavailable: get_metrics could not be imported. "
            "Check the VUS package and import path."
        )
        return failed

    try:
        y_true = np.asarray(labels, dtype=int).reshape(-1)
        y_scores = np.asarray(scores, dtype=float).reshape(-1)

        if len(y_true) != len(y_scores):
            raise ValueError(
                f"labels length={len(y_true)}, scores length={len(y_scores)}"
            )

        if not np.all(np.isfinite(y_scores)):
            raise ValueError("scores contain NaN or infinite values")

        diff = np.diff(
            np.concatenate(
                [
                    np.array([0], dtype=int),
                    y_true,
                    np.array([0], dtype=int),
                ]
            )
        )
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths = ends - starts

        sliding_window = (
            max(int(np.median(lengths)), 16)
            if len(lengths) > 0
            else 100
        )

        print(
            "Computing VUS metrics: "
            f"length={len(y_true)}, "
            f"anomalies={int(y_true.sum())}, "
            f"segments={len(starts)}, "
            f"sliding_window={sliding_window}"
        )

        metrics = get_metrics(
            score=y_scores,
            labels=y_true,
            metric="all",
            slidingWindow=sliding_window,
        )

        print(
            f"VUS get_metrics returned type={type(metrics).__name__}: "
            f"{metrics}"
        )

        if not isinstance(metrics, dict):
            raise TypeError(
                "get_metrics did not return a dictionary; "
                f"received {type(metrics).__name__}"
            )

        required = [
            "R_AUC_ROC",
            "R_AUC_PR",
            "VUS_ROC",
            "VUS_PR",
        ]
        missing = [key for key in required if key not in metrics]

        if missing:
            raise KeyError(
                f"Missing VUS result keys: {missing}. "
                f"Available keys: {list(metrics.keys())}"
            )

        return {
            "rauc_roc": float(metrics["R_AUC_ROC"]),
            "rauc_pr": float(metrics["R_AUC_PR"]),
            "vus_roc": float(metrics["VUS_ROC"]),
            "vus_pr": float(metrics["VUS_PR"]),
        }

    except Exception as exc:
        import traceback

        print(f"Error computing VUS metrics: {exc}")
        traceback.print_exc()
        return failed



def precision_recall_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    adjust: bool = False,
) -> PRF:
    """Compute P/R/F1 for ``scores > threshold`` under the chosen protocol."""
    pred = (scores > threshold).astype(int)
    if adjust:
        pred = point_adjust(pred, labels)

    labels = labels.astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision, recall, f1 = _prf_from_counts(tp, fp, fn)
    return PRF(precision, recall, f1, float(threshold))


def _candidate_thresholds(scores: np.ndarray, num: int = 400) -> np.ndarray:
    """Quantile-spaced candidate thresholds (robust to score outliers)."""
    qs = np.linspace(0.0, 1.0, num)
    cand = np.quantile(scores, qs)
    cand = np.unique(cand)
    # Nudge slightly so that the extreme thresholds behave sensibly.
    eps = 1e-12
    return np.concatenate([[scores.min() - eps], cand, [scores.max() + eps]])


def best_f1_search(
    scores: np.ndarray,
    labels: np.ndarray,
    adjust: bool = False,
    num_thresholds: int = 400,
) -> PRF:
    """Search the threshold that maximises F1 under the chosen protocol.

    This is the (oracle) "best-F1" number commonly reported in the literature.
    It peeks at the test labels and is therefore optimistic; we report it for
    comparability and additionally report label-free thresholds elsewhere.
    """
    candidates = _candidate_thresholds(scores, num_thresholds)
    best = PRF(0.0, 0.0, -1.0, candidates[0])
    for thr in candidates:
        res = precision_recall_f1(scores, labels, thr, adjust=adjust)
        if res.f1 > best.f1:
            best = res
    return best


def auroc_auprc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Threshold-free ranking metrics. AUPRC matters most for rare anomalies."""
    labels = labels.astype(int)
    if labels.min() == labels.max():  # degenerate (all normal / all anomalous)
        return float("nan"), float("nan")
    if _HAS_SKLEARN:
        return (
            float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)),
        )
    return _auroc_fallback(scores, labels), _auprc_fallback(scores, labels)


def _auroc_fallback(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _auprc_fallback(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(labels.sum(), 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def threshold_by_anomaly_ratio(
    scores: np.ndarray, anomaly_ratio: float
) -> float:
    """Threshold at the (1 - ratio) quantile of the scores (label-free)."""
    anomaly_ratio = float(np.clip(anomaly_ratio, 1e-4, 0.5))
    return float(np.quantile(scores, 1.0 - anomaly_ratio))


def pot_threshold(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    q: float = 1e-3,
    level: float = 0.98,
) -> float:
    """Peaks-Over-Threshold (POT) threshold via a Generalised Pareto fit.

    Fits a GPD to the upper-tail excesses of ``train_scores`` (normal data) and
    extrapolates a threshold that controls the tail probability ``q`` on
    ``test_scores``. This is fully label-free.
    """
    try:
        from scipy.stats import genpareto  # local import keeps scipy optional
    except Exception:
        # Fallback: simple high quantile of the normal scores.
        return float(np.quantile(train_scores, 1.0 - q))

    init_t = np.quantile(train_scores, level)
    excess = train_scores[train_scores > init_t] - init_t
    if len(excess) < 10:
        return float(np.quantile(train_scores, 1.0 - q))

    c, _, scale = genpareto.fit(excess, floc=0.0)
    n = len(train_scores)
    nt = len(excess)
    if abs(c) < 1e-6:
        thr = init_t - scale * np.log(q * n / nt)
    else:
        thr = init_t + (scale / c) * ((q * n / nt) ** (-c) - 1.0)
    return float(thr)


def evaluate_all(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    val_scores: Optional[np.ndarray] = None,
    anomaly_ratio: float = 0.05,
    pot_q: float = 1e-3,
    pot_level: float = 0.98,
    debug: bool = False,
) -> Dict[str, Dict]:
    """Run the complete anomaly-detection evaluation suite.

    Evaluation protocol
    -------------------
    - Pointwise best-F1 threshold is selected by maximizing raw pointwise F1.
    - Adjusted best-F1 threshold is selected independently by maximizing
      point-adjusted F1.
    - Ratio and POT thresholds are shared because they are selected without
      optimizing either protocol.
    - ADD is always calculated from raw, unadjusted predictions at the
      corresponding threshold.

    Therefore:

        results["pointwise"]["best_f1"]["add"]

    is raw ADD at the pointwise-optimal threshold, while:

        results["adjusted"]["best_f1"]["add"]

    is raw ADD at the adjusted-F1-optimal threshold.

    Point adjustment is used only for adjusted precision, recall, and F1.
    """
    test_scores = np.asarray(test_scores, dtype=float).reshape(-1)
    test_labels = np.asarray(test_labels, dtype=int).reshape(-1)

    if test_scores.shape != test_labels.shape:
        raise ValueError(
            "Test score and label shapes differ: "
            f"{test_scores.shape} versus {test_labels.shape}"
        )

    if not np.all(np.isfinite(test_scores)):
        raise ValueError("test_scores contains NaN or infinite values")

    if not np.all(np.isin(test_labels, [0, 1])):
        raise ValueError("test_labels must contain only binary values 0 and 1")

    if val_scores is not None:
        val_scores = np.asarray(val_scores, dtype=float).reshape(-1)

        if not np.all(np.isfinite(val_scores)):
            raise ValueError("val_scores contains NaN or infinite values")

    # Threshold-free metrics.
    auroc, auprc = auroc_auprc(
        test_scores,
        test_labels,
    )

    range_metrics = compute_range_metrics(
        test_labels,
        test_scores,
    )

    results: Dict[str, Dict] = {
        "auroc": auroc,
        "auprc": auprc,
        **range_metrics,
    }

    # Label-free anomaly-ratio threshold.
    ratio_thr = threshold_by_anomaly_ratio(
        test_scores,
        anomaly_ratio,
    )

    # POT is fitted using validation scores when available.
    threshold_source = (
        val_scores
        if val_scores is not None
        else test_scores
    )

    pot_thr = pot_threshold(
        threshold_source,
        test_scores,
        q=pot_q,
        level=pot_level,
    )

    for protocol, adjust in (
        ("pointwise", False),
        ("adjusted", True),
    ):
        # Independent oracle best-F1 search for each protocol.
        best_f1_result = best_f1_search(
            test_scores,
            test_labels,
            adjust=adjust,
        )

        ratio_result = precision_recall_f1(
            test_scores,
            test_labels,
            ratio_thr,
            adjust=adjust,
        )

        pot_result = precision_recall_f1(
            test_scores,
            test_labels,
            pot_thr,
            adjust=adjust,
        )

        best_f1_dict = best_f1_result.as_dict()
        ratio_dict = ratio_result.as_dict()
        pot_dict = pot_result.as_dict()

        best_f1_dict.update(
            add_values_at_threshold(
                test_scores,
                test_labels,
                best_f1_result.threshold,
            )
        )

        ratio_dict.update(
            add_values_at_threshold(
                test_scores,
                test_labels,
                ratio_result.threshold,
            )
        )

        pot_dict.update(
            add_values_at_threshold(
                test_scores,
                test_labels,
                pot_result.threshold,
            )
        )

        results[protocol] = {
            "best_f1": best_f1_dict,
            "ratio": ratio_dict,
            "pot": pot_dict,
        }

        if debug:
            raw_pred = (
                test_scores > best_f1_result.threshold
            ).astype(int)

            print(
                f"\n{'=' * 70}\n"
                f"{protocol.upper()} BEST-F1 EVALUATION\n"
                f"Threshold : {best_f1_result.threshold:.10f}\n"
                f"Precision : {best_f1_result.precision:.6f}\n"
                f"Recall    : {best_f1_result.recall:.6f}\n"
                f"F1        : {best_f1_result.f1:.6f}\n"
                f"{'=' * 70}"
            )

            print(
                "\nRAW PREDICTION ADD "
                f"AT {protocol.upper()}-OPTIMAL THRESHOLD:"
            )
            debug_anomaly_detection_delay(
                raw_pred,
                test_labels,
            )

            if adjust:
                adjusted_pred = point_adjust(
                    raw_pred.copy(),
                    test_labels,
                )

                print(
                    "\nPOINT-ADJUSTED PREDICTIONS "
                    "(for verification only):"
                )
                debug_anomaly_detection_delay(
                    adjusted_pred,
                    test_labels,
                )

    return results