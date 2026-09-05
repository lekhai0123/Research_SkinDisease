import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support


def compute_full_metrics(y_true, y_pred, num_classes, prefix=""):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels_idx = list(range(num_classes))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
    )

    return {
        f"{prefix}acc": float((y_true == y_pred).mean()),
        f"{prefix}balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1),
        f"{prefix}macro_precision": float(prec),
        f"{prefix}macro_recall": float(rec),
        f"{prefix}weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0)
        ),
    }
