import numpy as np
from sklearn.metrics import roc_auc_score


def classification_metrics(labels, predictions, classes):
    counts = np.bincount(labels, minlength=classes)
    correct = np.bincount(labels[labels == predictions], minlength=classes)
    per_class = np.divide(correct, counts, out=np.full(classes, np.nan), where=counts > 0)
    return {
        "top1": float(np.mean(labels == predictions)),
        "worst_class_accuracy": float(np.nanmin(per_class)),
        "per_class_accuracy": [None if np.isnan(value) else float(value) for value in per_class],
    }


def detection_metrics(observed, reference, scores, classes, threshold):
    clean = observed == reference
    selected = scores >= threshold
    true_positive = int(np.count_nonzero(clean & selected))
    selected_count = int(np.count_nonzero(selected))
    clean_count = int(np.count_nonzero(clean))
    recalls = []
    selected_counts = []
    for label in range(classes):
        members = observed == label
        positives = int(np.count_nonzero(members & clean))
        recalls.append(float(np.count_nonzero(members & clean & selected) / positives) if positives else None)
        selected_counts.append(int(np.count_nonzero(members & selected)))
    return {
        "auroc": float(roc_auc_score(clean, scores)) if np.unique(clean).size == 2 else None,
        "clean_precision": true_positive / selected_count if selected_count else None,
        "clean_recall": true_positive / clean_count if clean_count else None,
        "selected_fraction": selected_count / len(scores),
        "per_class_clean_recall": recalls,
        "per_class_selected_count": selected_counts,
        "threshold": threshold,
    }
