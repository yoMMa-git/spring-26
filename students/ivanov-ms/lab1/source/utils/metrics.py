import numpy as np
import pandas as pd
import warnings
from scipy.integrate import trapezoid


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return pd.DataFrame(
        [[tp, fp], [fn, tn]],
        columns=pd.MultiIndex.from_tuples([("Actual", "Positive"), ("Actual", "Negative")]),
        index=pd.MultiIndex.from_tuples([("Predict", "Positive"), ("Predict", "Negative")])
    )


def precision_score(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def roc_auc(fpr, tpr):
    area = trapezoid(tpr, fpr)
    return float(area)


def roc_curve(y_true, y_score):
    y_true = y_true == 1

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    if fpr[0] > 0:
        fpr = np.concat([[0], fpr], axis=0)
        tpr = np.concat([[0], tpr], axis=0)
    if fpr[-1] < 1:
        fpr = np.concat([fpr, [1]], axis=0)
        tpr = np.concat([tpr, [1]], axis=0)

    return fpr, tpr


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if y_scores is not None:
        auc_val = roc_auc(*roc_curve(y_true, y_scores))
    else:
        auc_val = None

    return accuracy, precision, recall, f1, auc_val


def eval_model(model, X_test: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        if y_scores is not None and len(y_scores.shape) > 1 and y_scores.shape[1] == 2:
            y_scores = y_scores[:, 1]
    return y_pred, y_scores


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, log_prefix: str = ""):
    # Sklearn models prints some RuntimeWarning-s, disable them
    y_pred, y_scores = eval_model(model, X_test)

    accuracy, precision, recall, f1, auc_val = get_metrics(y_test, y_pred, y_scores)
    print(f"{log_prefix}Accuracy: {accuracy:.4f}")
    print(f"{log_prefix}Precision: {precision:.4f}")
    print(f"{log_prefix}Recall: {recall:.4f}")
    print(f"{log_prefix}F1-Score: {f1:.4f}")
    if auc_val is not None:
        print(f"{log_prefix}AUC-ROC: {auc_val:.4f}")

    return confusion_matrix(y_test, y_pred)
