# metrics.py
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

def fbeta_macro(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5, num_classes: int = 3) -> float:
    """
    多分类 macro-Fbeta
    """
    beta2 = beta * beta
    f_list = []
    for k in range(num_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))

        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-12)
        f_list.append(f)
    return float(np.mean(f_list))
