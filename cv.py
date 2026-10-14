"""Class balancing, development holdout, and cross-validation utilities."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def make_splits(y, seed=42, dev_size=13, n_splits=10):
    """Balance classes by undersampling, hold out dev examples, create CV folds."""
    rng = np.random.default_rng(seed)
    heavy_idx = np.where(y == 1)[0]
    light_idx = rng.choice(np.where(y == 0)[0], len(heavy_idx), replace=False)
    idx = np.concatenate((heavy_idx, light_idx))
    rng.shuffle(idx)
    cv_idx, dev_idx = train_test_split(
        idx, test_size=dev_size, stratify=y[idx], random_state=seed
    )
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 .split(cv_idx, y[cv_idx]))
    return cv_idx, dev_idx, folds

def cross_validate(fit_predict, name, cv_idx, folds, y, cols=None):
    """fit_predict(train_indices, test_indices, cols) returns train/test predictions."""
    train_acc, test_acc, train_f1, test_f1, precision, recall = [], [], [], [], [], []
    for train_local, test_local in folds:
        train_idx, test_idx = cv_idx[train_local], cv_idx[test_local]
        pred_train, pred_test = fit_predict(train_idx, test_idx, cols)
        train_acc.append(accuracy_score(y[train_idx], pred_train))
        test_acc.append(accuracy_score(y[test_idx], pred_test))
        train_f1.append(f1_score(y[train_idx], pred_train, zero_division=0))
        test_f1.append(f1_score(y[test_idx], pred_test, zero_division=0))
        precision.append(precision_score(y[test_idx], pred_test, zero_division=0))
        recall.append(recall_score(y[test_idx], pred_test, zero_division=0))
    return {"model": name, "train_acc": np.mean(train_acc), "train_f1": np.mean(train_f1),
            "test_acc": np.mean(test_acc), "test_f1": np.mean(test_f1),
            "test_precision": np.mean(precision), "test_recall": np.mean(recall)}
