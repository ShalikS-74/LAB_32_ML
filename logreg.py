"""Logistic regression trained with Newton's method."""
import numpy as np

def newton_logreg(X, y, eps=1e-5, max_iter=50):
    Xb = np.column_stack((np.ones(len(X)), X))
    theta = np.zeros(Xb.shape[1])
    for iteration in range(1, max_iter + 1):
        scores = np.clip(Xb @ theta, -500, 500)
        p = 1 / (1 + np.exp(-scores))
        grad = Xb.T @ (p - y)
        hessian = Xb.T @ (Xb * (p * (1 - p))[:, None]) + 1e-8 * np.eye(Xb.shape[1])
        delta = np.linalg.solve(hessian, grad)
        theta -= delta
        if delta @ delta < eps:
            break
    return theta, iteration

def training_mean_std(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return mean, np.where(std == 0, 1.0, std)

def make_lr_fit_predict(X_all, y, feature_cols):
    def fit_predict(train_idx, test_idx, cols=None):
        selected = feature_cols if cols is None else cols
        mean, std = training_mean_std(X_all[train_idx][:, selected])
        X_train = (X_all[train_idx][:, selected] - mean) / std
        X_test = (X_all[test_idx][:, selected] - mean) / std
        theta, _ = newton_logreg(X_train, y[train_idx])
        def predict(X):
            scores = np.column_stack((np.ones(len(X)), X)) @ theta
            return (scores > 0).astype(int)
        return predict(X_train), predict(X_test)
    return fit_predict
