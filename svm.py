"""SVM model factories for the four kernels used in the experiment."""
from sklearn.svm import SVC
from .logreg import training_mean_std

def make_svm_fit_predict(X_all, y, feature_cols, kernel="rbf", degree=2):
    def fit_predict(train_idx, test_idx, cols=None):
        selected = feature_cols if cols is None else cols
        mean, std = training_mean_std(X_all[train_idx][:, selected])
        X_train = (X_all[train_idx][:, selected] - mean) / std
        X_test = (X_all[test_idx][:, selected] - mean) / std
        model = SVC(kernel=kernel, degree=degree)
        model.fit(X_train, y[train_idx])
        return model.predict(X_train), model.predict(X_test)
    return fit_predict
