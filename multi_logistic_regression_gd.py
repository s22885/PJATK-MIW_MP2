import numpy as np

from logistic_regression_gd import LogisticRegressionGD


class MultiLogisticRegressorGD:
    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

        self.n_classes = ()
        self.regressors = dict()

    def fit(self, X, y):
        self.n_classes = np.unique(y)
        self.regressors = dict()

        for class_val in self.n_classes:
            y_coded = np.where(y == class_val, 1, 0)
            regressor = LogisticRegressionGD(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state)
            regressor.fit(X, y_coded)
            self.regressors[class_val] = regressor

        return self

    def _predict(self, X):
        labels = []
        preds = []
        for k, regressor in self.regressors.items():
            regressor_preds = regressor.net_input(X)
            regressor_preds = np.array([regressor.activation(net) for net in regressor_preds])
            preds.append(regressor_preds)
            labels.append(k)
        return labels, preds

    def predict(self, X):
        labels, preds = self._predict(X)
        preds_id = np.argmax(np.array(preds), axis=0)
        return np.array([labels[pred_id] for pred_id in preds_id])

    def predict_proba(self, X):
        _, preds = self._predict(X)
        return np.transpose(preds)

