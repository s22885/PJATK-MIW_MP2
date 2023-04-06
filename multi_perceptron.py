import numpy as np
from perceptron import Perceptron


class MultiPerceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.n_classes = np.unique(y)
        self.perceptrons = dict()

        for class_val in self.n_classes:
            y_coded = np.where(y == class_val, 1, -1)
            ptron = Perceptron(self.eta, self.n_iter)
            ptron.fit(X, y_coded)
            self.perceptrons[class_val] = ptron

        return self

    def predict(self, X):
        labels = []
        preds = []
        for k, ptron in self.perceptrons.items():
            ptron_preds = ptron.net_input(X)
            preds.append(ptron_preds)
            labels.append(k)
        preds_id = np.argmax(np.array(preds), axis=0)
        return np.array([labels[pred_id] for pred_id in preds_id])
