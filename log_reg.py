import numpy as np


class LogisticRegression(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X_ = self._prepend_unit_column(X)
        self.w_ = np.zeros(X_.shape[1])
        self.log_lik_history_ = []

        for _ in range(self.n_iter):
            phi = self.activation(X_)
            dw = self.eta * X_.T.dot(y - phi)
            self.w_ += dw

            log_lik = self.log_likelihood(X_, y)
            self.log_lik_history_.append(log_lik)

        return self

    def activation(self, X):
        """Compute sigmoid activation"""
        return 1 / (1 + np.exp(-self.net_input(X)))

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_)

    def log_likelihood(self, X, y):
        z = np.dot(X, self.w_)
        phi = 1 / (1 + np.exp(-z))
        return y.dot(np.log(phi)) + (1 - y).dot(np.log(1 - phi))

    def predict(self, X):
        """Return class label after unit step"""
        X_ = self._prepend_unit_column(X)
        return np.where(self.activation(X_) >= 0.5, 1, 0)

    def predict_probs(self, X):
        """Return positive class label probabilities"""
        X_ = self._prepend_unit_column(X)
        return self.activation(X_)

    @staticmethod
    def _prepend_unit_column(X):
        h, _ = X.shape
        unit_column = np.full((h, 1), 1)
        return np.append(unit_column, X, axis=1)
