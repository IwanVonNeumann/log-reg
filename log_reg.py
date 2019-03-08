import numpy as np


class LogisticRegression(object):

    def __init__(self, learning_rate=0.01, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        n, m = X.shape

        self.w_ = np.zeros(shape=(m, 1))
        self.b_ = 0

        self.cost_history_ = []

        for _ in range(self.n_iter):
            a = self.activation(X)
            self.w_ += self.learning_rate * X.T.dot(y - a)
            self.b_ += self.learning_rate * np.sum(y - a)

            cost = self.cost(X, y)
            self.cost_history_.append(cost)

        return self

    def net_input(self, X):
        return X.dot(self.w_) + self.b_

    def activation(self, X):
        z = self.net_input(X)
        return 1 / (1 + np.exp(-z))

    def cost(self, X, y):
        return -self.log_likelihood(X, y)

    def log_likelihood(self, X, y):
        a = self.activation(X)
        return (y * np.log(a) + (1 - y) * np.log(1 - a)).sum()

    def predict(self, X, t=0.5):
        a = self.activation(X)
        return np.where(a >= t, 1, 0)

    def predict_probs(self, X):
        return self.activation(X)
