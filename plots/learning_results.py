import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def plot_error_history(history, plot_settings):
    epoch_numbers = range(1, len(history) + 1)
    plt.plot(epoch_numbers, history, marker='o')
    plt.title(plot_settings['title'])
    plt.xlabel(plot_settings['xlabel'])
    plt.ylabel(plot_settings['ylabel'])
    plt.show()


def plot_2d_decision_boundary(X, y, classifier, X_test=None, y_test=None, resolution=0.1,
                              colors=('blue', 'red'), markers=('x', 'o'), plot_settings=None):
    cmap = LinearSegmentedColormap.from_list(
        name='custom_colormap',
        colors=[colors[0], 'white', colors[1]],
        N=100
    )

    X = np.append(X, X_test, axis=0)
    y = np.append(y, y_test, axis=0)

    b = 0.5
    x1_min, x1_max = X[:, 0].min() - b, X[:, 0].max() + b
    x2_min, x2_max = X[:, 1].min() - b, X[:, 1].max() + b

    xx1, xx2 = np.meshgrid(
        np.arange(start=x1_min, stop=x1_max, step=resolution),
        np.arange(start=x2_min, stop=x2_max, step=resolution)
    )

    Z = classifier.predict_probs(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.pcolormesh(xx1, xx2, Z, alpha=0.25, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for i, label in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == label, 0],
            y=X[y == label, 1],
            c=colors[i], marker=markers[i], label=label
        )

    if X_test is not None and y_test is not None:
        plt.scatter(
            x=X_test[:, 0],
            y=X_test[:, 1],
            color='', marker='o', s=100, edgecolors='black', linewidth=1.5, label='test'
        )

    plt.legend(loc="upper left")

    plt.title(plot_settings['title'])
    plt.xlabel(plot_settings['xlabel'])
    plt.ylabel(plot_settings['ylabel'])

    plt.show()
