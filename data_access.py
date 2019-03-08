import numpy as np
import pandas as pd

from sklearn import datasets


def load_iris_data():
    TARGET_COLUMN = 'int_class'
    iris = datasets.load_iris()
    n = iris.target.shape[0]
    merged_data = np.append(iris.data, iris.target.reshape((n, 1)), axis=1)
    column_names = iris.feature_names[:]
    column_names.append(TARGET_COLUMN)
    df = pd.DataFrame(data=merged_data, columns=column_names)
    df[[TARGET_COLUMN]] = df[[TARGET_COLUMN]].astype(int)
    return df
