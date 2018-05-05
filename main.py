import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_access import load_iris_data
from learning_results import plot_error_history
from log_reg import LogisticRegression

iris_df = load_iris_data()

predictor_columns = ["petal length (cm)", "petal width (cm)"]
target_column = "int class"

set_ver_df = iris_df[iris_df["int class"].isin([0, 1])].copy()

X = set_ver_df[predictor_columns].values
y = set_ver_df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train_std, y_train)

n_test = y_test.shape[0]
y_test = y_test.reshape(n_test, 1)
y_prob = log_reg.predict_probs(X_test_std).reshape(n_test, 1)
y_pred = log_reg.predict(X_test_std).reshape(n_test, 1)

predictions = np.concatenate((y_prob, y_pred, y_test), axis=1)
predictions_df = pd.DataFrame(predictions, columns=["predicted prob", "predicted class", "class"])

print("Classification results:")
print(predictions_df)
print()

print("Classification accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))

error_plot_settings = {
    "title": "LogReg Gradient Ascent",
    "xlabel": "Epochs",
    "ylabel": "log likelihood"
}

plot_error_history(log_reg.log_lik_history_, error_plot_settings)
