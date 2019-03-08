import matplotlib.pyplot as plt

from plots.utils import iris_plot_settings


def classes_scatter_plot(iris_df):
    TARGET_COLUMN = 'int_class'

    class_labels = iris_df[TARGET_COLUMN].unique()
    split_df = {label: iris_df[iris_df[TARGET_COLUMN] == label] for label in class_labels}

    for label in class_labels:
        df = split_df[label]
        plot_settings = iris_plot_settings[label]
        plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], **plot_settings)

    plt.title('Iris flowers classes')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()
