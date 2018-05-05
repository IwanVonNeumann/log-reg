import matplotlib.pyplot as plt


def plot_error_history(history, plot_settings):
    epoch_numbers = range(1, len(history) + 1)
    plt.plot(epoch_numbers, history, marker="o")
    plt.title(plot_settings["title"])
    plt.xlabel(plot_settings["xlabel"])
    plt.ylabel(plot_settings["ylabel"])
    plt.show()
