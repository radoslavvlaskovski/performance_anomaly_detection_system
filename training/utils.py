import time

import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import statistics


def save_results(predictions, real_ys, results_output, label_data):
    results = pd.concat(predictions, axis=1)
    real_ys = pd.concat(real_ys, axis=1)
    if label_data is not None:
        results = pd.concat((label_data.reset_index(drop=True), results.reset_index(drop=True)), axis=1)
        real_ys = pd.concat((label_data.reset_index(drop=True), real_ys.reset_index(drop=True)), axis=1)

    results.to_csv(results_output + "_results_" + str(time.time()) + ".csv", columns=list(results.columns), index=False)
    real_ys.to_csv(results_output + "_results_" + str(time.time()) + ".csv", columns=list(real_ys.columns), index=False)


def plot_results_scores(y, y_hat, scores, fig_path):
    fig, ax = plt.subplots()
    textstr = '\n'.join((scores[0][0],
                         scores[0][1],
                         scores[1][0],
                         scores[1][1]))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.85, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    ax.plot(y)
    ax.plot(y_hat, c="red")
    ax.set_title("Real and Prediction Comparison")
    ax.set_xlabel("Time")
    ax.legend(["Real", "Prediction"], loc="upper left")
    fig.savefig(fig_path, dpi=fig.dpi)


def plot_results(y, y_hat):
    plt.figure(figsize=(14, 8))
    plt.plot(y)
    plt.plot(y_hat, c="red")
    plt.title("Real and Prediction Comparison")
    plt.xlabel("Time")
    plt.legend(["Real", "Prediction"], loc="upper left")
    plt.show()


def plot_loss(history):
    plt.figure(figsize=(14, 8))
    plt.plot(history.history["loss"], marker="o")
    plt.plot(history.history["val_loss"], marker="o")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Dev"], loc="upper left")
    plt.xticks(range(0, len(history.history["loss"])))
    plt.show()


def calculate_rmse(global_scores, test_scores):
    return "Final RMSE: {0:.2f}".format(sqrt(statistics.mean(global_scores))),\
           "Final Test RMSE: {0:.2f}".format(sqrt(statistics.mean(test_scores)))


def calculate_r2(train_rsq, test_rsq):
    return "Final R2: {0:.2f}".format(statistics.mean(train_rsq)),\
           "Final Test R2: {0:.2f}".format(statistics.mean(test_rsq))
