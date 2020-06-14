import os
import uuid

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from performance_anomaly_detection.anomaly_detection.detector import GaussianTailDetector, AccumulativeDetector, \
    DiffType
from performance_anomaly_detection.result_generator.train import ModelTrainer, ExperimentDataType, ExperimentModel
from performance_anomaly_detection.training.utils import plot_results_scores


class ExperimentData:

    def __init__(self, metrics_data_file, spans_data_file, traces_data_file):
        self.metrics_data = pd.read_csv(metrics_data_file)
        self.spans_data = pd.read_csv(spans_data_file)
        self.traces_data = pd.read_csv(traces_data_file)


class Experiment:

    def __init__(self, experiment_data: ExperimentData, results_dir):
        self.id = uuid.uuid4()
        self.experiment_data = experiment_data
        self.models = list()
        self.model_trainer = ModelTrainer()
        self.gt_detector = GaussianTailDetector(window_size=100, small_window_size=5, e=0.3)
        self.acc_detector = AccumulativeDetector(diff_type=DiffType.down)
        self.results_path = os.path.join(results_dir, str(self.id))
        os.mkdir(self.results_path)

    def add_model(self, model: ExperimentModel):
        self.models.append(model)

    def run(self):
        for model in self.models:
            self.create_model_dir(model.name)
            self.train_model(self.experiment_data.metrics_data, ExperimentDataType.metrics, model)
            #self.train_model(self.experiment_data.spans_data, ExperimentDataType.spans, model)
            #self.train_model(self.experiment_data.traces_data, ExperimentDataType.traces, model)

    def train_model(self, data, data_type, experiment_model: ExperimentModel):
        if not experiment_model.trained:
            y, y_hat, scores = self.model_trainer.train(data, data_type, experiment_model)
        else:
            y, y_hat, scores = self.model_trainer.predict(data, data_type, experiment_model)
        self.save_anomalies(y, y_hat, experiment_model.name, data_type)

        model_results_path = os.path.join(self.results_path, experiment_model.name)
        model_dump = os.path.join(model_results_path, str(data_type) + "_model")

        if not experiment_model.trained:
            experiment_model.save_model(model_dump)
        fig_path = os.path.join(model_results_path, str(data_type) + "_results.png")
        plot_results_scores(y, y_hat, scores, fig_path)

    def save_anomalies(self, y, y_hat, model_name, data_type):
        acc_anomalies = []
        gaussian_tail_anomalies = []
        for i in range(len(y_hat)):
            self.gt_detector.add_values(y[i], y_hat[i])
            self.acc_detector.add_values(y[i], y_hat[i])
            if self.gt_detector.is_anomaly():
                gaussian_tail_anomalies.append(i)
            if self.acc_detector.is_anomaly():
                acc_anomalies.append(i)

        gaussian_markers = list(np.array(gaussian_tail_anomalies).flatten())
        acc_markers = list(np.array(acc_anomalies).flatten())
        intersec_markers = list(np.intersect1d(acc_anomalies, gaussian_tail_anomalies).flatten())
        self.gt_detector.reset()
        self.acc_detector.reset()
        self.plot_anomalies(y, gaussian_markers, acc_markers, intersec_markers, model_name, data_type)

    def plot_anomalies(self, y, gaussian_markers, acc_markers, intersec_markers, model_name, data_type):
        fig, axs = plt.subplots(3)
        fig.set_size_inches(18.5, 10.5, forward=True)
        fig.suptitle('Anomalies')
        self.plot_markers(axs[0], "Gaussian Anomalies", y, gaussian_markers)
        self.plot_markers(axs[1], "Acc Anomalies", y, acc_markers)
        self.plot_markers(axs[2], "Intersected Anomalies", y, intersec_markers)
        model_results_path = os.path.join(self.results_path, model_name)
        fig_path = os.path.join(model_results_path, str(data_type) + "_anomalies.png")
        fig.savefig(fig_path, dpi=fig.dpi)

    def plot_markers(self, ax, title, y, markers):
        ax.set_title(title)
        ax.plot(range(len(y)), y, color='blue')
        ax.plot(range(len(y)), y, 'go', markevery=markers)

    def create_model_dir(self, model_name):
        model_results_path = os.path.join(self.results_path, model_name)
        os.mkdir(model_results_path)
        return model_results_path
