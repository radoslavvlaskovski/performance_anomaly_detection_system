import pandas as pd


class MetricsPreprocessor:

    def __init__(self, metrics_data_file):
        self.metrics_data_file = metrics_data_file
        self.metrics_data = pd.read_csv(metrics_data_file, sep=",")

    def normalize_time(self):
        self.metrics_data["time"] *= (10 ** 6)
        self.metrics_data = self.metrics_data.astype({'time': 'int64'})

    def minimize_start_time(self, value_for_minimization):
        self.metrics_data["time"] = self.metrics_data["time"] - value_for_minimization

    def clean_column_names(self):
        new_cols = {}
        for col in self.metrics_data.columns:
            if "container_network_" in col:
                new_cols[col] = col.replace("container_network_", "").replace("_total_value", "")
        self.metrics_data = self.metrics_data.rename(columns=new_cols)

    def decumulate_data(self):
        for col in self.metrics_data.columns:
            if col != "time":
                self.metrics_data[col] = self.metrics_data[col] - self.metrics_data[col].shift(1)
        self.metrics_data = self.metrics_data.iloc[1:]

    def get_data(self):
        return self.metrics_data

    def write_data(self, output_file):
        self.metrics_data.to_csv(output_file, sep=",")
