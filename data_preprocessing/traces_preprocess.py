import pandas as pd


class TracesPreprocessor:

    def __init__(self, tracing_data_file):
        self.tracing_data_file = tracing_data_file
        self.tracing_data = pd.read_csv(tracing_data_file, sep=",", encoding="utf-8")

    def minimize_start_time(self, value_for_minimization):
        self.tracing_data["start_time"] = self.tracing_data["start_time"] - value_for_minimization

    def create_ids_map(self, column):
        unique_trace_ids = self.tracing_data[column].unique()
        ids_map = {}
        for i in range(len(unique_trace_ids)):
            ids_map[unique_trace_ids[i]] = i
        return ids_map

    def map_ids(self, column, ids_map):
        self.tracing_data[column] = self.tracing_data[column].map(ids_map)

    def apply_lambda_to_column(self, column, lambda_func):
        return self.tracing_data[column].apply(lambda_func)

    def apply_lambda_to_column_overwrite(self, column, lambda_func):
        self.tracing_data[column] = self.apply_lambda_to_column(column, lambda_func)

    def sort_values(self):
        self.tracing_data = self.tracing_data.sort_values(by="start_time")

    def create_end_time(self):
        self.tracing_data["end_time"] = self.tracing_data["start_time"] + self.tracing_data["duration"]

    def drop_columns(self, columns):
        self.tracing_data = self.tracing_data.drop(columns, axis=1)

    def drop_single_column(self, column):
        self.drop_columns([column])

    def get_data(self):
        return self.tracing_data

    def write_data(self, output_file):
        self.tracing_data.to_csv(output_file, sep=",")
