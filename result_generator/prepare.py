import os

import pandas as pd

from performance_anomaly_detection.data_preprocessing.data_combiner import combine_on_metrics, combine_on_spans, \
    combine_on_traces
from performance_anomaly_detection.data_preprocessing.metrics_preprocess import MetricsPreprocessor
from performance_anomaly_detection.data_preprocessing.traces_preprocess import TracesPreprocessor


class DataPrepare:

    def __init__(self, tracing_data_file, metrics_data_file):
        self.initial_start_time = 1578330000000000
        self.tracing_data_file = tracing_data_file
        self.metrics_data_file = metrics_data_file
        self.tracing_data = None
        self.metrics_data = None

    def prepare_tracing_data(self):
        traces_preprocessor = TracesPreprocessor(self.tracing_data_file)
        traces_preprocessor.minimize_start_time(self.initial_start_time)

        trace_ids_map = traces_preprocessor.create_ids_map("trace_id")
        span_ids_map = traces_preprocessor.create_ids_map("span_id")
        traces_preprocessor.map_ids("trace_id", trace_ids_map)
        traces_preprocessor.map_ids("span_id", span_ids_map)

        clean_operation_column_lambda = lambda s: s.replace(".hipstershop", "").replace("./", ".").replace("/", ".")
        traces_preprocessor.apply_lambda_to_column_overwrite("operation_name", clean_operation_column_lambda)

        process_extract_lambda = self.create_lambda_for_cutting_start_end("process(service_name='", "', tags=[])")
        traces_preprocessor.apply_lambda_to_column_overwrite("process", process_extract_lambda)

        ref_type_extract_lambda = self.create_lambda_for_cutting_start_end("ref_type='", "', trace_id=")
        traces_preprocessor.get_data()["ref_type"] = traces_preprocessor.apply_lambda_to_column("refs",
                                                                                                ref_type_extract_lambda)
        ref_trace_extract_lambda = self.create_lambda_for_cutting_start_end(", trace_id=", ", span_id=")
        traces_preprocessor.get_data()["ref_trace"] = traces_preprocessor.apply_lambda_to_column("refs",
                                                                                                 ref_trace_extract_lambda)
        traces_preprocessor.map_ids("ref_trace", trace_ids_map)
        ref_trace_extract_lambda = self.create_lambda_for_cutting_start_end(", span_id=", ", span_id=")
        traces_preprocessor.get_data()["ref_span"] = traces_preprocessor.apply_lambda_to_column("refs",
                                                                                                ref_trace_extract_lambda)
        traces_preprocessor.map_ids("ref_span", span_ids_map)

        traces_preprocessor.drop_single_column("refs")
        traces_preprocessor.sort_values()
        traces_preprocessor.create_end_time()
        self.tracing_data = traces_preprocessor.get_data()

    def prepare_metrics_data(self):
        metrics_preprocessor = MetricsPreprocessor(self.metrics_data_file)
        metrics_preprocessor.normalize_time()
        metrics_preprocessor.minimize_start_time(self.initial_start_time)
        metrics_preprocessor.decumulate_data()
        metrics_preprocessor.clean_column_names()
        self.metrics_data = metrics_preprocessor.get_data()

    def create_lambda_for_cutting_start_end(self, start, end):
        return lambda s: s[s.find(start) + len(start):s.rfind(end)] if not pd.isnull(s) else s

    def combine_and_save(self, output_dir):
        self.combine_and_save_on_metrics(output_dir)
        self.combine_and_save_on_span(output_dir)
        self.combine_and_save_on_traces(output_dir)

    def combine_and_save_on_metrics(self, output_dir):
        metrics_data_path = os.path.join(output_dir, "metrics_data.csv")
        combined_on_metrics_data = combine_on_metrics(self.tracing_data, self.metrics_data)
        combined_on_metrics_data.to_csv(metrics_data_path, sep=",", index=False)

    def combine_and_save_on_span(self, output_dir):
        spans_data_path = os.path.join(output_dir, "spans_data.csv")
        combined_on_spans = combine_on_spans(self.tracing_data, self.metrics_data)
        combined_on_spans.to_csv(spans_data_path, sep=",", index=False)

    def combine_and_save_on_traces(self, output_dir):
        traces_data_path = os.path.join(output_dir, "traces_data.csv")
        combined_on_traces_data = combine_on_traces(self.tracing_data, self.metrics_data)
        combined_on_traces_data.to_csv(traces_data_path, sep=",", index=False)
