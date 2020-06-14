import numpy as np


def combine_on_metrics(tracing_data, metrics_data):
    combined_data = prepare_tracing_data(tracing_data, metrics_data,
                                         columns=["operation_name", "duration", "start_time"])
    set_steps_on_combined_data(combined_data, metrics_data)
    metrics_data["step"] = range(len(metrics_data))
    combined_data = combined_data.groupby(by=["operation_name", "step"]).mean()
    combined_data.reset_index(inplace=True)
    combined_data = combined_data.pivot(index='step', columns='operation_name', values='duration')
    combined_data = combined_data.fillna(0)
    combined_data = combined_data.merge(metrics_data, left_on="step", right_on="step")

    return combined_data


def combine_on_spans(tracing_data, metrics_data):
    combined_data = prepare_tracing_data(tracing_data, metrics_data,
                                         columns=["operation_name", "duration", "start_time"])
    set_steps_on_combined_data(combined_data, metrics_data)
    metrics_data["step"] = range(len(metrics_data))
    combined_data = combined_data.merge(metrics_data, left_on="step", right_on="step")

    return combined_data


def combine_on_traces(tracing_data, metrics_data):
    combined_data = prepare_tracing_data(tracing_data, metrics_data,
                                         columns=["trace_id", "operation_name", "duration", "start_time", "end_time"])
    set_steps_on_combined_data(combined_data, metrics_data)
    metrics_data["step"] = range(len(metrics_data))

    tracing_data_mean = combined_data.groupby(by=["trace_id", "operation_name"]).mean()
    tracing_data_mean.reset_index(inplace=True)
    tracing_data_mean = tracing_data_mean.pivot(index='trace_id', columns='operation_name', values='duration')
    tracing_data_mean = tracing_data_mean.fillna(0)

    tracing_data_min = combined_data.groupby(by=["trace_id"]).min()
    tracing_data_min.reset_index(inplace=True)
    tracing_data_max = combined_data.groupby(by=["trace_id"]).max()
    tracing_data_max.reset_index(inplace=True)
    tracing_data_times = tracing_data_max[["trace_id", "end_time"]].merge(
        tracing_data_min[["step", "start_time", "trace_id"]], left_on=["trace_id"], right_on=["trace_id"])
    combined_tracing_data = tracing_data_mean.merge(tracing_data_times,
                                                    left_on=["trace_id"], right_on=["trace_id"])
    combined_data = combined_tracing_data.merge(metrics_data, left_on="step", right_on="step")
    combined_data["trace_duration"] = combined_data["end_time"] - combined_data["start_time"]
    return combined_data


def prepare_tracing_data(tracing_data, metrics_data, columns):
    combined_data = filter_tracing_data_relevant_for_metrics(tracing_data, metrics_data)
    combined_data = combined_data[columns]
    combined_data["step"] = np.ones(len(combined_data))
    return combined_data


def filter_tracing_data_relevant_for_metrics(tracing_data, metrics_data):
    return tracing_data.loc[(tracing_data.start_time >= metrics_data.time.iloc[0])
                            & (tracing_data.start_time < metrics_data.time.iloc[-1])]


def set_steps_on_combined_data(combined_data, metrics_data):
    j = 0
    done = False
    current_step = metrics_data.iloc[j].time
    for i in range(len(combined_data)):
        if not done and combined_data.iloc[i].start_time > current_step:
            j += 1
            if len(metrics_data) == j:
                done = True
            else:
                current_step = metrics_data.iloc[j].time
        combined_data.step.iloc[i] = j
    return combined_data
