import numpy as np
from performance_anomaly_detection.data_preprocessing import utils


def create_endogenous_time_window(data, y_column, time_window_size, future=1):
    y = np.array(data[y_column])
    X = np.zeros((y.shape[0], time_window_size))

    for i in range(time_window_size, y.shape[0]):
        X[i, :] += y[i - time_window_size:i]

    X = X[time_window_size: -(future - 1), :] if future > 1 else X[time_window_size:, :]

    return X


def create_y(data, y_column, time_window_size, future=1):
    y = np.array(data[y_column])
    y = y[time_window_size + future - 1:]

    return y


def create_inputs_and_outputs(data, y_column, time_window_size, future=1):
    return create_endogenous_time_window(data, y_column, time_window_size, future), create_y(data, y_column,
                                                                                             time_window_size, future)


def create_x_y(data, y_column, scaler, scale_y, time_window, future):
    if scale_y:
        if scaler is not None:
            data = utils.scale_data(data, [y_column], scaler=scaler)
        X, y = create_inputs_and_outputs(data, y_column, time_window, future)
    else:
        X, y = create_inputs_and_outputs(data, y_column, time_window, future)
        if scaler is not None:
            X = scaler.fit_transform(X)

    return X.reshape(X.shape[0], X.shape[1], 1), y.reshape(y.shape[0], 1)


def create_exog_data(data, time_window, future, exog_columns, oneh_columns, exog_x_alignment, oneh_x_alignment,
                     scaler=None):
    if len(exog_columns) > 0:
        if scaler is not None:
            x = utils.scale_data(data, exog_columns, scaler=scaler)
        else:
            x = data

        x = np.array(x.loc[:, exog_columns])
        if exog_x_alignment:
            x = utils.window_stack(x, width=time_window)
        exog_data = np.array(x)
        exog_data.reshape(exog_data.shape[0], exog_data.shape[1], 1)
        exog_data = exog_data[future:]
        if not exog_x_alignment:
            exog_data = exog_data[time_window - 1:]

    oneh_data = []
    for oneh in oneh_columns:
        data_encoded = np.array(utils.oneh_encode_data(oneh, data))
        if not oneh_x_alignment:
            oneh_data.append(data_encoded[time_window + future - 1:])
        else:
            oneh_data.append(utils.window_stack(data_encoded, width=time_window)[future:])

    if len(oneh_columns) > 0:
        oneh_data = np.concatenate(oneh_data, axis=1)
        if len(exog_columns) > 0:
            exog_data = np.concatenate((exog_data, oneh_data), axis=1)
        else:
            exog_data = oneh_data

    return exog_data


def prepare_data(data, time_window, scaler, scale_y, exog_columns, oneh_columns, exog_scaler, future, y_column,
                 exog_x_alignment, oneh_x_alignment, combine_endog_exog=False, label_columns=None):
    original_data = data[y_column]
    original_y = np.array(original_data)[time_window + future - 1:]

    label_data = None
    if label_columns is not None:
        label_data = data[label_columns].iloc[time_window + future - 1:]

    X, y = create_x_y(data=data, y_column=y_column, scaler=scaler, scale_y=scale_y, time_window=time_window,
                      future=future)
    if len(exog_columns) > 0 or len(oneh_columns) > 0:
        exog_data = create_exog_data(data=data, time_window=time_window, future=future, exog_columns=exog_columns,
                                     oneh_columns=oneh_columns, scaler=exog_scaler, exog_x_alignment=exog_x_alignment,
                                     oneh_x_alignment=oneh_x_alignment)
    else:
        if label_data is not None:
            return X, y, original_y, label_data
        return X, y, original_y

    if combine_endog_exog:
        exog_data = exog_data.reshape(exog_data.shape[0], exog_data.shape[1], 1)
        X = np.concatenate((X, exog_data), axis=1)
        if label_data is not None:
            return X, y, original_y, label_data
        return X, y, original_y

    if label_data is not None:
        return X, y, original_y, exog_data, label_data
    return X, y, original_y, exog_data
