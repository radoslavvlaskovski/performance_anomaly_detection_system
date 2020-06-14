import enum
import json
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

from performance_anomaly_detection.data_preprocessing.time_series_preprocessing import *
from performance_anomaly_detection.training.nns import create_filter_net, create_rnn_net
from performance_anomaly_detection.training.time_series_training import train_scikit, train_nn, \
    prepare_data_for_training_last_n_fold


class ExperimentModelType(enum.Enum):
    rf = 1
    lstm = 2
    filternet = 3


class ExperimentDataType(enum.Enum):
    metrics = 1
    spans = 2
    traces = 3


class ExperimentModel:

    def __init__(self, name, model_type: ExperimentModelType, model, trained=False):
        self.name = name
        self.type = model_type
        self.model = model
        self.trained_model = None
        self.trained = trained

    def save_model(self, path):
        if self.type == ExperimentModelType.rf:
            pickle.dump(self.model, open(path, 'wb'))
            with open(path + "_metadata.json", 'w') as fp:
                json.dump(self.model.get_params(), fp)
        else:
            self.trained_model.save(path + ".h5")
            with open(path + "_metadata.json", 'w') as fp:
                fp.write(self.trained_model.to_json())


class ExperimentModelFile:

    def load_model(self, name, model_type: ExperimentModelType, file_path) -> ExperimentModel:
        if model_type == ExperimentModelType.rf:
            with open(file_path, 'rb') as f:
                rf = pickle.load(f)
                return ExperimentModel(name, model_type, rf, trained=True)
        else:
            keras_model = keras.models.load_model(file_path)
            return ExperimentModel(name, model_type, keras_model, trained=True)


class ModelTrainer:

    def train(self, data, data_type: ExperimentDataType, experiment_model: ExperimentModel):
        if experiment_model.type == ExperimentModelType.rf:
            return self.train_rf(data, data_type, experiment_model.model)
        elif experiment_model.type == ExperimentModelType.lstm:
            return self.train_lstm(data, data_type, experiment_model)
        elif experiment_model.type == ExperimentModelType.filternet:
            return self.train_filter_net(data, data_type, experiment_model)

    def train_rf(self, data, data_type: ExperimentDataType, model):
        _, training_data = self.create_training_data(data, data_type, scale_y=False)
        rmse_scores, r2_scores = train_scikit(data=training_data, columns=["1"], model=model, scalers=None, test_steps=1)
        (X_train, X_test), (_, _), (original_y, _) = training_data[0]
        y_hat = np.concatenate((model.predict(X_train), model.predict(X_test)), axis=0)

        scaler = MinMaxScaler()
        y = scaler.fit_transform(original_y.reshape(original_y.shape[0], 1))
        y_hat = scaler.transform(y_hat.reshape(y_hat.shape[0], 1))
        return y, y_hat, (rmse_scores, r2_scores)

    def train_lstm(self, data, data_type: ExperimentDataType, model):
        scaler, training_data = self.create_training_data(data, data_type, lstm=True)
        lstm = create_rnn_net(10, training_data[0][0][0].shape[2], rnn_type="LSTM", layers_size=model.model["layers_size"])
        y_hat, _, rmse_scores, r2_scores = train_nn(training_data,
                 ["all"],
                 [],
                 0.1,
                 None,
                 "adam",
                 2,
                 lstm,
                 epochs=3,
                 test_steps=1,
                 verbose=True,
                 loss="mean_squared_error",
                 save_results=False,
                 results_output="out/",
                 use_exog=False,
                 label_data=None)
        model.trained_model = lstm
        (X_train, X_test), (_, _), (original_y, _) = training_data[0]
        #y_hat = np.concatenate((lstm.predict(X_train), lstm.predict(X_test)), axis=0)
        y = MinMaxScaler().fit_transform(original_y.reshape(len(y_hat), 1))
        return y, y_hat, (rmse_scores, r2_scores)

    def train_filter_net(self, data, data_type: ExperimentDataType, model):
        scaler, training_data = self.create_training_data(data, data_type, filter_net=True)
        filter_net = create_filter_net(time_series_len=10, exog_input_size=training_data[0][2][0].shape[1],
                                       layer_one_size=model.model["layer_one_size"], layer_two_size=model.model["layer_two_size"],
                                       exog_layer_sizes=model.model["exog_layer_sizes"])
        y_hat, _, rmse_scores, r2_scores = train_nn(training_data,
                 ["all"],
                 [],
                 0.05,
                 None,
                 "adam",
                 2,
                 filter_net,
                 epochs=3,
                 test_steps=1,
                 verbose=True,
                 loss="mean_squared_error",
                 save_results=False,
                 results_output="out/",
                 use_exog=True,
                 label_data=None)
        model.trained_model = filter_net
        (X_train, X_test), (_, _), (exog_train, exog_test), (original_y, _) = training_data[0]
        #y_hat = np.concatenate((filter_net.predict([X_train, exog_train]), filter_net.predict([X_test, exog_test])),
          #                     axis=0)
        y = MinMaxScaler().fit_transform(original_y.reshape(len(y_hat), 1))
        return y, y_hat, (rmse_scores, r2_scores)

    def predict(self, data, data_type: ExperimentDataType, experiment_model: ExperimentModel):
        if experiment_model.type == ExperimentModelType.rf:
            return self.predict_rf(data, data_type, experiment_model.model)
        elif experiment_model.type == ExperimentModelType.lstm:
            return self.predict_lstm(data, data_type, experiment_model.model)
        elif experiment_model.type == ExperimentModelType.filternet:
            return self.predict_filter_net(data, data_type, experiment_model.model)

    def predict_rf(self, data, data_type: ExperimentDataType, model):
        _, training_data = self.create_training_data(data, data_type, scale_y=False)
        (X_train, X_test), (_, _), (original_y, _) = training_data[0]
        y_hat = np.concatenate((model.predict(X_train), model.predict(X_test)), axis=0)
        return original_y, y_hat, (("", ""), ("", ""))

    def predict_lstm(self, data, data_type: ExperimentDataType, model):
        scaler, training_data = self.create_training_data(data, data_type, lstm=True)
        (X_train, X_test), (_, _), (original_y, _) = training_data[0]
        y_hat = np.concatenate((model.predict(X_train), model.predict(X_test)), axis=0)
        y = MinMaxScaler().fit_transform(original_y.reshape(len(y_hat), 1))
        return y, y_hat, (("", ""), ("", ""))

    def predict_filter_net(self, data, data_type: ExperimentDataType, model):
        scaler, training_data = self.create_training_data(data, data_type, filter_net=True)
        (X_train, X_test), (_, _), (exog_train, exog_test), (original_y, _) = training_data[0]
        y_hat = np.concatenate((model.predict([X_train, exog_train]), model.predict([X_test, exog_test])),
                               axis=0)
        y = MinMaxScaler().fit_transform(original_y.reshape(len(y_hat), 1))
        return y, y_hat, (("", ""), ("", ""))

    def create_training_data(self, data, data_type: ExperimentDataType, lstm=False, filter_net=False, scale_y=True):
        if data_type == ExperimentDataType.spans:
            return self.process_data(data, "duration", ["duration", "step", "time", "operation_name", "start_time"],
                                     lstm, scale_y, filter_net=filter_net)
        elif data_type == ExperimentDataType.metrics:
            return self.process_data(data, "receive_bytes",
                                     ["receive_bytes", "step", "time", "operation_name", "start_time"], lstm, scale_y,
                                     filter_net=filter_net)
        elif data_type == ExperimentDataType.traces:
            return self.process_data(data, "trace_duration",
                                     ["trace_duration", "Recv.grpc.health.v1.Health.Check", "Recv.", "step", "time",
                                      "trace_id", "start_time", "end_time"], lstm, scale_y, filter_net=filter_net)
        else:
            raise Exception("Wrong data type specified. Must be: 'spans', 'metrics' or 'traces'")

    def process_data(self, data, column, not_exog, lstm, scale_y, filter_net=False, train_size=0.8):
        time_window = 10
        scaler = MinMaxScaler()
        exog_columns = [c for c in data.columns if
                        c not in not_exog]
        if not filter_net:
            X, y, original_y = prepare_data(data, time_window=time_window,
                                            scaler=scaler, scale_y=scale_y, exog_columns=exog_columns,
                                            oneh_columns=[], exog_scaler=StandardScaler(), future=1, y_column=column,
                                            exog_x_alignment=True, oneh_x_alignment=False, combine_endog_exog=True)
            if lstm:
                X = X.reshape(X.shape[0], time_window, int(X.shape[1] / time_window))
            else:
                X = X.reshape(X.shape[0], X.shape[1])

            train_size = int(len(X) * train_size)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            _, original_y_test = original_y[:train_size], original_y[train_size:]

            return scaler, [
                prepare_data_for_training_last_n_fold(X_train, X_test, y_train, y_test, original_y, original_y_test)]
        else:
            X, y, original_y, exog_data = prepare_data(data, time_window=time_window,
                                            scaler=scaler, scale_y=scale_y, exog_columns=exog_columns,
                                            oneh_columns=[], exog_scaler=StandardScaler(), future=1, y_column=column,
                                            exog_x_alignment=True, oneh_x_alignment=False, combine_endog_exog=False)

            train_size = int(len(X) * train_size)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            exog_train, exog_test = exog_data[:train_size], exog_data[train_size:]
            _, original_y_test = original_y[:train_size], original_y[train_size:]

            return scaler, [
                prepare_data_for_training_last_n_fold(X_train, X_test, y_train, y_test, original_y, original_y_test,
                                                      exog_train, exog_test)]

