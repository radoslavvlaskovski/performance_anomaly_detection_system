import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from performance_anomaly_detection.training import utils


def prepare_data_for_training_last_n_fold(X_train, X_test, y_train, y_test, original_y, original_y_test,
                                          exog_train=None, exog_test=None):
    if exog_train is not None and exog_test is not None:
        return (X_train, X_test), (y_train, y_test), (exog_train, exog_test), (original_y, original_y_test)
    return (X_train, X_test), (y_train, y_test), (original_y, original_y_test)


def train_scikit(data,
                 columns,
                 model,
                 scalers,
                 test_steps=1,
                 save_results=False,
                 results_output="out/",
                 label_data=None):
    start = time.time()
    global_scores, test_scores, train_rsq, test_rsq, predictions, real_ys = [], [], [], [], [], []

    scaler = None
    for ((X_train, X_test), (y_train, y_test), (original_y, original_y_test)), col, i in zip(data, columns,
                                                                                             range(0, len(columns))):
        if scalers is not None:
            scaler = scalers[i]
        print("Training for ", col)
        step = int(len(X_test) / test_steps)
        y_hat_test = []
        for s in range(test_steps):
            X_train_s = get_next_step_data(X_train, s, step)
            y_train_s = get_next_step_data(y_train, s, step)
            X_test_s = X_test[s * step: (s + 1) * step] if s < (test_steps - 1) else X_test[s * step:]

            model.fit(X_train_s, y_train_s)
            end = time.time()
            print("Execution time ", end - start)

            if s == 0:
                y_hat_train = model.predict(X_train_s)
            test_predictions = model.predict(X_test_s)
            y_hat_test.append(np.array(test_predictions).reshape(len(test_predictions), 1))
        y_hat_test = np.concatenate(y_hat_test)
        y_hat = np.concatenate((np.array(y_hat_train).reshape(len(y_hat_train), 1), y_hat_test))

        if scaler is not None:
            y_hat = scaler.inverse_transform(y_hat)
            y_hat_test = scaler.inverse_transform(y_hat_test)

        global_scores.append(mean_squared_error(original_y, y_hat))
        test_scores.append(mean_squared_error(original_y_test, y_hat_test))
        train_rsq.append(r2_score(original_y, y_hat))
        test_rsq.append(r2_score(original_y_test, y_hat_test))

        if save_results:
            predictions.append(pd.Series(y_hat, name=col))
            real_ys.append(pd.Series(original_y, name=col))
        #utils.plot_results(y=original_y, y_hat=y_hat)

    rmse_scores = utils.calculate_rmse(global_scores, test_scores)
    r2_scores = utils.calculate_r2(train_rsq, test_rsq)
    print(rmse_scores)
    print(r2_scores)
    if save_results:
        utils.save_results(predictions=predictions, real_ys=real_ys, results_output=results_output,
                           label_data=label_data)
    return rmse_scores, r2_scores

def train_nn(data,
             columns,
             callbacks,
             dev_size,
             scalers,
             optimizer,
             batch_size,
             model,
             epochs=5,
             test_steps=1,
             verbose=False,
             loss="mean_squared_error",
             save_results=False,
             results_output="out/",
             use_exog=True,
             label_data=None):
    start = time.time()
    scaler = None
    global_scores, test_scores, train_rsq, test_rsq, predictions, real_ys = [], [], [], [], [], []
    model.compile(loss=loss, optimizer=optimizer)

    for values_to_unpack, col, i in zip(data, columns, range(0, len(columns))):
        if scalers is not None:
            scaler = scalers[i]
        if use_exog:
            (X_train, X_test), (y_train, y_test), (exog_train, exog_test), (
            original_y, original_y_test) = values_to_unpack
        else:
            (X_train, X_test), (y_train, y_test), (original_y, original_y_test) = values_to_unpack

        print("Training for ", col)
        step = int(len(X_test) / test_steps)
        y_hat_test = []
        for s in range(test_steps):
            X_train_s = get_next_step_data(X_train, s, step)
            y_train_s = get_next_step_data(y_train, s, step)
            X_test_s = X_test[s * step: (s + 1) * step] if s < (test_steps - 1) else X_test[s * step:]

            if use_exog:
                exog_train_s = get_next_step_data(exog_train, s, step)
                exog_test_s = exog_test[s * step: (s + 1) * step] if s < (test_steps - 1) else exog_test[s * step:]

            train_inputs = [X_train_s, exog_train_s] if use_exog else [X_train_s]
            test_inputs = [X_test_s, exog_test_s] if use_exog else [X_test_s]

            history = model.fit(train_inputs, y_train_s, validation_split=dev_size, epochs=epochs,
                                batch_size=batch_size, verbose=verbose, callbacks=callbacks)

            end = time.time()
            print("Execution time ", end - start)

            if s == 0: y_hat_train = model.predict(train_inputs)
            test_predictions = model.predict(test_inputs)
            y_hat_test.append(np.array(test_predictions).reshape(len(test_predictions), 1))

        y_hat_test = np.concatenate(y_hat_test)
        y_hat = np.concatenate((np.array(y_hat_train).reshape(len(y_hat_train), 1), y_hat_test), axis=0)

        if scaler is not None: y_hat = scaler.inverse_transform(y_hat)
        if scaler is not None: y_hat_test = scaler.inverse_transform(y_hat_test)

        global_scores.append(mean_squared_error(original_y, y_hat))
        test_scores.append(mean_squared_error(original_y_test, y_hat_test))
        train_rsq.append(r2_score(original_y, y_hat))
        test_rsq.append(r2_score(original_y_test, y_hat_test))

        if save_results:
            predictions.append(pd.Series(y_hat, name=col))
            real_ys.append(pd.Series(original_y, name=col))

        #if verbose:
         #   utils.plot_results(y=original_y, y_hat=y_hat)
          #  utils.plot_loss(history)

    rmse_scores = utils.calculate_rmse(global_scores, test_scores)
    r2_scores = utils.calculate_r2(train_rsq, test_rsq)
    print(rmse_scores)
    print(r2_scores)

    if save_results:
        utils.save_results(predictions=predictions, real_ys=real_ys, results_output=results_output,
                           label_data=label_data)
    return y_hat, y_hat_test, rmse_scores, r2_scores


def get_next_step_data(data, s, step):
    return np.concatenate((data, data[:s * step])) if s != 0 else data

