import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def oneh_encode_data(column_name, data):
    onehenc = OneHotEncoder(categories="auto", sparse=False)
    return pd.DataFrame(onehenc.fit_transform(data.loc[:, [column_name]]), columns=onehenc.categories_[0])


def window_stack(a, stepsize=1, width=6):
    return np.hstack(a[i:i - width + 1 or None:stepsize] for i in range(0, width))


def scale_data(data, columns, scaler=StandardScaler(), overwrite=True):
    for column in columns:
        x = np.array(data[column]).reshape((len(data), 1))
        column_name = column if overwrite else column + "_scaled"
        data[column_name] = scaler.fit_transform(x)
    return data
