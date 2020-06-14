from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, AveragePooling1D, ReLU, Dropout, Concatenate, Flatten, RNN, GRU, LSTM

'''
################# FILTERNET #################
'''


def create_filter_net_only_conv(time_series_len, layer_one_size, layer_two_size):
    wave_input = Input(shape=(time_series_len, 1))
    inputs = [wave_input]
    layers = create_filter_net_conv_layers(wave_input, layer_one_size, layer_two_size)
    combined = Concatenate()(layers)
    out = Dense(1)(combined)
    return Model(inputs=inputs, outputs=out)


def create_filter_net(time_series_len, exog_input_size, layer_one_size, layer_two_size,
                      exog_layer_sizes=(2 ** 10, 2 ** 9, 2 ** 8, 2 ** 7), dropout=0.):
    wave_input = Input(shape=(time_series_len, 1))
    inputs = [wave_input]
    layers = create_filter_net_conv_layers(wave_input, layer_one_size, layer_two_size)

    dense_input = Input(shape=(exog_input_size,))
    inputs.append(dense_input)
    dense_layers = create_filternet_exog_layers(exog_layer_sizes, dense_input, dropout)
    layers.append(dense_layers)

    combined = Concatenate()(layers)
    out = Dense(1)(combined)
    return Model(inputs=inputs, outputs=out)


def create_filter_net_conv_layers(wave_input, layer_one_size, layer_two_size):
    layers = []
    for i in range(1, layer_one_size + 1):
        layers.append(Conv1D_ks(kernelsize=i)(wave_input))
    for i in range(1, layer_two_size + 1):
        layers.append(Conv1D_ks(kernelsize=i)(layers[i - 1]))
    for i, layer in enumerate(layers):
        layers[i] = Flatten()(layer)
    return layers


def create_filternet_exog_layers(exog_layer_sizes, dense_input, dropout):
    dense_layer = Dense(exog_layer_sizes[0], activation="relu")(dense_input)
    dense_layer = Dropout(dropout)(dense_layer)
    for i in range(1, len(exog_layer_sizes)):
        dense_layer = Dense(exog_layer_sizes[i], activation="relu")(dense_layer)
        dense_layer = Dropout(dropout)(dense_layer)
    return dense_layer


def Conv1D_ks(kernelsize=1, dilation=1, pooling_size=2):
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=kernelsize, data_format="channels_last", dilation_rate=dilation))
    model.add(AveragePooling1D(pool_size=pooling_size, padding="same"))
    model.add(ReLU())
    return model


'''
################# RNN #################
'''


def create_rnn_net(input_shape_time_sequence, input_shape_feature_count, rnn_type="LSTM",
                   layers_size=(64, 64), dropout=0.):
    model = Sequential()
    for i in range(0, len(layers_size)):
        model.add(create_rnn_layer(rnn_type, layers_size[i], True,
                                   input_shape=(input_shape_time_sequence, input_shape_feature_count)))
        if dropout > 0.:
            model.add(Dropout(dropout))
    model.add(create_rnn_layer(rnn_type, layers_size[-1], False,
                               input_shape=(input_shape_time_sequence, input_shape_feature_count)))
    if dropout > 0.:
        model.add(Dropout(dropout))
    model.add(Dense(1))
    return model


def create_rnn_layer(rnn_type, layer_size, return_seq, input_shape):
    if rnn_type.upper() == "RNN":
        return RNN(layer_size, return_sequences=return_seq, input_shape=input_shape)
    if rnn_type.upper() == "GRU":
        return GRU(layer_size, return_sequences=return_seq, input_shape=input_shape)
    if rnn_type.upper() == "LSTM":
        return LSTM(layer_size, return_sequences=return_seq, input_shape=input_shape)
