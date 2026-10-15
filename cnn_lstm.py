"""CNN-LSTM model factory and fit/predict adapter (requires TensorFlow)."""
def build_cnn_lstm(input_shape):
    import tensorflow as tf
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(5, 5, activation="relu"),
        keras.layers.LSTM(5),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def make_cnn_lstm_fit_predict(z, y, seed=42, epochs=8, batch_size=16):
    def fit_predict(train_idx, test_idx, cols=None):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        model = build_cnn_lstm(z.shape[1:])
        model.fit(z[train_idx], y[train_idx], epochs=epochs,
                  batch_size=batch_size, verbose=0)
        pred_train = (model.predict(z[train_idx], verbose=0).ravel() > 0.5).astype(int)
        pred_test = (model.predict(z[test_idx], verbose=0).ravel() > 0.5).astype(int)
        return pred_train, pred_test
    return fit_predict
