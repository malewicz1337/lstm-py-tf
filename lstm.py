import os

import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import save_model

from data import Data


class DataGeneratorSeq(object):

    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                # self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]

            self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):

        unroll_data, unroll_labels = [], []
        # init_data, init_label = None, None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(
                0, min((b + 1) * self._segments, self._prices_length - 1)
            )


data = Data()
train_data = data.get_train_data()
all_mid_data = data.get_all_mid_data()
df = data.get_df()
test_data = data.get_test_data()


num_unrollings = 50  # Number of time steps you look into the future.
batch_size = 500  # Number of samples in a batch
num_nodes = [
    200,
    200,
    150,
]  # Number of hidden nodes in each layer of the deep LSTM stack
dropout = 0.2  # Dropout amount

x_train = np.expand_dims(train_data, axis=-1)
x_test = np.expand_dims(test_data, axis=-1)


def create_sequences(data, num_unrollings):
    sequences = []
    labels = []
    for i in range(len(data) - num_unrollings):
        sequences.append(data[i : i + num_unrollings])
        labels.append(data[i + num_unrollings])
    return np.array(sequences), np.array(labels)


x_train, y_train = create_sequences(train_data, num_unrollings)
x_test, y_test = create_sequences(test_data, num_unrollings)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


def build_model(num_nodes, dropout_rate):
    model = Sequential(
        [
            Input(shape=(None, 1)),
            LSTM(num_nodes[0], return_sequences=True),
            Dropout(dropout_rate),
            LSTM(num_nodes[1], return_sequences=True),
            Dropout(dropout_rate),
            LSTM(num_nodes[2]),
            Dropout(dropout_rate),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def plot_training_validation_loss():
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Training and Validation Loss")
    plt.legend()
    plt.show()


def plot_predictions():
    plt.figure(figsize=(15, 7))
    plt.plot(y_test, label="True Value", color="blue", marker="o")
    plt.plot(predictions, label="Predicted Value", color="red", linestyle="--")
    plt.title("Prediction vs True Value")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Price")
    plt.grid(True)
    plt.legend()

    max_error_idx = np.argmax(np.abs(y_test - predictions[:, 0]))
    max_error_value = predictions[max_error_idx, 0]
    plt.annotate(
        "Max Error",
        xy=(max_error_idx, max_error_value),
        xytext=(max_error_idx, max_error_value + 0.05),
        arrowprops=dict(facecolor="black", shrink=0.05),
        fontsize=12,
        color="green",
    )

    plt.show()


try:
    if os.path.exists("model.keras"):
        model = load_model("model.keras")

    else:
        model = build_model(num_nodes, dropout)
        checkpoint = ModelCheckpoint(
            "model.keras", monitor="val_loss", save_best_only=True, verbose=1
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        history = model.fit(
            x_train,
            y_train,
            epochs=30,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[checkpoint, early_stopping],
        )
        save_model(model, "model.keras")
        if "history" in locals():
            plot_training_validation_loss()

except Exception as e:
    print("Exception:", e)

predictions = model.predict(x_test)
plot_predictions()
