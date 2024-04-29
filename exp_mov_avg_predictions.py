import datetime as dt

import matplotlib.pyplot as plt
import numpy as np


class ExponentialMovingAveragePredictions:
    def __init__(self, train_data, all_mid_data, df):
        self.window_size = 100
        self.train_data = train_data
        self.N = train_data.size
        self.all_mid_data = all_mid_data
        self.df = df
        self.run_avg_predictions = []
        self.run_avg_x = []
        self.mse_errors = []
        self.dates = self.df["Date"].tolist()
        self.calculate_predictions()

    def calculate_predictions(self):
        running_mean = 0.0
        self.run_avg_predictions.append(running_mean)

        decay = 0.5

        for pred_idx in range(1, self.N):

            if pred_idx >= self.N:
                date = dt.datetime.strptime(
                    self.dates[-1], "%Y-%m-%d"
                ).date() + dt.timedelta(days=1)
            else:
                date = self.df.loc[pred_idx, "Date"]

            running_mean = (
                running_mean * decay + (1.0 - decay) * self.train_data[pred_idx - 1]
            )
            self.run_avg_predictions.append(running_mean)
            self.mse_errors.append(
                (self.run_avg_predictions[-1] - self.train_data[pred_idx]) ** 2
            )
            self.run_avg_x.append(date)

        print("MSE error for EMA averaging: %.5f" % (0.5 * np.mean(self.mse_errors)))

    def plot_exp_mov_predictions(self):
        plt.figure(figsize=(18, 9))
        plt.plot(range(self.df.shape[0]), self.all_mid_data, color="b", label="True")
        plt.plot(
            range(0, self.N),
            self.run_avg_predictions,
            color="orange",
            label="Prediction",
        )
        plt.title("Exponential Moving Average Predictions vs. True Mid Prices")
        # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Mid Price")
        plt.legend(fontsize=18)
        plt.show()
