import datetime as dt

import matplotlib.pyplot as plt
import numpy as np


class StandartAveragePredictions:
    def __init__(self, train_data, all_mid_data, df):
        self.window_size = 100
        self.train_data = train_data
        self.N = train_data.size
        self.std_avg_predictions = []
        self.std_avg_x = []
        self.mse_errors = []
        self.all_mid_data = all_mid_data
        self.df = df
        self.dates = self.df["Date"].tolist()
        self.calculate_predictions()

    def calculate_predictions(self):
        for pred_idx in range(self.window_size, self.N):

            if pred_idx >= self.N:
                date = dt.datetime.strptime(
                    self.dates[-1], "%Y-%m-%d"
                ).date() + dt.timedelta(days=1)
            else:
                date = self.df.loc[pred_idx, "Date"]

            self.std_avg_predictions.append(
                np.mean(self.train_data[pred_idx - self.window_size : pred_idx])
            )
            self.mse_errors.append(
                (self.std_avg_predictions[-1] - self.train_data[pred_idx]) ** 2
            )
            self.std_avg_x.append(date)

        print(
            "MSE error for standard averaging: %.5f" % (0.5 * np.mean(self.mse_errors))
        )

    def plot_avg_predictions(self):
        plt.figure(figsize=(18, 9))
        plt.plot(range(self.df.shape[0]), self.all_mid_data, color="b", label="True")
        plt.plot(
            range(self.window_size, self.N),
            self.std_avg_predictions,
            color="orange",
            label="Prediction",
        )
        plt.title("Standard Average Predictions vs. True Mid Prices")
        # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Mid Price")
        plt.legend(fontsize=18)
        plt.show()
