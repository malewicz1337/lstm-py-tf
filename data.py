import datetime as dt
import json
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Data:
    def __init__(self, data_source="kaggle"):
        self.data_source = data_source
        self.df = None
        self.train_data = None
        self.test_data = None
        self.load_data()
        self.process_data()

    def load_data(self):
        if self.data_source == "alphavantage":
            self.load_data_from_alphavantage()
        else:
            self.load_data_from_kaggle()

    def load_data_from_alphavantage(self):
        api_key = "N70XAHFPHMEXBT21"
        ticker = "AAL"
        url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
        file_to_save = f"stock_market_data-{ticker}.csv"

        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                data = data["Time Series (Daily)"]
                self.df = pd.DataFrame(columns=["Date", "Low", "High", "Close", "Open"])
                for k, v in data.items():
                    date = dt.datetime.strptime(k, "%Y-%m-%d")
                    data_row = [
                        date.date(),
                        float(v["3. low"]),
                        float(v["2. high"]),
                        float(v["4. close"]),
                        float(v["1. open"]),
                    ]
                    self.df.loc[-1, :] = data_row
                    self.df.index = self.df.index + 1
                self.df.to_csv(file_to_save)
            print("Data saved to: %s" % file_to_save)
        else:
            print("File already exists. Loading data from CSV")
            self.df = pd.read_csv(file_to_save)

    def load_data_from_kaggle(self):
        self.df = pd.read_csv(
            os.path.join("Stocks", "hpq.us.txt"),
            delimiter=",",
            usecols=["Date", "Open", "High", "Low", "Close"],
        )
        print("Loaded data from the Kaggle repository")

    def process_data(self):
        self.df["Mid"] = (self.df["High"] + self.df["Low"]) / 2.0
        self.df.sort_values("Date", inplace=True)

        train_size = int(0.9 * len(self.df))
        self.train_data = self.df["Mid"].values[:train_size]
        self.test_data = self.df["Mid"].values[train_size:]

        scaler = MinMaxScaler()
        self.train_data = scaler.fit_transform(self.train_data.reshape(-1, 1)).flatten()
        self.test_data = scaler.transform(self.test_data.reshape(-1, 1)).flatten()

        self.smooth_data()

    def smooth_data(self, gamma=0.1):
        ema = 0.0
        for i in range(len(self.train_data)):
            ema = gamma * self.train_data[i] + (1 - gamma) * ema
            self.train_data[i] = ema

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_all_mid_data(self):
        all_mid_data = np.concatenate([self.train_data, self.test_data], axis=0)
        return all_mid_data

    def get_df(self):
        return self.df

    def plot_data(self):
        plt.figure(figsize=(18, 9))
        plt.plot(range(self.df.shape[0]), (self.df["Low"] + self.df["High"]) / 2.0)
        plt.xticks(
            range(0, self.df.shape[0], 500), self.df["Date"].loc[::500], rotation=45
        )
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Mid Price", fontsize=18)
        plt.show()
