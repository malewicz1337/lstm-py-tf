def main():
    from data import Data
    from exp_mov_avg_predictions import ExponentialMovingAveragePredictions
    from std_avg_predictions import StandartAveragePredictions

    data = Data()
    train_data = data.get_train_data()
    all_mid_data = data.get_all_mid_data()
    df = data.get_df()

    std_avg = StandartAveragePredictions(train_data, all_mid_data, df)

    std_avg.plot_avg_predictions()

    exp_avg = ExponentialMovingAveragePredictions(train_data, all_mid_data, df)
    exp_avg.plot_exp_mov_predictions()


if __name__ == "__main__":
    main()
