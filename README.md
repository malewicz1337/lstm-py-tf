# LSTM Python TensorFlow Model

This repository contains a Python implementation of an LSTM model using TensorFlow for time series prediction, specifically focusing on stock data.

## Overview

This project leverages Long Short-Term Memory (LSTM) networks, a special kind of Recurrent Neural Network (RNN), to predict stock market trends based on historical data. The model is built and trained using TensorFlow.

## Files and Directories

- data.py: Script for data manipulation and preparation.
- exp_mov_avg_predictions.py: Implementation of Exponential Moving Average for baseline prediction.
- lstm.py: Core LSTM model implementation.
- main.py: Main script to run the model training and prediction.
- model.keras: Pre-trained LSTM model saved in Keras format.
- requirements.txt: List of Python packages required to run the project.
- std_avg_predictions.py: Standard average prediction implementation.
- Stocks/: Directory containing stock data files.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/malewicz1337/lstm-py-tf.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the accuracy or functionality of the model.

## Acknowledgments

Inspiration and initial model parameters were drawn from various open-source projects and academic papers on stock prediction using LSTM networks.

## License

This project is licensed under the terms of the MIT license.
