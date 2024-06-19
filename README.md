
# Trend Finder Method Using Wavelet Transform

This repository contains a Python implementation of a trend identification method for financial time series data using wavelet transforms. The method applies Discrete Wavelet Transform (DWT) to the closing prices of a stock to identify underlying trends more accurately and with less delay compared to traditional methods like the Exponential Moving Average (EMA).

## Features

- **Wavelet Transform:** Utilizes DWT for trend analysis, providing better localization in both time and frequency domains.
- **Reduced Delay:** Identifies trends with less lag compared to EMA, enhancing real-time decision-making.
- **Visualization:** Generates candlestick charts with trend indications, aiding in visual analysis.
- **Flexibility:** Allows users to select different wavelet types for analysis.

## Requirements

- numpy
- pandas
- yfinance
- pywt
- matplotlib
- mplfinance

## Usage

1. Fetch historical stock data using Yahoo Finance.
2. Apply the `TrendIdentifier` class to identify trends in the data.
3. Visualize the trends using `mplfinance` to generate candlestick charts with trend indications.

## Example

```python
import numpy as np
import pandas as pd
import yfinance as yf
import pywt
import matplotlib.pyplot as plt
import mplfinance as mpf

# Define the TrendIdentifier class
class TrendIdentifier:
    def __init__(self, wavelet='db2'):
        self.wavelet = wavelet

    def identify_trend(self, df):
        df['Trend'] = 'sideways'
        df['Trend'] = df['Trend'].astype(object)
        data = df['Close'].values
        if len(data) % 2 != 0:
            data = np.pad(data, (0, 1), 'constant', constant_values=(data[-1],))
        cA, cD = pywt.dwt(data, self.wavelet, 'smooth')
        approx = pywt.idwt(cA, np.zeros_like(cD), self.wavelet)
        if len(approx) > len(df):
            approx = approx[:len(df)]
        elif len(approx) < len(df):
            approx = np.pad(approx, (0, len(df) - len(approx)), 'edge')
        df['approx'] = approx
        for i in range(3, len(df)):
            if (df['approx'].iloc[i] > df['approx'].iloc[i-1] and 
                df['approx'].iloc[i] > df['approx'].iloc[i-2] and 
                df['approx'].iloc[i] > df['approx'].iloc[i-3]):
                df.loc[df.index[i], 'Trend'] = 'up'
            elif (df['approx'].iloc[i] < df['approx'].iloc[i-1] and 
                  df['approx'].iloc[i] < df['approx'].iloc[i-2] and 
                  df['approx'].iloc[i] < df['approx'].iloc[i-3]):
                df.loc[df.index[i], 'Trend'] = 'down'
            else:
                df.loc[df.index[i], 'Trend'] = 'sideways'
        return df

# Fetch historical data
def fetch_historical_data(symbol, interval='1d', period='6mo'):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

# Example usage
df = fetch_historical_data("AAPL", "1d", "6mo")
trend_identifier = TrendIdentifier(wavelet='db24')
result_df = trend_identifier.identify_trend(df)

# Visualization
arrows_up = np.nan * np.ones(len(result_df))
arrows_down = np.nan * np.ones(len(result_df))
for i in range(len(result_df)):
    if result_df['Trend'].iloc[i] == 'up':
        arrows_up[i] = result_df['High'].iloc[i] + 0.1
    elif result_df['Trend'].iloc[i] == 'down':
        arrows_down[i] = result_df['Low'].iloc[i] - 0.1
arrows_up_series = pd.Series(arrows_up, index=result_df.index)
arrows_down_series = pd.Series(arrows_down, index=result_df.index)
apds = [
    mpf.make_addplot(result_df['approx'], color='blue', panel=0, ylabel='Approximation'),
    mpf.make_addplot(arrows_up_series, scatter=True, markersize=20, marker='^', alpha=0.35, color='green', panel=0),
    mpf.make_addplot(arrows_down_series, scatter=True, markersize=20, marker='v', alpha=0.35, color='red', panel=0)
]
mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc)
mpf.plot(result_df, type='candle', addplot=apds, style=s, title='Candlestick Chart with Wavelet Trends', volume=False, savefig='candlestick_chart_with_wavelet_trends.svg')
```

# Trend Finder Method Using Wavelet Transform and MLPClassifier

This repository contains a Python implementation of a trend identification and prediction method for financial time series data using wavelet transforms and MLPClassifier. The method applies Discrete Wavelet Transform (DWT) to the closing prices of a stock to identify underlying trends more accurately and with less delay compared to traditional methods like the Exponential Moving Average (EMA). After trend identification, it uses an MLPClassifier to predict the current candlestick trend based on these historical data patterns.

## Features

- **Wavelet Transform:** Utilizes DWT for trend analysis, providing better localization in both time and frequency domains.
- **MLPClassifier:** Employs a neural network classifier to predict future trends based on extracted features from historical data.
- **Reduced Delay:** Identifies trends with less lag compared to EMA, enhancing real-time decision-making.
- **Visualization:** Generates candlestick charts with trend indications, aiding in visual analysis.
- **Flexibility:** Allows users to select different wavelet types for analysis.

## Requirements

- numpy
- pandas
- yfinance
- pywt
- matplotlib
- mplfinance
- scikit-learn

## Usage

1. Fetch historical stock data using Yahoo Finance.
2. Apply the `TrendIdentifier` class to identify trends in the data.
3. Use the `MLPClassifier` to predict the trend for the current candlestick based on historical trends.
4. Visualize the trends and predictions using `mplfinance` to generate candlestick charts with trend indications.

## Example

```python
import numpy as np
import pandas as pd
import yfinance as yf
import pywt
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the TrendIdentifier class
class TrendIdentifier:
    def __init__(self, wavelet='db2'):
        self.wavelet = wavelet

    def identify_trend(self, df):
        df['Trend'] = 'sideways'
        df['Trend'] = df['Trend'].astype(object)
        data = df['Close'].values
        if len(data) % 2 != 0:
            data = np.pad(data, (0, 1), 'constant', constant_values=(data[-1],))
        cA, cD = pywt.dwt(data, self.wavelet, 'smooth')
        approx = pywt.idwt(cA, np.zeros_like(cD), self.wavelet)
        if len(approx) > len(df):
            approx = approx[:len(df)]
        elif len(approx) < len(df):
            approx = np.pad(approx, (0, len(df) - len(approx)), 'edge')
        df['approx'] = approx
        for i in range(3, len(df)):
            if (df['approx'].iloc[i] > df['approx'].iloc[i-1] and 
                df['approx'].iloc[i] > df['approx'].iloc[i-2] and 
                df['approx'].iloc[i] > df['approx'].iloc[i-3]):
                df.loc[df.index[i], 'Trend'] = 'up'
            elif (df['approx'].iloc[i] < df['approx'].iloc[i-1] and 
                  df['approx'].iloc[i] < df['approx'].iloc[i-2] and 
                  df['approx'].iloc[i] < df['approx'].iloc[i-3]):
                df.loc[df.index[i], 'Trend'] = 'down'
            else:
                df.loc[df.index[i], 'Trend'] = 'sideways'
        return df

# Fetch historical data
def fetch_historical_data(symbol, interval='1d', period='6mo'):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

# Example usage
df = fetch_historical_data("AAPL", "1d", "6mo")
trend_identifier = TrendIdentifier(wavelet='db24')
result_df = trend_identifier.identify_trend(df)

# Prediction using MLPClassifier
features = result_df[['High_pct_change', 'Low_pct_change', 'Open_pct_change', 'Close_pct_change']]
labels = result_df['Trend'].map({'up': 1, 'down': -1, 'sideways': 0})
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.4, shuffle=False)
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=200)
mlp.fit(X_train, y_train)
result_df['MLP_Trend'] = mlp.predict(scaler.transform(features))

# Visualization
apds = [mpf.make_addplot(result_df['approx'], color='blue', panel=0, ylabel='Approximation')]
mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc)
mpf.plot(result_df, type='candle', addplot=apds, style=s, title='Candlestick Chart with Wavelet and MLP Predicted Trends', volume=False)
```

## License

This project is licensed under the MIT License.
