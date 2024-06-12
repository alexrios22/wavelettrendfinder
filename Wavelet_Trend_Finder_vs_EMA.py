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

        # Apply DWT to the closing prices with padding
        # Ensure the array length matches the original DataFrame
        data = df['Close'].values
        if len(data) % 2 != 0:  # Checking if the length of the data is even
            data = np.pad(data, (0, 1), 'constant', constant_values=(data[-1],))
        
        cA, cD = pywt.dwt(data, self.wavelet, 'smooth')
        
        # Reconstruct the approximation component using inverse DWT
        approx = pywt.idwt(cA, np.zeros_like(cD), self.wavelet)
        if len(approx) > len(df):
            approx = approx[:len(df)]
        elif len(approx) < len(df):
            approx = np.pad(approx, (0, len(df) - len(approx)), 'edge')
        
        df['approx'] = approx

        # Determine trends based on the approximation component
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

# Fetch historical data using Yahoo Finance
def fetch_historical_data(symbol, interval='1d', period='1y'):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

# Calculate EMA
def add_ema(df, span=10):
    df['EMA'] = df['Close'].ewm(span=span, adjust=False).mean()

# Instantiate the TrendIdentifier and test the function
try:
    df = fetch_historical_data("AAPL", "1d", "6mo")
    trend_identifier = TrendIdentifier(wavelet='db24')
    result_df = trend_identifier.identify_trend(df)
    add_ema(result_df)  # Add EMA to the DataFrame

    # Plotting the candlestick chart along with trends and EMA
    apds = [
        mpf.make_addplot(result_df['approx'], color='blue', panel=0, ylabel='Approximation'),
        mpf.make_addplot(result_df['EMA'], color='purple', panel=0, ylabel='EMA'),
    ]

    # Color coding trends
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    # Save the plot as an SVG file
    mpf.plot(result_df, type='candle', addplot=apds, style=s,
              title='Candlestick Chart with Wavelet Trends and EMA', volume=False, 
              savefig='candlestick_chart_with_wavelet_trends_and_ema.eps')

except Exception as e:
    print(f"Error: {e}")
