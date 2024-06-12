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

        # Ensure approx is the same length as the original DataFrame
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
    print(df)
    return df

# Instantiate the TrendIdentifier and test the function
try:
    df = fetch_historical_data("AAPL", "1d", "6mo")
    trend_identifier = TrendIdentifier(wavelet='db24')
    result_df = trend_identifier.identify_trend(df)
    # Add arrows for trends
    arrows_up = np.nan * np.ones(len(result_df))
    arrows_down = np.nan * np.ones(len(result_df))
    for i in range(len(result_df)):
        if result_df['Trend'].iloc[i] == 'up':
            arrows_up[i] = result_df['High'].iloc[i] + 0.1
        elif result_df['Trend'].iloc[i] == 'down':
            arrows_down[i] = result_df['Low'].iloc[i] - 0.1

    # Convert arrows to pandas Series with datetime index
    arrows_up_series = pd.Series(arrows_up, index=result_df.index)
    arrows_down_series = pd.Series(arrows_down, index=result_df.index)

    # Plotting the candlestick chart along with trends
    apds = [
        mpf.make_addplot(result_df['approx'], color='blue', panel=0, ylabel='Approximation'),
        mpf.make_addplot(arrows_up_series, scatter=True, markersize=20, marker='^',
                          alpha=0.35, color='green', panel=0),
        mpf.make_addplot(arrows_down_series, scatter=True, markersize=20, marker='v',
                          alpha=0.35, color='red', panel=0)
    ]

    # Color coding trends
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    # Save the plot as an SVG file
    mpf.plot(result_df, type='candle', addplot=apds, style=s,
              title='Candlestick Chart with Wavelet Trends', volume=False, 
              savefig='candlestick_chart_with_wavelet_trends.eps')

except Exception as e:
    print(f"Error: {e}")
