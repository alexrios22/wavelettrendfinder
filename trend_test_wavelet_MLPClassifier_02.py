import numpy as np
import pandas as pd
import yfinance as yf
import pywt
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class TrendIdentifier:
    def __init__(self, wavelet='db2'):
        self.wavelet = wavelet

    def identify_trend(self, df):
        df['Trend'] = 'sideways'
        df['Trend'] = df['Trend'].astype(object)

        data = df['Close'].values
        if len(data) % 2 != 0:  # Ensuring even length for DWT
            data = np.pad(data, (0, 1), 'constant', constant_values=(data[-1],))

        cA, cD = pywt.dwt(data, self.wavelet, 'smooth')
        approx = pywt.idwt(cA, np.zeros_like(cD), self.wavelet)

        # Ensure approx length matches df
        approx = approx[:len(df)]
        df['approx'] = approx

        # Improved trend identification
        df['Trend'] = np.where(
            (df['approx'] > df['approx'].shift(1)) &
            (df['approx'] > df['approx'].shift(2)) &
            (df['approx'] > df['approx'].shift(3)),
            'up',
            np.where(
                (df['approx'] < df['approx'].shift(1)) &
                (df['approx'] < df['approx'].shift(2)) &
                (df['approx'] < df['approx'].shift(3)),
                'down',
                'sideways'
            )
        )

        return df

def fetch_historical_data(symbol, interval='1d', period='1y'):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

def add_percentage_change_features(df, window=3, shift_days=2):
    df['High_pct_change'] = df['High'].pct_change(periods=window).shift(shift_days)
    df['Low_pct_change'] = df['Low'].pct_change(periods=window).shift(shift_days)
    df['Open_pct_change'] = df['Open'].pct_change(periods=window).shift(shift_days)
    df['Close_pct_change'] = df['Close'].pct_change(periods=window).shift(shift_days)
    
    df['candle_close2open'] = ((df['Close'] - df['Open']) / (df['High'] - df['Low'])).shift(shift_days)
    df['candle_high2oc'] = ((df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'])).shift(shift_days)
    df['candle_high2low2close'] = ((df['High'] - df['Low']) / df['Close']).shift(shift_days)
    df['candle_low2oc'] = ((df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'])).shift(shift_days)
    
    df.dropna(inplace=True)
    return df

df = fetch_historical_data("AAPL", "1d", "5y")
trend_identifier = TrendIdentifier(wavelet='db24')
result_df = trend_identifier.identify_trend(df)

result_df = add_percentage_change_features(result_df, window=1, shift_days=1)

features = result_df[['High_pct_change',
                      'Low_pct_change',
                      'Open_pct_change',
                      'Close_pct_change',
                      'candle_close2open',
                      'candle_high2oc',
                      'candle_low2oc',
                      'candle_high2low2close']]
labels = result_df['Trend'].map({'up': 1, 'down': -1, 'sideways': 0})


X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.4, shuffle=False)
# Scale the training and test sets separately
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100),
                          activation='relu',
                          solver='sgd',
                          alpha=1e-5,
                          learning_rate_init=0.01,
                          max_iter=200,
                          warm_start=True)
mlp_model.out_activation_ = 'softmax'
mlp_model.fit(X_train_scaled, y_train)

# Plot the convergence of the neural network
plt.figure(figsize=(10, 6))
plt.plot(mlp_model.loss_curve_, label='Loss Curve')
plt.title('Neural Network Convergence')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

y_pred = mlp_model.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred, zero_division=0))

# Print the accuracy score
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

result_df['MLP_Trend'] = mlp_model.predict(scaler.transform(result_df[['High_pct_change',
                                                                       'Low_pct_change',
                                                                       'Open_pct_change',
                                                                       'Close_pct_change',
                                                                       'candle_close2open',
                                                                       'candle_high2oc',
                                                                       'candle_low2oc',
                                                                       'candle_high2low2close']].fillna(0)))

crp = 0.1
arrows_up = np.where(result_df['MLP_Trend'] == 1, result_df['High'] * (1+2*crp), np.nan)
arrows_down = np.where(result_df['MLP_Trend'] == -1, result_df['Low'] * (1-2*crp), np.nan)

arrows_up_org = np.where(result_df['Trend'] == 'up', result_df['High'] * (1+crp), np.nan)
arrows_down_org = np.where(result_df['Trend'] == 'down', result_df['Low'] * (1-crp), np.nan)

# Define the additional plots
alphac = 0.45
mkz = 5
apds = [
    mpf.make_addplot(result_df['approx'], color='blue', alpha= alphac, panel=0,
                     ylabel='Approximation'),
    
    mpf.make_addplot(arrows_up_org, scatter=True, markersize=mkz, marker='^',
                     color='green', alpha= alphac, panel=0, label='Original Up Trend'),
    
    mpf.make_addplot(arrows_up, scatter=True, markersize=mkz, marker='^',
                     color='black', alpha= alphac, panel=0, label='Predicted Up Trend'),
    
    mpf.make_addplot(arrows_down_org, scatter=True, markersize=mkz, marker='v',
                     color='red', alpha= alphac, panel=0, label='Original Down Trend'),
    
    mpf.make_addplot(arrows_down, scatter=True, markersize=mkz, marker='v',
                     color='orange', alpha= alphac, panel=0, label='Predicted Down Trend'),
]

mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc)

mpf.plot(result_df, type='candle', 
         addplot=apds, style=s,
         title='Candlestick Chart with Wavelet and MLP Predicted Trends', 
         volume=False,
         savefig='candlestick_chart_with_wavelet_MLP_Predicted_trends.svg')
