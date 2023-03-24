import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import ccxt


def fetch_data(symbol, timeframe='1d', since=None, limit=None):
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


def create_features(data):
    data['momentum'] = data['close'] - data['close'].shift(1)
    data['std_5'] = data['close'].rolling(window=5).std()
    data['std_10'] = data['close'].rolling(window=10).std()
    data['std_20'] = data['close'].rolling(window=20).std()
    data['return_5'] = data['close'].pct_change(5)
    data['return_10'] = data['close'].pct_change(10)
    data['return_20'] = data['close'].pct_change(20)
    data['signal'] = np.where(data['momentum'] > 0, 1, 0)
    data.dropna(inplace=True)
    return data


def trading_strategy(data, model):
    data['predicted_signal'] = model.predict(X)
    data['strategy_returns'] = data['return_5'] * data['predicted_signal'].shift(1)
    return data


# Fetch historical price data
symbol = 'BTC/USDT'
data = fetch_data(symbol)

# Preprocess and create features
data = create_features(data)

# Split data into training and testing sets
X = data[['momentum', 'std_5', 'std_10', 'std_20', 'return_5', 'return_10', 'return_20']]
y = data['signal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Implement the trading strategy and visualize the results
data = trading_strategy(data, model)

cumulative_returns = (1 + data['strategy_returns']).cumprod()
cumulative_returns.plot()
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.show()

