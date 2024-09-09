import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 1: Fetch Historical Stock Data
stock_symbol = 'TSLA'
df = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

# Step 2: Calculate Price-Based Features
# Daily return
df['Daily_Return'] = df['Close'].pct_change()

# Intraday range
df['Intraday_Range'] = df['High'] - df['Low']

# Opening gap
df['Opening_Gap'] = df['Open'] - df['Close'].shift(1)

# Moving averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# Price ratios
df['High_Low_Ratio'] = df['High'] / df['Low']
df['Close_Open_Ratio'] = df['Close'] / df['Open']

# Rolling volatility
df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=30).std()

# Drop NaN values
df.dropna(inplace=True)

# Step 3: Prepare Data for Training
features = ['Daily_Return', 'Intraday_Range', 'Opening_Gap', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'High_Low_Ratio', 'Close_Open_Ratio', 'Rolling_Volatility']
X = df[features]
y = df['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build and Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')

# Step 5: Generate Buy/Hold/Sell Recommendations
buy_threshold = 1.02  # Buy if predicted price is 2% higher than current price
sell_threshold = 0.98 # Sell if predicted price is 2% lower than current price

# Use the original DataFrame df to get the actual 'Close' prices for comparison
close_prices_test = df.loc[X_test.index, 'Close']

recommendations = np.where(y_pred_test > close_prices_test * buy_threshold, 'Buy',
                           np.where(y_pred_test < close_prices_test * sell_threshold, 'Sell', 'Hold'))

# Adding recommendations to X_test for easy visualization
X_test['Recommendations'] = recommendations
X_test['Close'] = close_prices_test

# Step 6: Visualize the Results
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(df.index[-len(y_test):], y_pred_test, label='Predicted Prices', color='red', linestyle='--')

for i, (date, recommendation) in enumerate(zip(df.index[-len(y_test):], recommendations)):
    if recommendation == 'Buy':
        plt.annotate('Buy', (date, y_test.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', color='green', fontsize=8)
    elif recommendation == 'Sell':
        plt.annotate('Sell', (date, y_test.iloc[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='red', fontsize=8)

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices with Buy/Hold/Sell Recommendations')
plt.legend()
plt.show()
