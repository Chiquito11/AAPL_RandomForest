import yfinance as yf
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

ticker = "AAPL"

df= yf.download(ticker, start='2018-01-01',  auto_adjust=False)
#print(df.head())

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df['Return'] = df['Adj Close'].pct_change()
df['Target'] = df['Return'].shift(-1) #Forecast tomorrow's return

for i in range(1,6):
    df[f'Lag_{i}'] = df['Adj Close'].shift(i)

df['Volatility_5d'] = df['Return'].rolling(5).std()
df['Volatility_22d'] = df['Return'].rolling(22).std()
df['Day_Range'] = (df['High']-df['Low'])/df['Adj Close']
df['Dist_SMA20'] = (df['Adj Close'] / df['Adj Close'].rolling(20).mean()) - 1
df['Volume_Change'] = df['Volume'].pct_change()

df_clean = df.dropna()

features = [f'Lag_{i}' for i in range(1, 6)] + ['Volatility_5d','Volatility_22d','Day_Range','Dist_SMA20','Volume_Change']
X = df_clean[features]
y = df_clean['Target']

split = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(n_estimators=100, random_state=42,max_depth=10,min_samples_leaf=10)
model.fit(X_train, y_train)

preds = model.predict(X_test)
ic, _ = spearmanr(preds, y_test)

print(f"Information Coefficient w/Price Action: {ic:.4f}")

importances = pd.Series(model.feature_importances_, index=features)
print("\nTop Features:")
print(importances.sort_values(ascending=False))

#Backtesting

test_results = pd.DataFrame({'Real_Return': y_test, 'Predicted_Return': preds}, index=y_test.index)
test_results['Strategy_Return'] = np.where(test_results['Predicted_Return'] > 0, test_results['Real_Return'], 0)

cumulative_market = (1 + test_results['Real_Return']).cumprod()
cumulative_strategy = (1 + test_results['Strategy_Return']).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_market, label='Comprar e Segurar (AAPL)', color='gray', alpha=0.5)
plt.plot(cumulative_strategy, label='Estratégia Random Forest', color='blue', lw=2)
plt.title(f'Estratégia vs Mercado (IC: {ic:.4f})')
plt.legend()
plt.show()