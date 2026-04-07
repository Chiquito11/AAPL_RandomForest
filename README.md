#APPL_RandomForest

This project implements a Quantitative Trading approach to forecast Apple's (AAPL) daily returns using Machine Learning. It focuses on price action features and robust model evaluation to avoid overfitting.

The goal is to predict the next-day price direction (return) using a Random Forest Regressor. The model helps identify potential entry/exit points by filtering market noise through volatility and trend indicators.

Language: Python

Data Source: Yahoo Finance (yfinance)

Machine Learning: scikit-learn

Financial Indicators: pandas-ta-classic

Visualization: matplotlib

Feature Engineering
The model's intelligence is built on the following features:

Dist_SMA20: Measures the distance between the Adjusted Close price and its 20-day Simple Moving Average to identify overextended trends.

Volatility_5d & Volatility_22d: Evaluates short and medium-term risk to filter out high-noise environments.

Day_Range: Analyzes intraday price amplitude (High-Low) relative to the Close price.

Volume_Change: Tracks momentum shifts through trading volume variations.

Lags (1-5): Historical price returns to capture short-term memory.

Model Performance
Information Coefficient (IC): The model consistently achieves an IC between 0.055 and 0.069, indicating a statistically significant predictive edge.

Overfitting Prevention: Optimized using GridSearchCV with parameters max_depth=10 and min_samples_leaf=10 to ensure the model generalizes well to new data.

Backtesting: The strategy focuses on capital preservation, often moving to "cash" (flat line) during high-drawdown periods for the benchmark.
