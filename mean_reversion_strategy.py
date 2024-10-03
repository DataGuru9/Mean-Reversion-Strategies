# Import necessary libraries for data handling, financial data acquisition, and visualization
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

# Define the stock ticker and date range for analysis
ticker = "AAPL"
start_date = "2021-01-01"
end_date = "2024-01-01"

# Download historical stock data using yfinance
# This provides the raw data for our analysis
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate 20-day moving average (MA) and standard deviation (STD)
# These metrics help identify the stock's typical behavior and volatility
data['MA'] = data['Close'].rolling(window=20).mean()
data['STD'] = data['Close'].rolling(window=20).std()

# Define upper and lower bounds for the trading range
# These bounds are used to generate buy and sell signals in our mean reversion strategy
data['Upper'] = data['MA'] + (data['STD'] * 2)
data['Lower'] = data['MA'] - (data['STD'] * 2)

def generate_signals(data):
    """
    Generate trading signals based on the mean reversion strategy.
    Buy when price drops below lower bound, sell when it rises above upper bound.
    """
    data['Signal'] = 0
    data.loc[data['Close'] < data['Lower'], 'Signal'] = 1  # Buy signal
    data.loc[data['Close'] > data['Upper'], 'Signal'] = -1  # Sell signal
    return data

# Apply the signal generation function to our dataset
data = generate_signals(data)

def backtest(data):
    """
    Backtest the trading strategy by calculating returns based on our signals.
    This helps evaluate the strategy's historical performance.
    """
    data['Position'] = data['Signal'].shift(1)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Position'] * data['Returns']
    return data

# Apply the backtesting function to our dataset
data = backtest(data)

# Visualize the results using mplfinance
# This creates a candlestick chart of the stock price, including our calculated metrics and trading signals
mpf.plot(data, type='line', style='charles',
         title=f'{ticker} - Mean Reversion Strategy',
         ylabel='Price',
         ylabel_lower='Returns',
         volume=True,
         figsize=(15, 10),
         panel_ratios=(6,3),
         addplot=[mpf.make_addplot(data[['MA', 'Upper', 'Lower']])],
         savefig='mean_reversion_strategy.png')

# Calculate performance metrics to assess the strategy's effectiveness
cumulative_returns = (1 + data['Strategy_Returns']).cumprod()
total_return = cumulative_returns.iloc[-1] - 1
sharpe_ratio = data['Strategy_Returns'].mean() / data['Strategy_Returns'].std() * (252 ** 0.5)

# Print the performance metrics
print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# This script demonstrates a complete workflow for developing, implementing, and evaluating a trading strategy.
# It showcases skills in financial analysis, algorithmic trading, data manipulation, and Python programming.
# The strategy's performance can be assessed using the total return and Sharpe ratio metrics.