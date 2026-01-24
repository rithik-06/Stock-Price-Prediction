"""
fetch stock data from yahoo finance
this scripts downloads historical stock prices
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

def fetch_stock_data(ticker: str ,period: str = "5y", output_dir: str= None):
    """
    Args:
     ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
        period: Time period ('1d', '5d', '1mo', '1y', '5y')
        output_dir: Directory to save the CSV file

    :return:
    dataframe with stock data
    """

    print(f"fetching data for {ticker}...")

    try:



