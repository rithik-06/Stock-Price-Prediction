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
        # Download stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            print("No data found for " + ticker)
            return None

        print("Fetched " + str(len(df)) + " records")
        print("Date range: " + str(df.index[0].date()) + " to " + str(df.index[-1].date()))

        # Save to CSV if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = output_path / (ticker + "_" + timestamp + ".csv")

            df.to_csv(filename)
            print("Saved to: " + str(filename))


        return df

    except Exception as e:
        print("Error fetching data for " + ticker + ": " + str(e))
        raise

if __name__ == "__main__":
    # Test the function
    fetch_stock_data("AAPL", period="1mo", output_dir="../../data/raw")

import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(project_root, "data", "raw")
fetch_stock_data("AAPL", period="1mo", output_dir=data_dir)




