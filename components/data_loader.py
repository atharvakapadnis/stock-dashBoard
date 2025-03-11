import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent

DATA_FILES = {
    "NVDA": SCRIPT_DIR / "web_data" / "nvidia_stock_data.csv",
    "AAPL": SCRIPT_DIR / "web_data" / "apple_stock_data.csv",
    "MSFT": SCRIPT_DIR / "web_data" / "microsoft_stock_data.csv",
}

def load_stock_data(ticker):
    """Loads stock data from CSV files."""
    data_file = DATA_FILES[ticker]
    df = pd.read_csv(data_file, parse_dates=["Date"], index_col="Date")
    return df

def calculate_technical_indicators(df):
    """Calculates moving averages and Bollinger Bands."""
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Upper_Band"] = df["Close"].rolling(window=20).mean() + (df["Close"].rolling(window=20).std() * 2)
    df["Lower_Band"] = df["Close"].rolling(window=20).mean() - (df["Close"].rolling(window=20).std() * 2)
    return df
