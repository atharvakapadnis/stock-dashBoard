import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent

DATA_FILES = {
    "NVDA": SCRIPT_DIR / "web_data" / "nvidia_stock_data.csv",
    "AAPL": SCRIPT_DIR / "web_data" / "apple_stock_data.csv",
    "MSFT": SCRIPT_DIR / "web_data" / "microsoft_stock_data.csv",
    "AAL": SCRIPT_DIR / "web_data" / "aal_stock_data.csv",
    "GOOGL": SCRIPT_DIR / "web_data" / "googl_stock_data.csv",
    "TSLA": SCRIPT_DIR / "web_data" / "tsla_stock_data.csv",
    "AMZN": SCRIPT_DIR / "web_data" / "amzn_stock_data.csv",
    "HPE": SCRIPT_DIR / "web_data" / "hpe_stock_data.csv",
    "META": SCRIPT_DIR / "web_data" / "meta_stock_data.csv",
    "INTC": SCRIPT_DIR / "web_data" / "intc_stock_data.csv",
}

def load_stock_data(ticker):
    """Loads stock data from CSV files."""
    data_file = DATA_FILES[ticker]
    df = pd.read_csv(data_file, parse_dates=["Date"], index_col="Date")
    return df

def calculate_technical_indicators(df):
    #  Calculates moving averages.
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    # Calculates Bollinger Bands.
    df["Upper_Band"] = df["Close"].rolling(window=20).mean() + (df["Close"].rolling(window=20).std() * 2)
    df["Lower_Band"] = df["Close"].rolling(window=20).mean() - (df["Close"].rolling(window=20).std() * 2)

    # Calculate RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Calculate MACD
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]
    return df
