import yfinance as yf
import pandas as pd
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

DATA_FILES = {
    "NVDA": SCRIPT_DIR.parent / "web_data" / "nvidia_stock_data.csv",
    "AAPL": SCRIPT_DIR.parent / "web_data" / "apple_stock_data.csv",
    "MSFT": SCRIPT_DIR.parent / "web_data" / "microsoft_stock_data.csv",
    "AAL": SCRIPT_DIR.parent / "web_data" / "aal_stock_data.csv",
    "GOOGL": SCRIPT_DIR.parent / "web_data" / "googl_stock_data.csv",
    "TSLA": SCRIPT_DIR.parent / "web_data" / "tsla_stock_data.csv",
    "AMZN": SCRIPT_DIR.parent / "web_data" / "amzn_stock_data.csv",
    "HPE": SCRIPT_DIR.parent / "web_data" / "hpe_stock_data.csv",
    "META": SCRIPT_DIR.parent / "web_data" / "meta_stock_data.csv",
    "INTC": SCRIPT_DIR.parent / "web_data" / "intc_stock_data.csv",
}


# Function to pull max interval data
def fetch_max_data(ticker):
    data_file = DATA_FILES[ticker]

    # Skip if file exists
    if os.path.exists(data_file):
        print(f"Data for {ticker} already exists. Skipping Download.")
        return

    # Fetch Data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    hist.reset_index(inplace=True)

    # Save to CSV
    hist.to_csv(data_file, index=False)
    print(f"Data for {ticker} downloaded successfully.")


# Fetch data for NVDA, AAPL, and MSFT
for ticker in DATA_FILES.keys():
    fetch_max_data(ticker)
