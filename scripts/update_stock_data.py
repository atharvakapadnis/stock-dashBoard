import yfinance as yf
import pandas as pd
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


def update_stock_data(ticker):
    data_file = DATA_FILES[ticker]

    #  Read the CSV with 'Date' as a parsed datetime column, then set as index.
    df = pd.read_csv(data_file, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    #  Fetch the latest stock data. By default, yfinance gives a DatetimeIndex.
    stock = yf.Ticker(ticker)
    latest_data = stock.history(period="1d", auto_adjust=False)

    #  Keep only the columns we need.
    columns_to_keep = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]
    latest_data = latest_data[columns_to_keep]

    # Check if the last fetched date is already in our existing data.
    last_date = latest_data.index[-1]
    if last_date in df.index:
        print(f"Data for {ticker} is already up to date.")
        return

    # Concatenate the new data and write back to CSV.
    df = pd.concat([df, latest_data])
    df.to_csv(data_file)
    print(f"Data for {ticker} updated successfully.")


# Update stock data for all tickers
for ticker in DATA_FILES:
    update_stock_data(ticker)
