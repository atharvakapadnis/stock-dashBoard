import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400)
def get_key_events(stock_ticker):
    """Fetches key events for a given stock ticker (dividends, ex-dividend date, splits)."""
    
    stock = yf.Ticker(stock_ticker)
    key_events = {}

    # Fetch dividends, stock splits, and ex-dividend date
    actions = stock.actions
    dividends = stock.dividends
    if isinstance(actions, pd.DataFrame) and not actions.empty:
        latest_dividend = actions["Dividends"].dropna().last_valid_index()
        latest_split = actions["Stock Splits"].dropna().last_valid_index()
        key_events["Last Dividend"] = latest_dividend.strftime("%Y-%m-%d") if latest_dividend else "N/A"
        key_events["Last Stock Split"] = latest_split.strftime("%Y-%m-%d") if latest_split else "N/A"
    else:
        key_events["Last Dividend"] = "N/A"
        key_events["Last Stock Split"] = "N/A"

    # Fetch Ex-Dividend Date
    if isinstance(dividends, pd.Series) and not dividends.empty:
        ex_dividend_date = dividends.index[-1]  # Get the most recent ex-dividend date
        key_events["Ex-Dividend Date"] = ex_dividend_date.strftime("%Y-%m-%d")
    else:
        key_events["Ex-Dividend Date"] = "N/A"

    return key_events
