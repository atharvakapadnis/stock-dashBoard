import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st

# Load environment variables
load_dotenv()

NEWS_API_KEY = "d3625e265719408da7ffaf7cd03ef924"
NEWS_URL = "https://newsapi.org/v2/everything"

@st.cache_data(ttl=86400)
def get_stock_news(stock_ticker):
    """Fetches the latest news articles for a given stock ticker."""

    # Map stock tickers to their full company names
    company_names = {
        "NVDA": "NVIDIA Corporation NASDAQ:NVDA",
        "AAPL": "Apple Inc. NASDAQ:AAPL",
        "MSFT": "Microsoft Corporation NASDAQ:MSFT",
        "AAL": "American Airlines Group Inc. NASDAQ:AAL",
        "GOOGL": "Alphabet Inc. NASDAQ:GOOGL",
        "TSLA": "Tesla Inc. NASDAQ:TSLA",
        "AMZN": "Amazon.com Inc. NASDAQ:AMZN",
        "META": "Meta Platforms Inc. NASDAQ:META",
        "INTC": "Intel Corporation NASDAQ:INTC",
        "HPE": "Hewlett Packard Enterprise NYSE:HPE",  # Added HPE
    }

    # Get the company name corresponding to the ticker
    company_name = company_names.get(stock_ticker, stock_ticker)

    # Define API request parameters
    params = {
        "q": f'"{company_name}" OR "{stock_ticker}"',
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "from": (datetime.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
    }

    # Fetch news data
    response = requests.get(NEWS_URL, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])[:3]  # Return top 3 articles
    return []
