import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st

# Load environment variables
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = "https://newsapi.org/v2/everything"

@st.cache_data(ttl=86400)
def get_stock_news(stock_ticker, company_name):
    params = {
        "q": f'"{company_name}" OR "{stock_ticker}"',
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "from": (datetime.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
    }
    response = requests.get(NEWS_URL, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])[:3]
    return []
