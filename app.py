import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from components.chart_generator import generate_chart
from components.data_loader import load_stock_data, calculate_technical_indicators
from components.news_fetcher import get_stock_news
from components.key_events_fetcher import get_key_events
from datetime import datetime

# Load environment variables
load_dotenv()

# Set Streamlit to wide mode
st.set_page_config(layout="wide")

# Website Heading
st.title("Stock Market Dashboard")

# Stock selection dropdown
stocks = {
    "Alphabet (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "American Airlines (AAL)": "AAL",
    "Apple (AAPL)": "AAPL",
    "Hewlett Packard Enterprise (HPE)": "HPE",
    "Intel (INTC)": "INTC",
    "Meta (META)": "META",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA"
}

selected_stock = st.selectbox("Select a stock:", list(stocks.keys()))
stock_ticker = stocks[selected_stock]


# Time period selection
time_periods = {
    "5 Days": "5D",
    "2 Weeks": "14D",
    "1 Month": "30D",
    "3 Months": "90D",
    "6 Months": "180D",
    "1 Year": "365D",
    "2 Years": "730D",
    "Max": "all",
}
selected_period = st.selectbox("Select Time Period:", list(time_periods.keys()), index=4)

# Load and process stock data
df = load_stock_data(stock_ticker)
df = calculate_technical_indicators(df)

# Filter based on time period
if selected_period != "Max":
    cutoff_date = df.index[-1] - pd.Timedelta(days=int(time_periods[selected_period][:-1]))
    df = df[df.index >= cutoff_date]

# Create layout for graph and data table
col1, col2, col3 = st.columns([4.3, 2, 2])

with col1:
    selected_model = generate_chart(df, selected_stock, stock_ticker)

with col2:
    # Display stock data table
    st.subheader("Latest Prices")
    df_display = df.copy()
    df_display.index = pd.to_datetime(df_display.index, utc=True).tz_localize(None).strftime("%Y-%m-%d")
    df_display = df_display.sort_index(ascending=False)
    for col in ["Open", "Close", "Low", "High", "Volume"]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}")
    st.dataframe(df_display[["Open", "Close", "Low", "High", "Volume"]].head(10))
    
with col3:
    # Fetch and display key events
    st.subheader(f"Key Events")
    key_events = get_key_events(stock_ticker)

    for event, date in key_events.items():
        st.markdown(f"**{event}:** {date}")
    

st.markdown("---")

# Fetch stock news

news = get_stock_news(stock_ticker)

st.subheader(f"Latest News on {selected_stock}")
for article in news:
    st.markdown(f"**[{article['title']}]({article['url']})**")
    st.caption(f"{article['source']['name']} - {article['publishedAt'][:10]}")
    st.write(article["description"])
    st.markdown("---")

st.subheader("About This Project")
st.markdown(
    """
    **Stock Market Dashboard** is an interactive platform I built for **stock analysis and predictive modeling**.  
    It has allowed me to explore historical stock trends, visualize key technical indicators, and evaluate deep learning-based stock price predictions.

    ---

    ## Research Background  
    This project is an extension of my **time series analysis research initiative**, focused on stock price forecasting.  
    Throughout this research, I explored a variety of methodologies, including:

    - **Classical models**: ARIMA, VAR  
    - **Deep learning models**: LSTM, GRU  
    - **Hybrid models**: ARIMA + LSTM, XGBoost + LSTM, CNN + LSTM  
    - **Advanced techniques**: Graph Neural Networks, AutoML  

    **Key Achievements:**
    - Developed **predictive models** tailored for multiple stocks  
    - Integrated **technical indicators** to improve forecasting accuracy  
    - Experimented with **hybrid modeling techniques** to enhance predictive performance  

    ---

    ## Disclaimer  
    This dashboard is a **learning project** I created for research and educational purposes.  
    It **does not provide financial advice** and should not be used for actual trading or investment decisions.  

    **Check out the source code on [GitHub](https://github.com/atharvakapadnis/stock-forecasting-ai) for more details.**
    """
)