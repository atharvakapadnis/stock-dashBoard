import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def generate_chart(df, selected_stock):
    """Generates stock charts based on user selection."""
    graph_type = st.selectbox("Select Graph Type:", ["Base Graph", "Moving Averages", "Bollinger Bands", "Upcoming Feature"])
    
    if graph_type == "Base Graph":
        base_graph_type = st.selectbox("Select Base Graph Type:", ["Mountain", "Candle"])
        st.subheader(f"{selected_stock} - {base_graph_type} Chart")

        if base_graph_type == "Candle":
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)  # Ensure DatetimeIndex is timezone-naive
            fig, axes = mpf.plot(df, type="candle", volume=True, style="charles", returnfig=True)
            st.pyplot(fig)

        elif base_graph_type == "Mountain":
            st.line_chart(df["Close"].astype(float).fillna(0), use_container_width=True)

    elif graph_type == "Moving Averages":
        st.subheader(f"{selected_stock} - Moving Averages")
        st.line_chart(df[["Close", "SMA_50", "SMA_200", "EMA_20"]])

    elif graph_type == "Bollinger Bands":
        st.subheader(f"{selected_stock} - Bollinger Bands")
        st.line_chart(df[["Close", "Upper_Band", "Lower_Band"]])

    elif graph_type == "Upcoming Feature":
        st.subheader("🚀 New Features Coming Soon!")
