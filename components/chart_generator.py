import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def generate_chart(df, selected_stock):
    """Generates stock charts with toggles for indicators."""
    
    # Base graph type selection
    base_graph_type = st.selectbox("Select Base Graph Type:", ["Mountain", "Candle"])
    st.subheader(f"{selected_stock} - {base_graph_type} Chart")

    # Create columns to align checkboxes horizontally
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_moving_averages = st.checkbox("Moving Averages", value=False)
    with col2:
        show_bollinger_bands = st.checkbox("Bollinger Bands", value=False)
    with col3:
        show_rsi = st.checkbox("RSI", value=False)
    with col4:
        show_macd = st.checkbox("MACD", value=False)

    if base_graph_type == "Candle":
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)  # Ensure DatetimeIndex is timezone-naive
        
        # Custom MPLFinance Style for Black Background
        custom_style = mpf.make_mpf_style(
            base_mpf_style="charles", 
            facecolor="#000000",  # Black background
            gridcolor="gray", 
            edgecolor="white",
            figcolor="#000000",
            rc={"axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"}
        )

        # Prepare additional plots for indicators
        add_plots = []
        if show_moving_averages:
            add_plots.extend([
                mpf.make_addplot(df["SMA_50"], color="cyan", linestyle="dashed"),  
                mpf.make_addplot(df["SMA_200"], color="red", linestyle="dashed"),
                mpf.make_addplot(df["EMA_20"], color="lime", linestyle="dashed"),  
            ])
        if show_bollinger_bands:
            add_plots.extend([
                mpf.make_addplot(df["Upper_Band"], color="magenta", linestyle="dotted"),
                mpf.make_addplot(df["Lower_Band"], color="magenta", linestyle="dotted"),
            ])
        if show_macd:
            add_plots.extend([
                mpf.make_addplot(df["MACD"], color="orange", panel=1),
                mpf.make_addplot(df["Signal_Line"], color="blue", panel=1),
                mpf.make_addplot(df["MACD_Histogram"], color="gray", panel=1, type="bar"),  
            ])
        if show_rsi:
            add_plots.append(mpf.make_addplot(df["RSI"], color="yellow", panel=2))  

        fig, axes = mpf.plot(df, type="candle", volume=True, style=custom_style, returnfig=True, addplot=add_plots)
        st.pyplot(fig)

    elif base_graph_type == "Mountain":
        # Select base data (Close Price only by default)
        plot_data = df[["Close"]].copy()

        # Add indicators if toggled
        if show_moving_averages:
            plot_data["SMA_50"] = df["SMA_50"]
            plot_data["SMA_200"] = df["SMA_200"]
            plot_data["EMA_20"] = df["EMA_20"]

        if show_bollinger_bands:
            plot_data["Upper_Band"] = df["Upper_Band"]
            plot_data["Lower_Band"] = df["Lower_Band"]

        if show_rsi:
            plot_data["RSI"] = df["RSI"]

        if show_macd:
            plot_data["MACD"] = df["MACD"]
            plot_data["Signal_Line"] = df["Signal_Line"]

        st.line_chart(plot_data.astype(float).fillna(0), use_container_width=True)
