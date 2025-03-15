import streamlit as st
import pandas as pd
import os

# Get the absolute path of the project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "results"))

# Define paths for predictions and metrics
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

def load_predictions(ticker, model_type):
    """Loads predictions from CSV based on selected stock and model type."""
    file_path = os.path.join(PREDICTIONS_DIR, f"{ticker}_{model_type}_results.csv")
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    else:
        st.warning(f"No predictions found for {ticker} using {model_type.upper()}.")
        return None

def load_metrics(ticker, model_type):
    """Loads model performance metrics from CSV based on selected stock and model type."""
    file_path = os.path.join(METRICS_DIR, f"{ticker}_{model_type}_metrics.csv")
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.warning(f"No metrics found for {ticker} using {model_type.upper()}.")
        return None

def generate_chart(df, selected_stock, stock_ticker):
    """Generates stock charts with toggles for indicators and predictive models."""

    # Select Graph Type
    base_graph_type = st.selectbox(
        "Select Chart Type:", 
        ["Mountain", "Candle", "Trend Predictive Model"]
    )
    
    st.subheader(f"{selected_stock} - {base_graph_type} Chart")

    if base_graph_type == "Trend Predictive Model":
        # Allow user to select model type
        model_type = st.selectbox("Select Predictive Model:", ["LSTM", "GRU"]).lower()
        
        # Load predictions
        predictions_df = load_predictions(stock_ticker, model_type)

        if predictions_df is not None:
            # Merge actual and predicted close prices
            merged_df = df[["Close"]].merge(predictions_df, left_index=True, right_index=True, how="inner")
            merged_df.rename(columns={"Predicted": "Predicted Close"}, inplace=True)

            # Convert index to string for Streamlit compatibility
            merged_df.index = pd.to_datetime(merged_df.index, utc=True).tz_localize(None)

            # Use st.line_chart() for interactive plotting
            st.subheader(f"{selected_stock} - {model_type.upper()} Predictions vs. Actual")
            st.line_chart(merged_df[["Close", "Predicted Close"]])

        else:
            st.warning(f"No predictions available for {selected_stock} using {model_type.upper()}.")

    else:
        # Existing functionality for Mountain/Candle charts
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            show_moving_averages = st.checkbox("Moving Averages", value=False)
        with col2:
            show_bollinger_bands = st.checkbox("Bollinger Bands", value=False)
        with col3:
            show_rsi = st.checkbox("RSI", value=False)
        with col4:
            show_macd = st.checkbox("MACD", value=False)

        if base_graph_type == "Mountain":
            plot_data = df[["Close"]].copy()
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

        elif base_graph_type == "Candle":
            import mplfinance as mpf
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

            custom_style = mpf.make_mpf_style(
                base_mpf_style="charles",
                facecolor="#000000",
                gridcolor="gray",
                edgecolor="white",
                figcolor="#000000",
                rc={"axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"}
            )

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
