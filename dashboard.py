import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from datetime import date

# Title and Greeting
st.set_page_config(page_title="Personalized Anomaly Detection Dashboard", layout="wide")
st.title("📈 Anomaly Detection Dashboard")

# Sidebar controls with tooltips
st.sidebar.header("🔧 Controls")

# Ticker selection with popular options and search bar
ticker_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
selected_ticker = st.sidebar.selectbox("Select a Popular Ticker", ticker_options)
custom_ticker = st.sidebar.text_input("Or Enter Any Company Ticker")

# Determine final ticker based on user input
final_ticker = custom_ticker.upper() if custom_ticker else selected_ticker

# Date Range Selection
start_date = st.sidebar.date_input("Start Date", date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2024, 1, 1))

if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

# Anomaly Threshold Slider
anomaly_threshold = st.sidebar.slider(
    "Set Anomaly Detection Threshold", 0.01, 0.5, 0.05, step=0.01, help="Adjust the threshold to detect more or fewer anomalies."
)

# Fetch data function
@st.cache_data
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Validate ticker and fetch data
if final_ticker:
    try:
        data = fetch_data(final_ticker, start_date, end_date)
        if data.empty:
            st.error(f"No data found for ticker: {final_ticker}")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data for ticker '{final_ticker}'. Please check the ticker symbol.")
        st.stop()
else:
    st.error("Please select or enter a valid company ticker.")
    st.stop()

# Feature Engineering
def add_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['ATR'] = data['Close'].rolling(window=14).std()
    data['Momentum'] = data['Close'].diff(4)
    data.dropna(inplace=True)
    return data

data = add_technical_indicators(data)

# Scaling and PCA
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'ATR', 'Momentum']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Anomaly Detection
clf = IsolationForest(contamination=anomaly_threshold)
data['anomaly'] = clf.fit_predict(scaled_data)
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})
data['anomaly_score'] = clf.decision_function(scaled_data)

# Sidebar for Visualization Selection
st.sidebar.header("📊 Choose Visualization")
visualization_options = [
    "Summary Statistics",
    "Closing Price with Anomalies",
    "Error Histogram (Anomaly Scores)",
    "Anomaly Clusters (Close Price vs Volume)",
    "Trading Volume Over Time",
    "SMA vs EMA Comparison",
    "PCA Anomaly Clusters"
]
selected_visualization = st.sidebar.radio("Select a Visualization", visualization_options)

# Conditional rendering for each visualization
# Summary Statistics
if selected_visualization == "Summary Statistics":
    st.subheader(f"📊 {final_ticker} Summary Statistics")

    try:
        # Ensure all values are extracted correctly and are numeric
        recent_close = float(data['Close'].iloc[-1])

        anomalies_detected = int(data['anomaly'].sum()) if 'anomaly' in data.columns else 0
        anomaly_percentage = (anomalies_detected / len(data)) * 100 if len(data) > 0 else 0

        avg_volume = float(data['Volume'].mean()) if 'Volume' in data.columns else 0.0
        daily_returns = data['Close'].pct_change() if 'Close' in data.columns else pd.Series([0])
        volatility = float(daily_returns.std() * np.sqrt(252)) if not daily_returns.empty else 0.0

        latest_rsi = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns and not data['RSI'].isna().all() else "N/A"
        price_vs_sma_20 = (float(data['Close'].iloc[-1] - data['SMA_20'].iloc[-1]) / data['SMA_20'].iloc[-1] * 100) if 'SMA_20' in data.columns and not data['SMA_20'].isna().all() else "N/A"
        price_vs_sma_50 = (float(data['Close'].iloc[-1] - data['SMA_50'].iloc[-1]) / data['SMA_50'].iloc[-1] * 100) if 'SMA_50' in data.columns and not data['SMA_50'].isna().all() else "N/A"

        # Display metrics
        st.metric(label="Most Recent Close Price", value=f"${recent_close:.2f}")
        st.metric(label="Anomalies Detected", value=anomalies_detected)
        st.metric(label="Average Trading Volume", value=f"{avg_volume:,.0f}")
        st.metric(label="Annualized Volatility", value=f"{volatility:.2%}")
        st.metric(label="Latest RSI", value=latest_rsi if isinstance(latest_rsi, float) else latest_rsi)
        st.metric(label="Price vs 20-Day SMA", value=f"{price_vs_sma_20:.2f}%" if isinstance(price_vs_sma_20, float) else "N/A")
        st.metric(label="Price vs 50-Day SMA", value=f"{price_vs_sma_50:.2f}%" if isinstance(price_vs_sma_50, float) else "N/A")
        st.metric(label="Percentage of Anomalies", value=f"{anomaly_percentage:.2f}%")

    except Exception as e:
        st.error(f"Error in displaying summary statistics: {e}")

elif selected_visualization == "Closing Price with Anomalies":
    st.subheader("Closing Price with Anomalies Highlighted")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], color="blue", label="Closing Price")
    anomaly_points = data[data['anomaly'] == 1]
    ax.scatter(anomaly_points.index, anomaly_points['Close'], color="red", label="Anomalies", s=50)
    ax.set_title("Closing Price with Anomalies Highlighted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    ax.legend()
    st.pyplot(fig)

elif selected_visualization == "Error Histogram (Anomaly Scores)":
    st.subheader("Error Histogram (Anomaly Scores)")
    fig, ax = plt.subplots()
    sns.histplot(data[data['anomaly'] == 1]['anomaly_score'], bins=20, color="red", label="Anomalies", ax=ax)
    sns.histplot(data[data['anomaly'] == 0]['anomaly_score'], bins=20, color="blue", label="Normal", ax=ax)
    ax.set_title("Histogram of Anomaly Scores")
    ax.legend()
    st.pyplot(fig)

elif selected_visualization == "Anomaly Clusters (Close Price vs Volume)":
    st.subheader("Anomaly Clusters")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['Close'].values.ravel(), y=data['Volume'].values.ravel(), hue=data['anomaly'].values.ravel(), palette={0: "blue", 1: "red"}, ax=ax)
    ax.set_xlabel("Close Price")
    ax.set_ylabel("Volume")
    ax.set_title("Anomaly Clusters by Close Price and Volume")
    st.pyplot(fig)

elif selected_visualization == "Trading Volume Over Time":
    st.subheader("Trading Volume Over Time")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Volume'], color="green", label="Volume")
    ax.set_title("Trading Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend()
    st.pyplot(fig)

elif selected_visualization == "SMA vs EMA Comparison":
    st.subheader("SMA vs EMA Comparison")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], color="blue", label="Closing Price")
    ax.plot(data.index, data['SMA_20'], color="orange", label="SMA 20")
    ax.plot(data.index, data['EMA_20'], color="purple", label="EMA 20")
    ax.set_title("SMA vs EMA Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

elif selected_visualization == "PCA Anomaly Clusters":
    st.subheader("PCA Anomaly Clusters")
    fig, ax = plt.subplots()
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=data['anomaly'], palette={0: "blue", 1: "red"}, ax=ax)
    ax.set_title("Anomaly Clusters in PCA Space")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    st.pyplot(fig)

