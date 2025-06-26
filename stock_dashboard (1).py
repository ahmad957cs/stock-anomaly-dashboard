import streamlit as st
import pandas as pd
import yfinance as yf
import ta
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Anomaly Detection", layout="wide")
st.title("📈 Stock Anomaly Detection Dashboard")
st.markdown("Detect anomalies in stock price trends using technical indicators, Isolation Forest, and Prophet.")

# --- Sidebar Inputs ---
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# --- Load Data ---
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)  # Make sure Date is a column
    return df

data = load_data(ticker, start_date, end_date)

# --- Validate Data ---
if data.empty or 'Date' not in data.columns or 'Close' not in data.columns:
    st.error("⚠️ No data found or missing 'Date'/'Close' columns. Please check the ticker and date range.")
    st.stop()

# --- Price Chart ---
st.subheader(f"{ticker} Stock Price")
st.line_chart(data[['Date', 'Close']].set_index('Date'))

# --- Add Technical Indicators ---
close = data['Close']
data['SMA_20'] = ta.trend.SMAIndicator(close=close).sma_indicator()
data['EMA_20'] = ta.trend.EMAIndicator(close=close).ema_indicator()
data['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
bb = ta.volatility.BollingerBands(close=close)
data['BB_high'] = bb.bollinger_hband()
data['BB_low'] = bb.bollinger_lband()
data.dropna(inplace=True)

# --- Isolation Forest Anomaly Detection ---
features = data[['Close', 'SMA_20', 'EMA_20', 'RSI', 'BB_high', 'BB_low']]
iso_model = IsolationForest(contamination=0.01, random_state=42)
data['iso_anomaly'] = iso_model.fit_predict(features)
data['iso_anomaly'] = data['iso_anomaly'].map({1: 0, -1: 1})

st.subheader("🔍 Isolation Forest Anomalies")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(data['Date'], data['Close'], label='Close Price')
ax1.scatter(data[data['iso_anomaly'] == 1]['Date'],
            data[data['iso_anomaly'] == 1]['Close'],
            color='red', label='Anomaly', s=50)
ax1.legend()
st.pyplot(fig1)

# --- Prophet Forecast & Anomalies ---
st.subheader("🔮 Prophet Forecast Anomalies")

# Prepare data for Prophet
prophet_data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
prophet_data['y'] = prophet_data['y'].astype(float)

# Train and Forecast
model = Prophet(daily_seasonality=True)
model.fit(prophet_data)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Merge forecast with actual
merged = pd.merge(prophet_data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                  on='ds', how='left')
merged['prophet_anomaly'] = 0
merged.loc[merged['y'] > merged['yhat_upper'], 'prophet_anomaly'] = 1
merged.loc[merged['y'] < merged['yhat_lower'], 'prophet_anomaly'] = -1

# Plot
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(merged['ds'], merged['y'], label='Actual', color='blue')
ax2.plot(merged['ds'], merged['yhat'], label='Forecast', color='orange')
ax2.fill_between(merged['ds'], merged['yhat_lower'], merged['yhat_upper'], alpha=0.3, color='orange')
ax2.scatter(merged[merged['prophet_anomaly'] == 1]['ds'],
            merged[merged['prophet_anomaly'] == 1]['y'],
            color='red', label='Spike')
ax2.scatter(merged[merged['prophet_anomaly'] == -1]['ds'],
            merged[merged['prophet_anomaly'] == -1]['y'],
            color='green', label='Drop')
ax2.legend()
st.pyplot(fig2)

# --- End ---
st.success("✅ Anomaly detection complete.")
