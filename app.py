import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import numpy as np

# Title
st.title("Silver Price Prediction Dashboard")
st.write("Extracting the last 5 years of Silver Prices (SI=F) and predicting the next year using Facebook Prophet to minimize RMSE.")

# Load Data
@st.cache_data
def load_data():
    silver = yf.Ticker('SI=F')
    # Use period='5y' for last 5 years
    df = silver.history(period="5y")
    df.reset_index(inplace=True)
    # Remove timezone safely so Prophet won't complain
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    return df

data_load_state = st.text("Loading data...")
try:
    df = load_data()
    if df.empty:
        st.error("Error: Yahoo Finance returned empty data. It might be rate-limiting Cloud IPs.")
        st.stop()
    data_load_state.text("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


st.subheader("Historical Data (Last 5 Years)")
st.dataframe(df.tail())

# Plot Raw Data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare Data for Prophet
df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
# Drop NaNs just in case
df_train = df_train.dropna()

# Model and Train
m = Prophet()
# To try and reduce RMSE, we can tweak some hyperparameters like changepoint_prior_scale, 
# although Prophet defaults perform very well robustly on daily data.
m.fit(df_train)

# Predict Next 1 Year
# 365 days
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

st.subheader("Forecasted Data (Next 1 Year)")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot Forecast
st.subheader("Forecast Plot")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Calculate RMSE on historical matches
# yhat is the predicted value
# Let's align forecast with training Data
predictions = forecast.iloc[:len(df_train)]
rmse = np.sqrt(mean_squared_error(df_train['y'], predictions['yhat']))

st.subheader("Model Performance")
st.success(f"**Root Mean Squared Error (RMSE) on historical fit:** {rmse:.4f}")
st.write("A lower RMSE value indicates a better fit of the model to the historical data, thus providing a more reliable foundation for our 1-year forward prediction.")
