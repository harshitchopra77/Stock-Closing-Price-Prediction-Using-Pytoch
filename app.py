import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


class StockLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(1, 50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(50, 60, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)

        self.lstm3 = nn.LSTM(60, 80, batch_first=True)
        self.dropout3 = nn.Dropout(0.4)

        self.lstm4 = nn.LSTM(80, 120, batch_first=True)
        self.dropout4 = nn.Dropout(0.5)

        self.fc = nn.Linear(120, 1)

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = self.dropout3(x)

        x, _ = self.lstm4(x)
        x = self.dropout4(x)

        x = x[:, -1, :]
        x = self.fc(x)

        return x


model = StockLSTM()
model.load_state_dict(torch.load("stock_model.pth", map_location="cpu"))
model.eval()


st.header('Stock Closing Price Predictor')

stock = st.text_input('Enter Stock Symbol', '')

start = '2015-01-01'
end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

if not stock:
    st.info('Please enter a stock symbol to begin.')
    st.stop()

data = yf.download(stock, start, end)

# ── prepare scaler ──────────────────────────────────────────────
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

past_100_days = data_train.tail(100)
data_test_concat = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test_concat)

# ── build test sequences ─────────────────────────────────────────
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x = np.array(x)
y = np.array(y)

x_tensor = torch.tensor(x, dtype=torch.float32)

with torch.no_grad():
    predict = model(x_tensor).numpy()

predict = scaler.inverse_transform(predict)
y = scaler.inverse_transform(y.reshape(-1, 1))

predict = predict.flatten()
y = y.flatten()

# ── tomorrow's prediction ────────────────────────────────────────
last_100_days = data[['Close']].tail(100)
last_100_scaled = scaler.transform(last_100_days.values) 
input_seq = torch.tensor(
    last_100_scaled.reshape(1, 100, 1),
    dtype=torch.float32
)

with torch.no_grad():
    tomorrow_scaled = model(input_seq).numpy()

tomorrow_price = scaler.inverse_transform(tomorrow_scaled)


def get_next_trading_day():
    today = datetime.today()
    tomorrow = today + timedelta(days=1)
    if tomorrow.weekday() == 5:    # Saturday → Monday
        tomorrow = tomorrow + timedelta(days=2)
    elif tomorrow.weekday() == 6:  # Sunday → Monday
        tomorrow = tomorrow + timedelta(days=1)
    return tomorrow.strftime('%d %B %Y')


next_trading_day = get_next_trading_day()




# 1 ── Tomorrow's Predicted Price ────────────────────────────────
st.subheader("Tomorrow's Predicted Closing Price")
st.metric(
    label=f"Predicted closing price for {next_trading_day}",
    value=f"${tomorrow_price[0][0]:.2f}"
)

st.divider()

# 2 ── Predicted vs Actual ────────────────────────────────────────
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'r', label='Original Price')
plt.plot(predict, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

st.divider()

# 3 ── Moving Average Charts ──────────────────────────────────────
ma_50_days  = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

st.subheader('Price vs MA50')
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days,  'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close,  'g', label='Close Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close,  'g', label='Close Price')
plt.legend()
st.pyplot(fig3)

st.divider()

# 4 ── Raw Data ───────────────────────────────────────────────────
st.subheader('Stock Data')
st.write(data)