from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import time

app = FastAPI(title="Stock Price Prediction API", version="1.0.0")

# Allow CORS so that the Flutter Web frontend can call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model at startup — resolved relative to the project root
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Stock Prediction Model1.onnx")
session = ort.InferenceSession(MODEL_PATH)

# In-memory response cache
CACHE = {}  # key: symbol, value: (timestamp, data_dict)
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))


@app.get("/")
def root():
    return {"message": "Stock Price Prediction API is running. Use /api/predict?symbol=GOOG"}


@app.get("/api/predict")
def predict(symbol: str = Query("GOOG", description="Stock Ticker Symbol")):
    symbol = symbol.strip().upper()

    # Cache lookup
    now = time.time()
    if symbol in CACHE:
        cached_time, cached_data = CACHE[symbol]
        if now - cached_time < CACHE_TTL:
            return cached_data

    start = "2012-01-01"

    # 1. Fetch stock data using yfinance
    try:
        data = yf.download(symbol, start=start)
    except Exception as e:
        return {"error": f"Failed to download data: {str(e)}"}

    if data.empty:
        return {"error": f"No stock data found for ticker symbol: {symbol}"}

    # Flatten MultiIndex columns if present (yfinance might return MultiIndex for single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 2. Parse series (Close prices and dates)
    close_prices = data["Close"].values.flatten().tolist()
    dates = [str(d.date()) for d in data.index]

    # Prepare raw table data (Vectorized)
    temp_df = data[['Close', 'Open', 'High', 'Low', 'Volume']].fillna(0).copy()
    temp_df['date'] = temp_df.index.strftime('%Y-%m-%d')
    temp_df = temp_df.rename(columns={
        'Close': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    temp_df['volume'] = temp_df['volume'].astype(int)
    temp_df['close'] = temp_df['close'].astype(float)
    temp_df['open'] = temp_df['open'].astype(float)
    temp_df['high'] = temp_df['high'].astype(float)
    temp_df['low'] = temp_df['low'].astype(float)
    table_data = temp_df[['date', 'close', 'open', 'high', 'low', 'volume']].iloc[::-1].to_dict(orient='records')

    # 3. Calculate Moving Averages (filled with 0 for initial values)
    ma_50 = data["Close"].rolling(50).mean().fillna(0).values.flatten().tolist()
    ma_100 = data["Close"].rolling(100).mean().fillna(0).values.flatten().tolist()
    ma_200 = data["Close"].rolling(200).mean().fillna(0).values.flatten().tolist()

    # 4. Prepare test data for prediction (80/20 split)
    split_idx = int(len(data) * 0.80)
    data_train = pd.DataFrame(data["Close"][0:split_idx])
    data_test = pd.DataFrame(data["Close"][split_idx:])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)

    past_100_days = data_train.tail(100)
    data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.transform(data_test_combined)

    M = len(data_test_scaled)
    if M >= 100:
        idx = np.arange(100) + np.arange(M - 100)[:, None]
        x_test = data_test_scaled[idx]  # Shape: (M - 100, 100, 1)
        y_test_actual = data_test_scaled[100:, 0]  # Shape: (M - 100,)
    else:
        x_test = np.array([], dtype=np.float32)
        y_test_actual = np.array([], dtype=np.float32)

    x_test = np.array(x_test, dtype=np.float32)  # Shape: (N, 100, 1)

    y_predicted_prices = []
    y_actual_prices = []
    test_dates = []

    if len(x_test) > 0:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        outputs = session.run([output_name], {input_name: x_test})
        y_predicted = outputs[0]  # Shape: (N, 1)

        scale_factor = 1.0 / scaler.scale_[0]
        y_predicted_prices = (y_predicted.flatten() * scale_factor).tolist()
        y_actual_prices = (np.array(y_test_actual) * scale_factor).tolist()
        test_dates = [str(d.date()) for d in data.index[split_idx:]]

    response_data = {
        "symbol": symbol,
        "dates": dates,
        "close": close_prices,
        "ma_50": ma_50,
        "ma_100": ma_100,
        "ma_200": ma_200,
        "table_data": table_data,
        "predictions": {
            "dates": test_dates,
            "actual": y_actual_prices,
            "predicted": y_predicted_prices,
        },
    }

    # Store successful prediction in the cache
    CACHE[symbol] = (now, response_data)

    return response_data
