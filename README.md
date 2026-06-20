# Stock Price Prediction — Backend

FastAPI backend that serves predictions from an LSTM model (exported to ONNX format).

## Project Structure

```
stock_price_prediction_backend/
├── api/
│   └── predict.py              # FastAPI app + /api/predict endpoint
├── Stock Prediction Model1.onnx # ONNX model file
├── requirements.txt
├── Procfile                    # Render start command
├── render.yaml                 # Render deployment config
└── runtime.txt                 # Python version for Render
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn api.predict:app --reload

# Test the API
# Open http://127.0.0.1:8000/api/predict?symbol=GOOG
# Swagger docs at http://127.0.0.1:8000/docs
```

## Deploy to Render

1. Push this folder to a **new GitHub repository** (e.g. `stock-price-prediction-backend`)
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` — just click **Deploy**
5. After deployment, copy your Render URL (e.g. `https://stock-price-prediction-api.onrender.com`)
6. Paste that URL into the Flutter frontend's `api_service.dart` as the `_backendUrl`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/api/predict?symbol=GOOG` | Fetch predictions for a stock |

## Response Format

```json
{
  "symbol": "GOOG",
  "dates": ["2012-01-03", "..."],
  "close": [300.5, "..."],
  "ma_50": [0, "..."],
  "ma_100": [0, "..."],
  "ma_200": [0, "..."],
  "table_data": [{ "date": "...", "close": 0, "open": 0, "high": 0, "low": 0, "volume": 0 }],
  "predictions": {
    "dates": ["2022-01-03", "..."],
    "actual": [120.5, "..."],
    "predicted": [118.2, "..."]
  }
}
```
