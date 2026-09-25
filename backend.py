"""
Wharton Sim QuantStats - Python FastAPI Backend
================================================
Analytical Helper for the Wharton Global High School Investment Competition (StockTrak Simulator).
Fetches historical market data, computes technical quantitative indicators using pandas_ta,
and applies a machine learning Random Forest classifier to forecast 3-day directional bias.

Run with:
    python backend.py
    or
    uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
"""

import sys
import re
from typing import Optional, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yfinance as yf
import pandas as pd
import numpy as np

# Try importing pandas_ta; if not installed, provide resilient native fallback
try:
    import pandas_ta as ta  # type: ignore
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from sklearn.ensemble import RandomForestClassifier

app = FastAPI(
    title="Wharton Sim QuantStats API",
    description="Quantitative breakdown and ML bias classifier for StockTrak / Wharton competition assets.",
    version="1.0.0"
)

# Enable CORS fully for Chrome extension and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes required quantitative indicators:
    - 20 EMA, 50 EMA, 200 EMA
    - RSI (14)
    - MACD (12, 26, 9)
    - ADX (14)
    - ATR (14)
    - Bollinger Bands (20, 2)
    - 20 SMA of Volume
    """
    df = df.copy()

    # Ensure required columns are float and lowercase standardized
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if HAS_PANDAS_TA:
        try:
            # 20, 50, 200 EMAs
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)

            # RSI (14)
            df.ta.rsi(length=14, append=True)

            # MACD (12, 26, 9)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)

            # ADX (14)
            df.ta.adx(length=14, append=True)

            # ATR (14)
            df.ta.atr(length=14, append=True)

            # Bollinger Bands (20, 2)
            df.ta.bbands(length=20, std=2, append=True)
        except Exception as e:
            print(f"[Warning] pandas_ta error: {e}. Falling back to native calculation.", file=sys.stderr)

    # Resilient native calculations for any missing columns
    if 'EMA_20' not in df.columns:
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    if 'EMA_50' not in df.columns:
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    if 'EMA_200' not in df.columns:
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # RSI (14)
    if 'RSI_14' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss.replace(0, np.nan))
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = df['RSI_14'].fillna(50.0)

    # MACD Histogram
    macd_hist_col = [c for c in df.columns if c.startswith('MACDh_')]
    if macd_hist_col:
        df['MACD_HIST'] = df[macd_hist_col[0]]
    else:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = macd - signal

    # ADX (14)
    adx_col = [c for c in df.columns if c.startswith('ADX_')]
    if adx_col:
        df['ADX_14'] = df[adx_col[0]]
    else:
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        up_move = df['High'] - df['High'].shift()
        down_move = df['Low'].shift() - df['Low']
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        pos_di = 100 * (pd.Series(pos_dm, index=df.index).rolling(14).mean() / atr14)
        neg_di = 100 * (pd.Series(neg_dm, index=df.index).rolling(14).mean() / atr14)
        dx = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan))
        df['ADX_14'] = dx.rolling(14).mean().fillna(25.0)

    # ATR (14)
    atr_cols = [c for c in df.columns if c.startswith('ATR')]
    if atr_cols:
        df['ATR_14'] = df[atr_cols[0]]
    else:
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()

    # Bollinger Bands & Bandwidth
    bbb_cols = [c for c in df.columns if c.startswith('BBB_')]
    if bbb_cols:
        df['BB_WIDTH'] = df[bbb_cols[0]]
    else:
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        upper = sma20 + (std20 * 2)
        lower = sma20 - (std20 * 2)
        df['BB_WIDTH'] = ((upper - lower) / sma20) * 100

    # 20 SMA of Volume
    df['VOL_SMA_20'] = df['Volume'].rolling(window=20).mean()

    return df


def train_ml_classifier(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Builds a binary target where 1 means the price went up 3 days into the future.
    Trains a scikit-learn RandomForestClassifier on historical features:
    RSI, MACD Hist, ADX, Bollinger Band Width.
    Predicts the upward probability of the current data point.
    """
    df = df.copy()

    # 3-day forward return binary target
    df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(float)

    feature_cols = ['RSI_14', 'MACD_HIST', 'ADX_14', 'BB_WIDTH']
    train_df = df.dropna(subset=feature_cols + ['Target'])

    if len(train_df) < 30:
        return {
            "probability": 0.50,
            "bias_string": "50% Neutral Bias (Insufficient Samples)",
            "sample_count": len(train_df)
        }

    X_train = train_df[feature_cols].values
    y_train = train_df['Target'].values.astype(int)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=4,
        random_state=42
    )
    rf.fit(X_train, y_train)

    latest_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
    if np.isnan(latest_features).any():
        col_means = train_df[feature_cols].mean().values.reshape(1, -1)
        latest_features = np.where(np.isnan(latest_features), col_means, latest_features)

    proba = rf.predict_proba(latest_features)[0]
    classes = list(rf.classes_)
    if 1 in classes:
        up_idx = classes.index(1)
        up_prob = proba[up_idx]
    else:
        up_prob = 0.50

    pct = int(round(up_prob * 100))

    if pct >= 55:
        bias_str = f"{pct}% Bullish Bias"
    elif pct <= 45:
        bias_str = f"{100 - pct}% Bearish Bias"
    else:
        bias_str = f"{pct}% Neutral / Balanced"

    return {
        "probability": round(float(up_prob), 4),
        "bias_string": bias_str,
        "sample_count": len(train_df)
    }


@app.get("/")
def root():
    return {
        "service": "Wharton Sim QuantStats Backend",
        "status": "online",
        "endpoint": "/metrics?ticker={TICKER}"
    }


@app.get("/metrics")
def get_metrics(ticker: str = Query(..., description="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)")):
    try:
        clean_ticker = re.sub(r'[^A-Za-z0-9\.\-]', '', ticker.strip()).upper()
        if not clean_ticker:
            raise HTTPException(status_code=400, detail="Invalid ticker provided. Ticker must contain alphanumeric characters.")

        t = yf.Ticker(clean_ticker)
        df = t.history(period="1y", interval="1d", auto_adjust=True)

        if df is None or df.empty or len(df) < 15:
            df = yf.download(clean_ticker, period="1y", interval="1d", progress=False, auto_adjust=True)

        if df is None or df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"Could not retrieve historical data for ticker '{clean_ticker}'. Asset may be delisted or invalid."
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = compute_indicators(df)

        if len(df) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient historical data bars ({len(df)}) for ticker '{clean_ticker}'."
            )

        latest = df.iloc[-1]
        close_price = float(latest['Close'])

        # 1. Trend: "Long" if Close > 200 EMA, else "Short"
        ema200 = latest.get('EMA_200')
        if pd.isna(ema200):
            ema200 = latest.get('EMA_50', close_price)
        trend = "Long" if close_price > float(ema200) else "Short"

        # 2. Strength: ADX value normalized to a 0-100% scale (assuming ADX of 60 is 100% intensity)
        raw_adx = float(latest.get('ADX_14', 25.0))
        if pd.isna(raw_adx):
            raw_adx = 25.0
        normalized_adx = min(100.0, max(0.0, (raw_adx / 60.0) * 100.0))
        strength_str = f"{round(normalized_adx)}% (ADX: {raw_adx:.1f})"

        # 3. Momentum: "Bullish" if RSI > 50 and MACD Histogram > 0, else "Bearish"
        rsi = float(latest.get('RSI_14', 50.0))
        macd_hist = float(latest.get('MACD_HIST', 0.0))
        momentum = "Bullish" if (rsi > 50.0 and macd_hist > 0) else "Bearish"

        # 4. Volatility: "High", "Medium", or "Low" based on trailing 50-period percentile rank of ATR
        atr_series = df['ATR_14'].dropna()
        if len(atr_series) >= 5:
            trailing_atr = atr_series.tail(min(50, len(atr_series)))
            latest_atr = float(trailing_atr.iloc[-1])
            percentile = float((trailing_atr <= latest_atr).mean() * 100.0)
            if percentile > 66.6:
                volatility = f"High ({round(percentile)}th %ile)"
            elif percentile >= 33.3:
                volatility = f"Medium ({round(percentile)}th %ile)"
            else:
                volatility = f"Low ({round(percentile)}th %ile)"
        else:
            volatility = "Medium"

        # 5. Volume Status: Latest volume compared to its 20 SMA
        latest_vol = float(latest.get('Volume', 0.0))
        vol_sma20 = float(latest.get('VOL_SMA_20', latest_vol))
        if vol_sma20 > 0 and (latest_vol > 1.2 * vol_sma20):
            ratio = latest_vol / vol_sma20
            volume_status = f"Above Avg ({ratio:.2f}x 20-SMA)"
        else:
            volume_status = "Normal / Below Avg"

        # 6. Close vs 20-EMA
        ema20 = float(latest.get('EMA_20', close_price))
        close_vs_ema20 = f"Close: ${close_price:.2f} | 20-EMA: ${ema20:.2f}"

        # 7. Structure
        last_10_closes = df['Close'].tail(10)
        max_c = float(last_10_closes.max())
        min_c = float(last_10_closes.min())
        variance_pct = ((max_c - min_c) / min_c) * 100.0 if min_c > 0 else 0.0

        if variance_pct < 2.0:
            structure = f"Consolidating (Tight, {variance_pct:.2f}% range)"
        else:
            structure = f"Trending / Ranging ({variance_pct:.2f}% range)"

        # 8. Machine Learning Module
        ml_result = train_ml_classifier(df)
        ml_forecast_bias = ml_result["bias_string"]

        return {
            "status": "success",
            "ticker": clean_ticker,
            "close_price": round(close_price, 2),
            "close_price_formatted": f"$${close_price:.2f}",
            "trend": trend,
            "strength": strength_str,
            "momentum": momentum,
            "volatility": volatility,
            "volume_status": volume_status,
            "close_vs_ema20": close_vs_ema20,
            "structure": structure,
            "ml_forecast_bias": ml_forecast_bias,
            "details": {
                "rsi_14": round(rsi, 2),
                "macd_histogram": round(macd_hist, 4),
                "adx_14": round(raw_adx, 2),
                "ema_20": round(ema20, 2),
                "ema_50": round(float(latest.get('EMA_50', 0)), 2),
                "ema_200": round(float(ema200), 2),
                "10_bar_range_pct": round(variance_pct, 2),
                "ml_probability_up": ml_result["probability"],
                "bars_analyzed": len(df)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error analyzing ticker '{ticker}': {str(e)}"
        )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
