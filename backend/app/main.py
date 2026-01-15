# backend/app/main.py

from typing import List, Optional, Union
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


app = FastAPI(title="Quant API (SMA + ML + Indicators + Controls)")

# Allow your React frontend (port 5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for local dev; restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic models ----------

class QuoteResponse(BaseModel):
    symbol: str
    price: float
    pe: Optional[float] = None
    market_cap: Optional[float] = None


class BacktestRequest(BaseModel):
    symbol: str
    model_type: str = "sma_crossover"  # "sma_crossover", "logistic", "random_forest"
    years: int = 3                     # backtest window in years

    # New user controls
    sma_fast: int = 10                 # fast SMA window
    sma_slow: int = 50                 # slow SMA window
    rsi_window: int = 14               # RSI lookback
    vol_window: int = 20               # realized vol lookback
    mom_lookback: int = 10             # momentum lookback (days)
    prob_threshold: float = 0.55       # ML long/flat cutoff


class BacktestResponse(BaseModel):
    symbol: str
    model_type: str
    years: int
    metrics: dict
    equity_curve: List[List[Union[str, float]]]
    benchmark_curve: List[List[Union[str, float]]]


# ---------- Utility functions ----------

def _annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).prod()
    n = returns.shape[0]
    return float(cumulative ** (periods_per_year / n) - 1)


def _annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def _sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    vol = _annualized_vol(returns, periods_per_year)
    if vol == 0:
        return 0.0
    ar = _annualized_return(returns, periods_per_year)
    return float((ar - rf) / vol)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _sanitize_params(req: BacktestRequest) -> BacktestRequest:
    """
    Clamp user-provided parameters to reasonable ranges
    so they can't accidentally blow up the math.
    """
    years = max(1, min(req.years, 10))

    sma_fast = max(2, min(req.sma_fast, 100))
    sma_slow = max(sma_fast + 1, min(req.sma_slow, 300))

    rsi_window = max(5, min(req.rsi_window, 50))
    vol_window = max(5, min(req.vol_window, 60))
    mom_lookback = max(2, min(req.mom_lookback, 60))

    prob_threshold = max(0.5, min(req.prob_threshold, 0.8))

    return BacktestRequest(
        symbol=req.symbol,
        model_type=req.model_type,
        years=years,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        rsi_window=rsi_window,
        vol_window=vol_window,
        mom_lookback=mom_lookback,
        prob_threshold=prob_threshold,
    )


def _download_price_df(
    symbol: str,
    years: int,
    sma_fast: int,
    sma_slow: int,
    rsi_window: int,
    vol_window: int,
    mom_lookback: int,
) -> pd.DataFrame:
    """
    Download daily price data from Yahoo and build a feature dataframe with:
      - close
      - ret_1
      - sma_fast, sma_slow
      - rsi_<rsi_window>
      - vol_<vol_window> (realized)
      - mom_<mom_lookback>
    """

    end = date.today()
    start = end - timedelta(days=365 * years + 30)

    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading data: {e}")

    if data.empty:
        raise HTTPException(status_code=400, detail="No price data for this symbol / period.")

    # Flatten possible MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join([str(c) for c in col if c != ""]).strip()
                        for col in data.columns.values]

    # Choose price column robustly (Adj Close preferred, else Close)
    price_col = None
    candidates = [
        "Adj Close",
        f"Adj Close_{symbol}",
        "Close",
        f"Close_{symbol}",
    ]

    for cand in candidates:
        if cand in data.columns:
            price_col = cand
            break

    if price_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"No usable price column found. Columns: {list(data.columns)}",
        )

    df = data[[price_col]].rename(columns={price_col: "close"}).copy()

    # --- Core primitive series ---
    df["ret_1"] = df["close"].pct_change()

    # SMAs
    df["sma_fast"] = df["close"].rolling(sma_fast).mean()
    df["sma_slow"] = df["close"].rolling(sma_slow).mean()

    # Momentum
    df["mom"] = df["close"] / df["close"].shift(mom_lookback) - 1.0

    # Realized volatility (annualized)
    df["vol"] = df["ret_1"].rolling(vol_window).std() * np.sqrt(252)

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi

    # Drop early NaNs
    df = df.dropna()
    if len(df) < 200:
        raise HTTPException(status_code=400, detail="Not enough data after feature engineering.")

    return df


def _build_sma_strategy(df: pd.DataFrame):
    """
    SMA(long/flat) strategy using ret_1 and sma_fast / sma_slow.
    Returns (strat_ret, bench_ret) as Series aligned on df index.
    """
    out = df.copy()
    out["position"] = np.where(out["sma_fast"] > out["sma_slow"], 1.0, 0.0)
    out["position_shifted"] = out["position"].shift(1).fillna(0.0)

    strat_ret = out["position_shifted"] * out["ret_1"]
    bench_ret = out["ret_1"]

    return strat_ret, bench_ret


def _build_ml_strategy(df: pd.DataFrame, model_type: str, prob_threshold: float):
    """
    ML-based long/flat strategy:
      - Label: future_ret_1 > 0 ? 1 : 0
      - Features: [ret_1, sma_fast, sma_slow, rsi, vol, mom]
      - Train on first 70%, test on last 30%
      - Position: long if P(up) > prob_threshold, else flat.
      - Returns: future_ret_1 as payoff.
    """
    out = df.copy()
    out["future_ret_1"] = out["close"].shift(-1) / out["close"] - 1.0
    out["y"] = (out["future_ret_1"] > 0).astype(int)

    out = out.dropna()
    if len(out) < 250:
        raise HTTPException(status_code=400, detail="Not enough data for ML strategy.")

    feature_cols = ["ret_1", "sma_fast", "sma_slow", "rsi", "vol", "mom"]
    for col in feature_cols:
        if col not in out.columns:
            raise HTTPException(status_code=500, detail=f"Feature '{col}' missing in dataframe.")

    features = out[feature_cols]
    y = out["y"].values
    fut_ret = out["future_ret_1"].values
    dates = out.index

    n = len(out)
    split = int(n * 0.7)
    if split <= 0 or split >= n:
        raise HTTPException(status_code=500, detail="Invalid train/test split.")

    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    fut_ret_test = fut_ret[split:]
    dates_test = dates[split:]

    # Choose model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        )
    else:  # "logistic"
        model = LogisticRegression(max_iter=2000)

    model.fit(X_train, y_train)
    proba_up = model.predict_proba(X_test)[:, 1]

    # Position: long if P(up) > prob_threshold
    position = (proba_up > prob_threshold).astype(float)
    strat_ret = position * fut_ret_test
    bench_ret = fut_ret_test  # buy & hold in same period

    strat_ret_series = pd.Series(strat_ret, index=dates_test)
    bench_ret_series = pd.Series(bench_ret, index=dates_test)

    return strat_ret_series, bench_ret_series


def _build_equity_and_metrics(strat_ret: pd.Series, bench_ret: pd.Series):
    equity = (1 + strat_ret).cumprod()
    benchmark = (1 + bench_ret).cumprod()

    strat_ret_series = strat_ret.dropna()

    metrics = {
        "cagr": _annualized_return(strat_ret_series),
        "sharpe": _sharpe_ratio(strat_ret_series),
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": float((strat_ret_series > 0).mean()),
    }

    equity_curve = [
        [d.strftime("%Y-%m-%d"), float(v)] for d, v in equity.items()
    ]
    benchmark_curve = [
        [d.strftime("%Y-%m-%d"), float(v)] for d, v in benchmark.items()
    ]

    return metrics, equity_curve, benchmark_curve


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"message": "Backend is running with SMA + ML + indicators + controls"}


@app.get("/api/quote", response_model=QuoteResponse)
def get_quote(symbol: str):
    """
    Simple dummy quote endpoint.
    You can later replace this with a real data source.
    """
    return QuoteResponse(
        symbol=symbol.upper(),
        price=123.45,
        pe=20.1,
        market_cap=1.5e12,
    )


@app.post("/api/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest):
    """
    Dispatch to different strategies based on model_type, with
    user-controllable SMA windows & ML probability threshold.
    """
    clean_req = _sanitize_params(req)

    symbol = clean_req.symbol.upper()
    years = clean_req.years

    df = _download_price_df(
        symbol,
        years,
        clean_req.sma_fast,
        clean_req.sma_slow,
        clean_req.rsi_window,
        clean_req.vol_window,
        clean_req.mom_lookback,
    )

    model_type = clean_req.model_type

    if model_type == "sma_crossover":
        strat_ret, bench_ret = _build_sma_strategy(df)
    elif model_type in ("logistic", "random_forest"):
        strat_ret, bench_ret = _build_ml_strategy(df, model_type, clean_req.prob_threshold)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model_type '{model_type}'. "
                   f"Use 'sma_crossover', 'logistic', or 'random_forest'.",
        )

    metrics, equity_curve, benchmark_curve = _build_equity_and_metrics(strat_ret, bench_ret)

    return BacktestResponse(
        symbol=symbol,
        model_type=model_type,
        years=years,
        metrics=metrics,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
    )
