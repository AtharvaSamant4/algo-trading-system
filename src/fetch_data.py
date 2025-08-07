"""
fetch_data.py  – unified price/volume fetcher (6-month window)

• Pulls exactly the most-recent N months (default = 6) of daily OHLCV data.
• Tries Alpha Vantage first (if ALPHA_VANTAGE_KEY is set).
• Falls back to Yahoo Finance if AV fails or the symbol isn’t covered.
• Returns a tidy pandas DataFrame with columns:
      ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from dateutil.relativedelta import relativedelta   # pip install python-dateutil
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
load_dotenv()
API_KEY          = os.getenv("ALPHA_VANTAGE_KEY")
DEFAULT_MONTHS   = 6          # <<< hard spec from assignment
CACHE_DIR        = Path("data")   # optional offline cache
CACHE_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Telegram notifier (optional)                                                #
# --------------------------------------------------------------------------- #
def _notify_telegram(msg: str) -> None:
    token   = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=4)
    except Exception as exc:  # noqa: BLE001
        logging.debug("Telegram notify failed: %s", exc)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _start_date_str(months_back: int) -> str:
    """Return YYYY-MM-DD string N months ago from today."""
    dt = datetime.now() - relativedelta(months=months_back)
    return dt.strftime("%Y-%m-%d")

# --------------------------------------------------------------------------- #
# Alpha Vantage fetcher                                                       #
# --------------------------------------------------------------------------- #
def _av_daily(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    if not API_KEY:
        return None
    try:
        ts = TimeSeries(key=API_KEY, output_format="pandas")
        raw, _meta = ts.get_daily_adjusted(symbol=symbol, outputsize="full")
        if raw.empty:
            raise ValueError("Empty payload")
        raw.index = pd.to_datetime(raw.index)
        data = raw[raw.index >= pd.to_datetime(start_date)]
        if data.empty:
            raise ValueError(f"No rows after {start_date}")
        data.rename(columns=lambda c: c.split(". ")[1].upper(), inplace=True)
        return data[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]].astype(float).sort_index()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Alpha Vantage failed for %s → %s", symbol, exc)
        _notify_telegram(f"⚠️ Alpha Vantage failed for {symbol}: {exc}")
        return None

# --------------------------------------------------------------------------- #
# Yahoo Finance fetcher                                                       #
# --------------------------------------------------------------------------- #
def _yf_daily(symbol: str, start_date: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start_date,
        end=datetime.now().strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.rename(columns=lambda c: c.upper().replace(" ", "_"), inplace=True)

    required = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns from yfinance: {missing}")

    return df[required].astype(float).sort_index()

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def get_daily(symbol: str, months: int = DEFAULT_MONTHS) -> pd.DataFrame:
    """
    Fetch daily OHLCV for the most-recent `months` months (default = 6).
    Priority: Alpha Vantage → Yahoo Finance → RuntimeError.
    """
    start_date = _start_date_str(months)

    data = _av_daily(symbol, start_date)
    if data is None:
        data = _yf_daily(symbol, start_date)

    if data.empty:
        raise RuntimeError(f"No data found for {symbol} from any source")

    return data

# --------------------------------------------------------------------------- #
# CLI smoke-test                                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    for sym in symbols:
        print(f"\n>>> {sym}")
        df = get_daily(sym)          # default 6 months
        print(
            f"Rows: {len(df):3d}  |  From {df.index.min().date()} "
            f"to {df.index.max().date()}"
        )
        print(df.tail())

    import fetch_data, indicators

    df = indicators.add_indicators(fetch_data.get_daily("RELIANCE.NS"))

  