# src/working_data_fetcher.py - FIXED VERSION WITH CORRECT IMPORTS
from __future__ import annotations

import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def fetch_nse_data_direct():
    """Execute your working fetch logic directly"""
    
    # Your exact config
    load_dotenv()
    API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
    DEFAULT_MONTHS = 6
    
    def _start_date_str(months_back: int) -> str:
        dt = datetime.now() - relativedelta(months=months_back)
        return dt.strftime("%Y-%m-%d")
    
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
        except Exception as exc:
            logging.warning("Alpha Vantage failed for %s → %s", symbol, exc)
            return None
    
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
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns from yfinance: {missing}")
        
        return df[required].astype(float).sort_index()
    
    def get_daily(symbol: str, months: int = DEFAULT_MONTHS) -> pd.DataFrame:
        start_date = _start_date_str(months)
        data = _av_daily(symbol, start_date)
        if data is None:
            data = _yf_daily(symbol, start_date)
        if data.empty:
            raise RuntimeError(f"No data found for {symbol} from any source")
        return data
    
    # Execute your exact test
    symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    data_dict = {}
    
    for sym in symbols:
        try:
            print(f"\n>>> {sym}")
            df = get_daily(sym)  # Your exact working call
            
            if not df.empty:
                print(f"Rows: {len(df):3d}  |  From {df.index.min().date()} to {df.index.max().date()}")
                
                # Convert to expected format
                converted = df.rename(columns={
                    'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 
                    'CLOSE': 'Close', 'VOLUME': 'Volume'
                })
                
                data_dict[sym] = converted
                print(f"✅ {sym}: Ready for analysis ({len(converted)} rows)")
                
        except Exception as e:
            print(f"❌ {sym}: Failed - {e}")
    
    return data_dict

if __name__ == "__main__":
    result = fetch_nse_data_direct()
    print(f"\n🎯 Final Results: {len(result)} stocks ready")
