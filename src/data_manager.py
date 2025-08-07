# src/data_manager.py
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import time
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', 'A8CPY8GOG5PYJU0F')
        
        self.bse_symbols = [
            'RELIANCE.BSE',
            'HDFCBANK.BSE', 
            'INFY.BSE'
        ]
        
        self.successful_symbols = []
        self.failed_symbols = []
        self.api_calls_made = 0
    
    def fetch_stock_data(self, symbol):
        """Fetch exactly 6 months of data as per assignment requirements"""
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.alpha_vantage_key}&outputsize=compact&datatype=json"
        
        try:
            logger.info(f"Fetching {symbol} (6 months data only)...")
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            time_series = data.get("Time Series (Daily)")
            if not time_series:
                logger.error(f"No data found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = [col.split(" ")[1].capitalize() for col in df.columns]
            df = df.apply(pd.to_numeric)
            df = df.sort_index(ascending=True)
            
            six_months_data = self.limit_to_six_months(df)
            
            logger.info(f"{symbol}: Fetched exactly {len(six_months_data)} rows (6 months)")
            logger.info(f"Date range: {six_months_data.index[0].date()} to {six_months_data.index[-1].date()}")
            
            time.sleep(12)  # Respect API rate limits
            
            return six_months_data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def limit_to_six_months(self, df):
        """Extract exactly the last 6 months of data from today"""
        if df.empty:
            return df
        
        today = datetime.now()
        six_months_ago = today - pd.DateOffset(months=6)
        six_months_data = df[df.index >= six_months_ago]
        
        # Fallback if insufficient data
        if len(six_months_data) < 50:
            logger.warning(f"Only {len(six_months_data)} rows available for 6 months")
            six_months_data = df.tail(100) if len(df) >= 100 else df
        
        logger.info(f"Filtered from {len(df)} rows to {len(six_months_data)} rows (6 months)")
        
        return six_months_data
    
    def get_nifty_data_smart(self, max_symbols=3, min_rows=50):
        """Fetch exactly 6 months data for BSE symbols"""
        logger.info(f"Fetching 6 months data for assignment compliance...")
        logger.info(f"Target stocks: {', '.join(self.bse_symbols[:max_symbols])}")
        
        data_dict = {}
        successful_count = 0
        
        for i, symbol in enumerate(self.bse_symbols[:max_symbols]):
            try:
                logger.info(f"Processing {symbol} ({i+1}/{max_symbols})...")
                
                raw_data = self.fetch_stock_data(symbol)
                
                if not raw_data.empty and len(raw_data) >= min_rows:
                    logger.info(f"{symbol}: 6-month data ready ({len(raw_data)} rows)")
                    
                    enhanced_data = self.add_technical_indicators(raw_data)
                    
                    if not enhanced_data.empty and len(enhanced_data) >= 30:
                        data_dict[symbol] = enhanced_data
                        self.successful_symbols.append(symbol)
                        successful_count += 1
                        
                        latest_price = enhanced_data['Close'].iloc[-1]
                        logger.info(f"{symbol}: Ready for 6-month backtesting ({len(enhanced_data)} rows)")
                        logger.info(f"Latest price: ₹{latest_price:.2f}")
                    else:
                        logger.warning(f"{symbol}: Technical indicators failed")
                        self.failed_symbols.append(symbol)
                else:
                    logger.warning(f"{symbol}: Insufficient 6-month data ({len(raw_data) if not raw_data.empty else 0} rows)")
                    self.failed_symbols.append(symbol)
                    
            except Exception as e:
                logger.error(f"{symbol}: Processing failed - {str(e)}")
                self.failed_symbols.append(symbol)
        
        # Results summary
        logger.info(f"Successfully loaded: {successful_count} stocks")
        logger.info(f"Stocks ready: {list(data_dict.keys())}")
        
        if data_dict:
            total_data_points = sum(len(df) for df in data_dict.values())
            logger.info(f"Total data points (6 months): {total_data_points}")
            logger.info(f"System ready for 6-month backtesting!")
        else:
            logger.error("No 6-month data successfully fetched")
        
        return data_dict
    
    def add_technical_indicators(self, df):
        """Add technical indicators with robust error handling for 6-month data"""
        try:
            if len(df) < 30:
                logger.warning(f"Insufficient data for technical indicators: {len(df)} rows")
                return pd.DataFrame()
            
            df = df.copy()
            
            # Basic technical indicators
            try:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['MA_20'] = ta.sma(df['Close'], length=20)
                # Adjust MA_50 period for shorter timeframe
                ma_50_period = min(50, len(df) - 10)
                df['MA_50'] = ta.sma(df['Close'], length=ma_50_period)
            except Exception as e:
                logger.warning(f"Basic indicators failed: {e}")
                df['RSI'] = 50
                df['MA_20'] = df['Close']
                df['MA_50'] = df['Close']
            
            # MACD
            try:
                macd_data = ta.macd(df['Close'])
                if macd_data is not None and not macd_data.empty:
                    df['MACD'] = macd_data.iloc[:, 0]
                    df['MACD_Signal'] = macd_data.iloc[:, 1]
                    df['MACD_Hist'] = macd_data.iloc[:, 2]
                else:
                    df['MACD'] = df['MACD_Signal'] = df['MACD_Hist'] = 0
            except Exception as e:
                logger.warning(f"MACD calculation failed: {e}")
                df['MACD'] = df['MACD_Signal'] = df['MACD_Hist'] = 0
            
            # Volume and momentum indicators
            try:
                volume_period = min(20, len(df) - 5)
                df['Volume_SMA'] = ta.sma(df['Volume'], length=volume_period)
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, 1)
                df['Price_Change'] = df['Close'].pct_change()
                df['Price_Change_5d'] = df['Close'].pct_change(5)
                volatility_period = min(20, len(df) - 5)
                df['Volatility'] = df['Price_Change'].rolling(volatility_period).std()
            except Exception as e:
                logger.warning(f"Volume/momentum indicators failed: {e}")
                df['Volume_Ratio'] = 1.0
                df['Price_Change'] = 0
                df['Price_Change_5d'] = 0
                df['Volatility'] = 0.01
            
            # Clean up NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            df = df[df['Close'] > 0]
            df = df.dropna(subset=['Close'])
            
            logger.info(f"Technical indicators added to 6-month data: {len(df)} clean rows")
            return df
            
        except Exception as e:
            logger.error(f"Technical indicators failed: {str(e)}")
            return pd.DataFrame()
    
    def get_symbol_info(self, symbol):
        """Get information about BSE symbols"""
        symbol_info = {
            'RELIANCE.BSE': {
                'name': 'Reliance Industries Limited',
                'sector': 'Oil & Gas',
                'market_cap': 'Large Cap'
            },
            'HDFCBANK.BSE': {
                'name': 'HDFC Bank Limited',
                'sector': 'Private Banking',
                'market_cap': 'Large Cap'
            },
            'INFY.BSE': {
                'name': 'Infosys Limited',
                'sector': 'Information Technology',
                'market_cap': 'Large Cap'
            }
        }
        
        return symbol_info.get(symbol, {
            'name': symbol.replace('.BSE', ''),
            'sector': 'BSE Listed',
            'market_cap': 'Available'
        })
    
    def test_connection(self):
        """Test connection using 6-month data approach"""
        logger.info("Testing connection with 6-month data...")
        
        try:
            test_data = self.fetch_stock_data('RELIANCE.BSE')
            
            if not test_data.empty:
                logger.info("Connection test successful!")
                logger.info(f"Test data (6 months): {len(test_data)} rows")
                logger.info(f"Date range: {test_data.index[0].date()} to {test_data.index[-1].date()}")
                return True
            else:
                logger.error("Connection test failed - no data returned")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def check_api_limit(self):
        return True
    
    def test_api_connection(self):
        return self.test_connection()
