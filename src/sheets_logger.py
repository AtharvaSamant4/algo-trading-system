# src/sheets_logger.py - FINAL CORRECTED VERSION
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
import os
import numpy as np
import logging
import time
import json
# Set up logging
logger = logging.getLogger(__name__)

# Google Sheets API scope
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

class SheetsLogger:
    # In src/sheets_logger.py - Replace the __init__ method

    def __init__(self, credentials_path=None):
        try:
            # Check for OAuth2 credentials first
            if os.path.exists('google_credentials.json'):
                logger.info("Using google_credentials.json for OAuth2 authentication")
                
                # Load the credentials file to check its type
                with open('google_credentials.json', 'r') as f:
                    creds_data = json.load(f)
                
                # Check if it's OAuth2 "installed" credentials
                if 'installed' in creds_data:
                    logger.info("Detected OAuth2 installed app credentials")
                    
                    # Use gspread's OAuth2 authentication
                    import gspread
                    self.client = gspread.oauth(
                        credentials_filename='google_credentials.json',
                        authorized_user_filename='authorized_user.json'  # This will be created after first auth
                    )
                    
                elif 'type' in creds_data and creds_data['type'] == 'service_account':
                    logger.info("Detected service account credentials")
                    
                    # Use service account authentication
                    creds = Credentials.from_service_account_file(
                        'google_credentials.json', scopes=SCOPES
                    )
                    self.client = gspread.authorize(creds)
                    
                else:
                    raise ValueError("Unknown credential format")
                    
            elif os.path.exists('service_account.json'):
                logger.info("Using service_account.json for authentication")
                creds = Credentials.from_service_account_file(
                    'service_account.json', scopes=SCOPES
                )
                self.client = gspread.authorize(creds)
                
            else:
                logger.warning("No credentials found. Google Sheets disabled.")
                self.client = None
                
        except Exception as e:
            logger.error(f"Google Sheets authentication failed: {str(e)}")
            self.client = None
            
        self.spreadsheet = None

    
    def safe_float_convert(self, value, default=0.0):
        """Safely convert value to float with fallback"""
        try:
            if pd.isna(value) or value in [None, '', 'nan', 'NaN', 'NAN']:
                return default
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert '{value}' to float, using {default}")
            return default
    
    def safe_int_convert(self, value, default=0):
        """Safely convert value to integer with fallback"""
        try:
            if pd.isna(value) or value in [None, '', 'nan', 'NaN', 'NAN']:
                return default
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert '{value}' to int, using {default}")
            return default
    
    def safe_string_convert(self, value, default=''):
        """Safely convert value to string with fallback"""
        try:
            if value is None:
                return default
            return str(value)
        except:
            return default

    def create_trading_dashboard(self, spreadsheet_name="Algo Trading Dashboard"):
        """Create comprehensive trading dashboard"""
        if self.client is None:
            logger.warning("Google Sheets disabled - skipping dashboard creation")
            return "Google Sheets disabled"
            
        try:
            # Try to open existing spreadsheet
            try:
                self.spreadsheet = self.client.open(spreadsheet_name)
                logger.info(f"Using existing spreadsheet: {spreadsheet_name}")
            except gspread.SpreadsheetNotFound:
                # Create new spreadsheet
                self.spreadsheet = self.client.create(spreadsheet_name)
                logger.info(f"Created new spreadsheet: {spreadsheet_name}")
            
            # Setup sheets
            self._setup_trade_log_sheet()
            self._setup_portfolio_summary_sheet()
            self._setup_performance_metrics_sheet()
            
            return self.spreadsheet.url
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def _setup_trade_log_sheet(self):
        """Setup trade log sheet"""
        try:
            try:
                sheet = self.spreadsheet.worksheet("Trade Log")
                sheet.clear()
            except gspread.WorksheetNotFound:
                sheet = self.spreadsheet.add_worksheet(title="Trade Log", rows=1000, cols=12)
            
            # Headers
            headers = [
                'Date', 'Symbol', 'Action', 'Price', 'Quantity', 
                'P&L', 'RSI', 'MA_Signal', 'ML_Prediction', 'ML_Confidence',
                'Strategy', 'Notes'
            ]
            sheet.update('A1', [headers])
            
            # Format headers
            sheet.format('A1:L1', {
                'backgroundColor': {'red': 0.2, 'green': 0.6, 'blue': 1.0},
                'textFormat': {'bold': True, 'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
            })
            logger.info("Trade Log sheet setup complete")
            
        except Exception as e:
            logger.error(f"Trade Log setup failed: {str(e)}")
        
    def _setup_portfolio_summary_sheet(self):
        """Setup portfolio summary sheet"""
        try:
            try:
                sheet = self.spreadsheet.worksheet("Portfolio Summary")
                sheet.clear()
            except gspread.WorksheetNotFound:
                sheet = self.spreadsheet.add_worksheet(title="Portfolio Summary", rows=100, cols=8)
            
            # Create summary structure
            summary_data = [
                ['PORTFOLIO OVERVIEW', '', '', '', '', '', '', ''],
                ['Metric', 'Value', 'Target', 'Status', '', 'Updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ''],
                ['', '', '', '', '', '', '', ''],
                ['Total Portfolio Value', '₹100,000', '₹120,000', '=IF(B4>C4,"✅","⚠️")', '', '', '', ''],
                ['Total P&L', '₹0', '₹10,000', '=IF(B5>0,"✅","❌")', '', '', '', ''],
                ['Win Rate', '0%', '60%', '=IF(B6>C6,"✅","⚠️")', '', '', '', ''],
                ['Total Trades', '0', '50', '=IF(B7>0,"✅","⚠️")', '', '', '', ''],
                ['Active Positions', '0', '3', '=IF(B8<=C8,"✅","⚠️")', '', '', '', ''],
                ['', '', '', '', '', '', '', ''],
                ['RISK METRICS', '', '', '', '', '', '', ''],
                ['Max Drawdown', '0%', '10%', '=IF(B11<C11,"✅","❌")', '', '', '', ''],
                ['Sharpe Ratio', '0', '1.5', '=IF(B12>C12,"✅","⚠️")', '', '', '', ''],
                ['Volatility', '0%', '15%', '=IF(B13<C13,"✅","❌")', '', '', '', ''],
            ]
            
            sheet.update('A1:H13', summary_data)

            formula_updates = [
            {'range': 'D4', 'values': [['=IF(B4>C4,"✅","⚠️")']]},
            {'range': 'D5', 'values': [['=IF(B5>0,"✅","❌")']]},
            {'range': 'D6', 'values': [['=IF(B6>C6,"✅","⚠️")']]},
            {'range': 'D7', 'values': [['=IF(B7>0,"✅","⚠️")']]},
            {'range': 'D8', 'values': [['=IF(B8<=C8,"✅","⚠️")']]},
            {'range': 'D11', 'values': [['=IF(B11<C11,"✅","❌")']]},
            {'range': 'D12', 'values': [['=IF(B12>C12,"✅","⚠️")']]},
            {'range': 'D13', 'values': [['=IF(B13<C13,"✅","❌")']]}
        ]
        
        # Insert formulas with proper value input option
            for formula_update in formula_updates:
                sheet.update(
                    formula_update['range'], 
                    formula_update['values'],
                    value_input_option='USER_ENTERED'  # This makes formulas execute
                )
            
            # Format
            sheet.format('A1:H1', {'backgroundColor': {'red': 0.8, 'green': 0.2, 'blue': 0.2}, 'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}})
            sheet.format('A10:H10', {'backgroundColor': {'red': 0.2, 'green': 0.8, 'blue': 0.2}, 'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}})
            logger.info("Portfolio Summary sheet setup complete")
            
        except Exception as e:
            logger.error(f"Portfolio Summary setup failed: {str(e)}")
        
    def _setup_performance_metrics_sheet(self):
        """Setup detailed performance metrics"""
        try:
            try:
                sheet = self.spreadsheet.worksheet("Performance Metrics")
                sheet.clear()
            except gspread.WorksheetNotFound:
                sheet = self.spreadsheet.add_worksheet(title="Performance Metrics", rows=500, cols=10)
            
            headers = [
                'Date', 'Symbol', 'Daily_Return', 'Cumulative_Return', 
                'Portfolio_Value', 'Drawdown', 'Volume_Traded', 'Trades_Count',
                'ML_Accuracy', 'Strategy_Performance'
            ]
            sheet.update('A1:J1', [headers])
            logger.info("Performance Metrics sheet setup complete")
            
        except Exception as e:
            logger.error(f"Performance Metrics setup failed: {str(e)}")

    def log_trades_batch(self, trades_list):
        """Log multiple trades with proper data validation"""
        if self.client is None:
            logger.warning(f"Trades logged locally: {len(trades_list)} trades")
            return
            
        if not trades_list:
            logger.warning("No trades to log")
            return
            
        try:
            sheet = self.spreadsheet.worksheet("Trade Log")
            
            # Prepare all trade data for batch operation
            batch_data = []
            
            for trade_data in trades_list:
                try:
                    # Handle date conversion
                    date_value = trade_data.get('Date')
                    if hasattr(date_value, 'strftime'):
                        date_str = date_value.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        date_str = str(date_value)
                    
                    # Prepare row data
                    row_data = [
                        self.safe_string_convert(date_str),
                        self.safe_string_convert(trade_data.get('Symbol', '')),
                        self.safe_string_convert(trade_data.get('Action', '')),
                        self.safe_float_convert(trade_data.get('Price', 0)),
                        self.safe_int_convert(trade_data.get('Quantity', 0)),
                        self.safe_float_convert(trade_data.get('P&L', 0)),
                        self.safe_float_convert(trade_data.get('RSI', 0)),
                        self.safe_string_convert(trade_data.get('MA_Signal', '')),
                        self.safe_string_convert(trade_data.get('ML_Prediction', '')),
                        self.safe_float_convert(trade_data.get('ML_Confidence', 0)),
                        self.safe_string_convert(trade_data.get('Strategy', 'RSI_MA_ML')),
                        self.safe_string_convert(trade_data.get('Notes', ''))
                    ]
                    
                    batch_data.append(row_data)
                    
                except Exception as e:
                    logger.error(f"Error processing trade: {str(e)}")
                    continue
            
            if batch_data:
                # Append in chunks to avoid API limits
                CHUNK_SIZE = 50
                for i in range(0, len(batch_data), CHUNK_SIZE):
                    chunk = batch_data[i:i+CHUNK_SIZE]
                    sheet.append_rows(chunk)
                    time.sleep(1.2)  # Avoid rate limiting
                
                logger.info(f"Successfully logged {len(batch_data)} trades to Google Sheets")
                
                # Update portfolio summary
                self.update_portfolio_summary()
            else:
                logger.warning("No valid trades to log after data validation")
            
        except Exception as e:
            logger.error(f"Batch logging failed: {str(e)}")

    def update_portfolio_summary(self):
        """Update portfolio summary metrics"""
        if self.client is None:
            return
            
        try:
            trade_sheet = self.spreadsheet.worksheet("Trade Log")
            summary_sheet = self.spreadsheet.worksheet("Portfolio Summary")
            
            # Get all trade data
            records = trade_sheet.get_all_records()
            if not records:
                logger.warning("No trade data found in Google Sheets")
                return
                
            df = pd.DataFrame(records)
            
            # Filter sell trades
            sell_trades = df[df['Action'].str.upper() == 'SELL'].copy()
            
            # Calculate metrics
            total_pnl = sell_trades['P&L'].sum() if not sell_trades.empty else 0
            portfolio_value = 100000 + total_pnl
            
            if not sell_trades.empty:
                win_rate = (len(sell_trades[sell_trades['P&L'] > 0]) / len(sell_trades)) * 100
            else:
                win_rate = 0
                
            total_trades = len(df)
            
            # Calculate risk metrics
            max_drawdown = self._calculate_drawdown_for_sheets(sell_trades)
            volatility, sharpe_ratio = self._calculate_risk_metrics(sell_trades)
            
            # Update summary
            summary_sheet.batch_update([
                {
                    'range': 'B4:B8',
                    'values': [
                        [f'₹{portfolio_value:,.2f}'],
                        [f'₹{total_pnl:,.2f}'],
                        [f'{win_rate:.1f}%'],
                        [total_trades],
                        [0]  # Active positions
                    ]
                },
                {
                    'range': 'B11:B13',
                    'values': [
                        [f'{abs(max_drawdown):.1f}%'],
                        [f'{sharpe_ratio:.2f}'],
                        [f'{volatility:.1f}%']
                    ]
                },
                {
                    'range': 'G2',
                    'values': [[datetime.now().strftime('%Y-%m-%d %H:%M:%S')]]
                }
            ])
            
            logger.info("Portfolio summary updated")
            logger.info(f"Total P&L: ₹{total_pnl:,.2f}, Win Rate: {win_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Portfolio summary update failed: {str(e)}")

    def _calculate_drawdown_for_sheets(self, sell_trades):
        """Calculate maximum drawdown"""
        if len(sell_trades) < 2:
            return 0.0
            
        try:
            pnl_series = pd.Series([trade['P&L'] for trade in sell_trades])
            cumulative = pnl_series.cumsum() + 100000
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak * 100
            return drawdown.min()
        except:
            return 0.0

    def _calculate_risk_metrics(self, sell_trades):
        """Calculate volatility and Sharpe ratio"""
        if len(sell_trades) < 2:
            return 0.0, 0.0
            
        try:
            returns = [trade['P&L'] / (trade['Price'] * trade['Quantity']) for _, trade in sell_trades.iterrows()]
            volatility = np.std(returns) * 100
            mean_return = np.mean(returns)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            return volatility, sharpe_ratio
        except:
            return 0.0, 0.0

    def log_ml_performance(self, ml_results, portfolio_performance=None, trades_df=None):
        """Log ML performance data"""
        if self.client is None:
            logger.warning("ML performance logged locally")
            return
            
        try:
            sheet = self.spreadsheet.worksheet("Performance Metrics")
            
            # Prepare data
            ml_accuracy = ml_results.get('accuracy', 0) * 100
            best_model = ml_results.get('best_model', 'N/A')
            portfolio_value = portfolio_performance.get('Final_Portfolio_Value', 100000) if portfolio_performance else 100000
            
            # Calculate volume
            volume = 0
            if trades_df is not None:
                volume = (trades_df['Price'] * trades_df['Quantity']).sum()
            
            row = [
                datetime.now().strftime('%Y-%m-%d'),
                'ALL',
                portfolio_performance.get('ROI', 0) if portfolio_performance else 0,
                portfolio_performance.get('CAGR', 0) if portfolio_performance else 0,
                portfolio_value,
                abs(portfolio_performance.get('Max_Drawdown', 0)) if portfolio_performance else 0,
                volume,
                portfolio_performance.get('Total_Trades', 0) if portfolio_performance else 0,
                ml_accuracy,
                best_model
            ]
            
            sheet.append_row(row)
            logger.info("ML performance metrics logged")
            
        except Exception as e:
            logger.error(f"ML performance logging failed: {str(e)}")

    def create_csv_friendly_summary(self):
        """Create CSV-friendly summary"""
        if self.client is None:
            return
            
        try:
            # Create or get sheet
            try:
                csv_sheet = self.spreadsheet.worksheet("Portfolio_Summary_CSV")
            except gspread.WorksheetNotFound:
                csv_sheet = self.spreadsheet.add_worksheet(title="Portfolio_Summary_CSV", rows=50, cols=10)
            
            # Get trade data
            trade_sheet = self.spreadsheet.worksheet("Trade Log")
            records = trade_sheet.get_all_records()
            df = pd.DataFrame(records)
            
            # Calculate metrics
            sell_trades = df[df['Action'].str.upper() == 'SELL']
            total_pnl = sell_trades['P&L'].sum() if not sell_trades.empty else 0
            portfolio_value = 100000 + total_pnl
            win_rate = (len(sell_trades[sell_trades['P&L'] > 0]) / len(sell_trades)) * 100 if not sell_trades.empty else 0
            
            # Prepare data
            data = [
                ['Metric', 'Value', 'Unit'],
                ['Portfolio Value', f'{portfolio_value:.2f}', 'INR'],
                ['Total P&L', f'{total_pnl:.2f}', 'INR'],
                ['ROI', f'{(total_pnl/100000)*100:.2f}', '%'],
                ['Win Rate', f'{win_rate:.1f}', '%'],
                ['Total Trades', len(df), 'count'],
                ['Last Updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '']
            ]
            
            # Update sheet
            csv_sheet.clear()
            csv_sheet.update('A1', data)
            logger.info("CSV-friendly summary created")
            
        except Exception as e:
            logger.error(f"CSV summary creation failed: {str(e)}")
        
    def get_spreadsheet_url(self):
        """Get spreadsheet URL"""
        return self.spreadsheet.url if self.spreadsheet else None