# src/main.py - UPDATED VERSION BASED ON YOUR WORKING ORIGINAL
import os
import sys
import asyncio
import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Import our modules
from data_manager import DataManager
from strategy import RSIMAStrategy
from ml_engine import MLTradingEngine
from sheets_logger import SheetsLogger
from telegram_bot import TradingTelegramBot

# Load environment variables
load_dotenv()

class AlgoTradingSystem:
    def __init__(self):
        # Initialize components (your original clean approach)
        self.data_manager = DataManager()
        self.strategy = RSIMAStrategy()
        self.ml_engine = MLTradingEngine()
        self.sheets_logger = SheetsLogger()  # Your original working initialization
        
        # Telegram bot setup (your original approach)
        telegram_token = os.getenv('TELEGRAM_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if telegram_token:
            self.telegram_bot = TradingTelegramBot(telegram_token, telegram_chat_id)
        else:
            self.telegram_bot = None
            
        # Updated to use BSE symbols (current working symbols)
        self.symbols = ['RELIANCE.BSE', 'HDFCBANK.BSE', 'INFY.BSE']
        
        # Performance tracking
        self.performance_log = []
        
    def run_full_analysis(self):
        """Run complete trading analysis"""
        print("🚀 Starting Algo Trading Analysis...")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Fetch Data (updated method name)
            print("\n📊 Step 1: Fetching market data...")
            data_dict = self.data_manager.get_nifty_data_smart(max_symbols=len(self.symbols), min_rows=80)
            
            if not data_dict:
                raise Exception("No data fetched")
                
            print(f"✅ Successfully fetched data for {len(data_dict)} symbols")
            
            # Step 2: Train ML Model
            print("\n🤖 Step 2: Training ML models...")
            ml_results = self.ml_engine.train_models(data_dict)
            print(f"✅ Best Model: {ml_results['best_model']} with accuracy: {ml_results['accuracy']:.4f}")
            
            # Step 3: Run Strategy Backtest
            print("\n📈 Step 3: Running strategy backtest...")
            trades_df, performance = self.strategy.backtest_strategy(data_dict)
            
            if not trades_df.empty:
                print(f"✅ Generated {len(trades_df)} trades")
                print(f"📊 Win Rate: {performance['Win_Rate']:.1f}%")
                print(f"💰 Total P&L: ₹{performance['Total_PnL']:.2f}")
                print(f"📈 ROI: {performance['ROI']:.2f}%")
            
            # Step 4: Setup Google Sheets dashboard...
            print("\n📊 Step 4: Setting up Google Sheets dashboard...")
            try:
                sheets_url = self.sheets_logger.create_trading_dashboard()
                print(f"✅ Dashboard created: {sheets_url}")
            except Exception as e:
                print(f"⚠️ Dashboard creation failed: {e}")
                sheets_url = "Google Sheets unavailable"

            # Step 5: Log trades to Google Sheets...
            print("\n📝 Step 5: Logging trades to Google Sheets...")
            try:
                if not trades_df.empty:
                    # Convert DataFrame to list format that your log_trades_batch expects
                    trades_list = trades_df.to_dict('records')
                    self.sheets_logger.log_trades_batch(trades_list)
                    print(f"✅ Logged {len(trades_list)} trades to Google Sheets")
                else:
                    print("⚠️ No trades to log")
                
                # Log ML performance using your actual method signature
                self.sheets_logger.log_ml_performance(
                    ml_results=ml_results,
                    portfolio_performance=performance,
                    trades_df=trades_df
                )
                print("✅ ML performance logged to Google Sheets")
                
                # Create CSV summary
                self.sheets_logger.create_csv_friendly_summary()
                print("✅ CSV-friendly summary created")
                
            except Exception as e:
                print(f"⚠️ Logging failed: {e}")

            
            # Step 6: Generate live predictions
            print("\n🔮 Step 6: Generating live predictions...")
            live_signals = self.generate_live_signals(data_dict)
            
            # Step 7: Send notifications (your original approach)
            print("\n📱 Step 7: Sending notifications...")
            if self.telegram_bot:
                asyncio.run(self._send_notifications(performance, ml_results, live_signals))
            
            print("\n🎉 Analysis completed successfully!")
            
            # Step 8: Print summary
            self.print_summary(performance, ml_results, sheets_url)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            if self.telegram_bot:
                asyncio.run(self.telegram_bot.send_error_alert(str(e)))
            return False
    
    def generate_live_signals(self, data_dict):
        """Generate live trading signals"""
        live_signals = []
        
        for symbol, df in data_dict.items():
            try:
                # Get latest signals from strategy
                signals = self.strategy.generate_signals(df, symbol)
                latest_signal = signals.iloc[-1]
                
                # Get ML prediction
                ml_prediction, ml_confidence = self.ml_engine.predict_next_day(df)
                
                if latest_signal['Signal'] != 0:  # If there's a signal
                    signal_data = {
                        'Symbol': symbol,
                        'Action': 'BUY' if latest_signal['Signal'] == 1 else 'SELL',
                        'Price': latest_signal['Price'],
                        'RSI': latest_signal['RSI'],
                        'MA_Signal': 'Bullish' if latest_signal['MA_20'] > latest_signal['MA_50'] else 'Bearish',
                        'ML_Prediction': 'UP' if ml_prediction == 1 else 'DOWN',
                        'ML_Confidence': ml_confidence,
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    live_signals.append(signal_data)
                    
                    # Send individual trade alert
                    if self.telegram_bot:
                        asyncio.run(self.telegram_bot.send_trade_alert(signal_data))
                        
            except Exception as e:
                print(f"⚠️ Error generating signal for {symbol}: {e}")
                
        return live_signals
    
    async def _send_notifications(self, performance, ml_results, live_signals):
        """Send comprehensive notifications (your original approach)"""
        if not self.telegram_bot:
            return
            
        # Portfolio update
        portfolio_data = {
            'total_value': 100000 + performance.get('Total_PnL', 0),
            'daily_pnl': performance.get('Total_PnL', 0),
            'daily_return': performance.get('ROI', 0),
            'total_return': performance.get('ROI', 0),
            'win_rate': performance.get('Win_Rate', 0),
            'sharpe_ratio': performance.get('Sharpe_Ratio', 0),
            'max_drawdown': performance.get('Max_Drawdown', 0),
            'total_trades': performance.get('Total_Trades', 0),
            'ml_accuracy': ml_results.get('accuracy', 0)
        }
        
        await self.telegram_bot.send_portfolio_update(portfolio_data)
    
    def print_summary(self, performance, ml_results, sheets_url):
        """Print comprehensive summary (your original design)"""
        print("\n" + "="*60)
        print("📊 ALGO TRADING SYSTEM - EXECUTION SUMMARY")
        print("="*60)
        
        print(f"🎯 Strategy Performance:")
        print(f"   • Total Trades: {performance.get('Total_Trades', 0)}")
        print(f"   • Win Rate: {performance.get('Win_Rate', 0):.1f}%")
        print(f"   • Total P&L: ₹{performance.get('Total_PnL', 0):,.2f}")
        print(f"   • ROI: {performance.get('ROI', 0):.2f}%")
        print(f"   • Sharpe Ratio: {performance.get('Sharpe_Ratio', 0):.2f}")
        print(f"   • Max Drawdown: {performance.get('Max_Drawdown', 0):.1f}%")
        
        print(f"\n🤖 ML Performance:")
        print(f"   • Best Model: {ml_results.get('best_model', 'N/A')}")
        print(f"   • Accuracy: {ml_results.get('accuracy', 0):.1f}%")
        
        print(f"\n📊 Integration Status:")
        print(f"   • Google Sheets: ✅ Active")
        print(f"   • Dashboard URL: {sheets_url}")
        print(f"   • Telegram Bot: {'✅ Active' if self.telegram_bot else '❌ Not configured'}")
        
        print(f"\n⏰ Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    def schedule_trading(self):
        """Schedule regular trading operations (your original design)"""
        # Schedule daily analysis
        schedule.every().day.at("09:30").do(self.run_full_analysis)  # Market opening
        schedule.every().day.at("15:30").do(self.run_full_analysis)  # Market closing
        
        print("⏰ Scheduled trading operations:")
        print("   • 09:30 AM - Morning analysis")
        print("   • 03:30 PM - End-of-day analysis")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main execution function (your original design)"""
    print("🚀 ALGO TRADING SYSTEM - ML & AUTOMATION")
    print("=" * 50)
    
    # Initialize system
    system = AlgoTradingSystem()
    
    # Run immediate analysis
    success = system.run_full_analysis()
    
    if success:
        print("\n🎉 System successfully executed!")
        print("📊 Check your Google Sheets dashboard for detailed results")
        if system.telegram_bot:
            print("📱 Telegram notifications sent")
            
        # Ask if user wants to schedule regular operations
        schedule_choice = input("\n⏰ Schedule regular trading operations? (y/n): ")
        if schedule_choice.lower() == 'y':
            print("🔄 Starting scheduled operations...")
            system.schedule_trading()
    else:
        print("❌ System execution failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
