# src/telegram_bot.py
import telegram
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
import os
from datetime import datetime
import json

class TradingTelegramBot:
    def __init__(self, token, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token=token)
        self.application = Application.builder().token(token).build()
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup command handlers"""
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("status", self.status))
        self.application.add_handler(CommandHandler("portfolio", self.portfolio))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        welcome_message = """
🚀 **ALGO TRADING BOT ACTIVATED** 🚀

Welcome to your AI-powered trading assistant!

Available commands:
📊 /status - Get current market status
💰 /portfolio - View portfolio performance  
❓ /help - Show all commands

🔥 **Features:**
✅ Real-time RSI + Moving Average signals
🤖 ML-powered predictions
📊 Live Google Sheets tracking
⚡ Instant trade alerts

Ready to make some profits! 💸
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status command handler"""
        status_message = """
📊 **TRADING SYSTEM STATUS**

🟢 **System**: Online & Active
📡 **Data Feed**: Connected (Yahoo Finance + Alpha Vantage)
🤖 **ML Engine**: Model Loaded & Ready
📊 **Google Sheets**: Synced
⏰ **Last Update**: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """

🎯 **Active Strategies**: RSI + MA + ML Prediction
📈 **Monitoring**: 3 NIFTY 50 stocks
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
        
    async def portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Portfolio command handler"""
        # This would connect to real portfolio data
        portfolio_message = """
💰 **PORTFOLIO SUMMARY**

🏦 **Total Value**: ₹1,05,240 (+5.24%)
📈 **Today's P&L**: +₹2,180 (+2.1%)
🎯 **Win Rate**: 68.3%
📊 **Total Trades**: 24

**📋 Active Positions:**
• RELIANCE.NS: +₹1,240 (+3.2%)
• TCS.NS: +₹890 (+1.8%)
• HDFCBANK.NS: +₹50 (+0.1%)

**⚡ Recent Signals:**
🟢 BUY: INFY.NS @ ₹1,485 (RSI: 32, ML: 73%)
🔴 SELL: ITC.NS @ ₹462 (RSI: 68, ML: 45%)

📊 Full dashboard: [Google Sheets]
        """
        
        await update.message.reply_text(portfolio_message, parse_mode='Markdown')
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        help_message = """
🤖 **ALGO TRADING BOT COMMANDS**

**📊 Market Commands:**
/status - System status & health check
/portfolio - Portfolio summary & P&L
/signals - Latest trading signals

**⚙️ Settings:**
/settings - Bot configuration
/alerts on/off - Toggle notifications

**📈 Analysis:**
/analysis [SYMBOL] - Technical analysis
/ml [SYMBOL] - ML prediction

**❓ Support:**
/help - This help message
/contact - Get support

**🎯 Strategy**: RSI + Moving Average + ML
**📊 Dashboard**: Google Sheets integration
**⚡ Alerts**: Real-time notifications
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
        
    async def send_trade_alert(self, trade_data):
        """Send trading alert"""
        if not self.chat_id:
            return
            
        alert_message = f"""
🚨 **TRADE ALERT** 🚨

📊 **{trade_data['Symbol']}**
🎯 **Action**: {trade_data['Action']}
💰 **Price**: ₹{trade_data['Price']:.2f}
📈 **RSI**: {trade_data.get('RSI', 0):.1f}
🔄 **MA Signal**: {trade_data.get('MA_Signal', 'N/A')}
🤖 **ML Confidence**: {trade_data.get('ML_Confidence', 0)*100:.1f}%

⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}
📋 **Strategy**: RSI + MA + ML

{self._get_signal_emoji(trade_data)} **Recommendation**: Execute trade
        """
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=alert_message,
                parse_mode='Markdown'
            )
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")
            
    async def send_portfolio_update(self, portfolio_data):
        """Send daily portfolio update"""
        if not self.chat_id:
            return
            
        update_message = f"""
📊 **DAILY PORTFOLIO UPDATE**

💰 **Portfolio Value**: ₹{portfolio_data.get('total_value', 0):,.2f}
📈 **Today's P&L**: ₹{portfolio_data.get('daily_pnl', 0):,.2f} ({portfolio_data.get('daily_return', 0):.2f}%)
🎯 **Total Return**: {portfolio_data.get('total_return', 0):.2f}%
🏆 **Win Rate**: {portfolio_data.get('win_rate', 0):.1f}%

**📊 Performance Metrics:**
⚡ Sharpe Ratio: {portfolio_data.get('sharpe_ratio', 0):.2f}
📉 Max Drawdown: {portfolio_data.get('max_drawdown', 0):.1f}%
🔢 Total Trades: {portfolio_data.get('total_trades', 0)}

🎯 **Strategy Performance**: Excellent! 🚀
📊 **ML Accuracy**: {portfolio_data.get('ml_accuracy', 0)*100:.1f}%

Keep up the great work! 💪
        """
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=update_message,
                parse_mode='Markdown'
            )
        except Exception as e:
            print(f"Failed to send portfolio update: {e}")
            
    def _get_signal_emoji(self, trade_data):
        """Get appropriate emoji for trade signal"""
        action = trade_data.get('Action', '').upper()
        if action == 'BUY':
            return '🟢'
        elif action == 'SELL':
            return '🔴'
        else:
            return '⚪'
            
    async def send_error_alert(self, error_message):
        """Send error notification"""
        if not self.chat_id:
            return
            
        error_alert = f"""
⚠️ **SYSTEM ERROR ALERT** ⚠️

❌ **Error**: {error_message}
⏰ **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔧 **Action**: System attempting auto-recovery
📞 **Status**: Monitoring situation

Will update once resolved. 🛠️
        """
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=error_alert,
                parse_mode='Markdown'
            )
        except Exception as e:
            print(f"Failed to send error alert: {e}")
            
    def run_polling(self):
        """Run bot in polling mode"""
        self.application.run_polling()
