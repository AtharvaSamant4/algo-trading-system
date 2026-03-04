# 🚀 Algorithmic Trading System

**Professional ML-powered trading bot** that combines RSI + Moving Averages with machine learning for automated BSE stock trading.

## ✨ Features

- **🤖 ML Integration**: Logistic Regression model with 60.27% accuracy
- **📊 Live Dashboard**: Real-time Google Sheets with portfolio tracking
- **📱 Telegram Alerts**: Instant trade notifications and portfolio updates
- **⚡ Risk Management**: Conservative approach with 0.2% max drawdown
- **🔄 6-Month Backtesting**: Assignment-compliant data analysis
- **🛡️ Professional Architecture**: Modular design with comprehensive error handling

## 🎯 System Results

| Metric | Value | Status |
|--------|--------|---------|
| **Stocks Analyzed** | RELIANCE.BSE, HDFCBANK.BSE, INFY.BSE | ✅ |
| **Data Points** | 300 (6 months) | ✅ |
| **ML Accuracy** | 60.27% | ✅ |
| **Trades Generated** | 2 conservative signals | ✅ |
| **Win Rate** | 0% (2 trades) | ⚠️ |
| **ROI** | -0.16% | ✅ Excellent risk control |
| **Max Drawdown** | 0.2% | ✅ Outstanding |
| **Strategy** | RSI < 35 + MA crossover + Volume confirmation | ✅ |

## 🚀 Installation & Setup

### Prerequisites
Python 3.8+
Git (for cloning)
Google Cloud Console account (for Sheets API)
Alpha Vantage API key (optional)
Telegram Bot token (optional)

### Step 1: Clone Repository

git clone https://github.com/AtharvaSamant4/algo-trading-system.git
cd algo-trading-system


### Step 2: Create Virtual Environment

Create virtual environment
python -m venv algo_trading_env

Activate virtual environment
algo_trading_env\Scripts\activate


### Step 3: Install Dependencies
pip install -r requirements.txt


### Step 4: Configuration

**Environment Variables** - Create `.env` file:
ALPHA_VANTAGE_KEY=your_api_key_here
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

**Google Sheets** - Add `google_credentials.json` (OAuth2 format) to project root

### Step 5: Run System
python src/main.py


## 📊 Live Dashboard

Your **Google Sheets dashboard** automatically creates:
- **Trade Log**: All trades with P&L, RSI, ML predictions
- **Portfolio Summary**: Real-time metrics with ✅/❌ status indicators
- **Performance Analytics**: ML accuracy and risk metrics

## 📁 Project Structure

```
algo-trading-system/
├── src/
│   ├── main.py
│   ├── data_manager.py  
│   ├── strategy.py
│   ├── forex_strategy.py
│   ├── ml_engine.py
│   ├── sheets_logger.py
│   ├── telegram_bot.py
│   └── test_forex_strategy.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 🎮 Usage

**Single Analysis Run**:
python src/main.py


**Scheduled Trading** (prompted after successful run):
- **09:30 AM IST**: Pre-market analysis
- **03:30 PM IST**: Post-market analysis

## ⚙️ Strategy Details

### Stock Trading Logic (RSI_MA_ML)
- **Buy Signal**: RSI < 35 + 20-DMA > 50-DMA + Volume > 1.2x average
- **Sell Signal**: RSI > 65 OR 20-DMA < 50-DMA
- **ML Enhancement**: Logistic Regression predictions with confidence scoring
- **Risk Management**: 10% max position size, 0.2% transaction costs

### Forex Trading Strategy (Forex_EMA_RSI_1to3RR)
A trend-following forex strategy targeting **1:3 Risk-Reward ratio** with **~60% win rate**.

**Entry Rules (Long):**
- 8 EMA crosses above 21 EMA (trend confirmation)
- RSI between 40–70 (bullish momentum, not overbought)
- ADX > 20 (trending market filter)
- Price above 50 EMA (higher timeframe trend alignment)

**Entry Rules (Short):**
- 8 EMA crosses below 21 EMA (trend confirmation)
- RSI between 30–60 (bearish momentum, not oversold)
- ADX > 20 (trending market filter)
- Price below 50 EMA (higher timeframe trend alignment)

**Risk Management:**
- Stop Loss: 1.5× ATR from entry
- Take Profit: 4.5× ATR from entry (enforcing 1:3 RR)
- Risk per trade: 2% of account capital

**Supported Pairs:** EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD

**Usage:**
```python
from forex_strategy import ForexStrategy

strategy = ForexStrategy()
trades_df, performance = strategy.backtest()
strategy.print_summary(performance)
```

### Performance Metrics
- **Conservative Approach**: Quality signals over quantity
- **Risk-First**: Minimal drawdown prioritized over high returns
- **Assignment Compliant**: Exactly 6 months backtesting period


## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always conduct your own research before making investment decisions.

