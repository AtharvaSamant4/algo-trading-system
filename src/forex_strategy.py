# src/forex_strategy.py
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ForexStrategy:
    """
    Forex trading strategy targeting 1:3 Risk-Reward ratio with ~60% win rate.

    Strategy uses EMA crossover for trend direction, RSI for momentum
    confirmation, and ATR for dynamic stop-loss/take-profit placement.

    Entry Rules (Long):
        - 8 EMA crosses above 21 EMA (trend confirmation)
        - RSI is between 40 and 70 (bullish momentum, not overbought)
        - ADX > 20 (trending market filter)
        - Price is above 50 EMA (higher timeframe trend alignment)

    Entry Rules (Short):
        - 8 EMA crosses below 21 EMA (trend confirmation)
        - RSI is between 30 and 60 (bearish momentum, not oversold)
        - ADX > 20 (trending market filter)
        - Price is below 50 EMA (higher timeframe trend alignment)

    Risk Management:
        - Stop Loss: 1.5x ATR from entry
        - Take Profit: 4.5x ATR from entry (1:3 RR ratio)
        - Risk per trade: 2% of account capital
    """

    FOREX_PAIRS = [
        'EURUSD=X',
        'GBPUSD=X',
        'USDJPY=X',
        'AUDUSD=X',
        'USDCAD=X',
    ]

    def __init__(self, starting_capital=100000, risk_per_trade=0.02):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.risk_per_trade = risk_per_trade
        self.rr_ratio = 3.0
        self.atr_sl_multiplier = 1.5
        self.atr_tp_multiplier = self.atr_sl_multiplier * self.rr_ratio
        self.transaction_cost = 0.0002  # 2 pips spread cost
        self.trades = []
        self.positions = {}

    def fetch_forex_data(self, pair, period='6mo'):
        """Fetch forex data using yfinance"""
        try:
            logger.info(f"Fetching forex data for {pair}...")
            ticker = yf.Ticker(pair)
            df = ticker.history(period=period)

            if df.empty:
                logger.warning(f"No data returned for {pair}")
                return pd.DataFrame()

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} for {pair}")
                    return pd.DataFrame()

            df = df[required_columns].copy()
            df = df[df['Close'] > 0]

            logger.info(f"{pair}: Fetched {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {pair}: {e}")
            return pd.DataFrame()

    def add_indicators(self, df):
        """Add technical indicators required for the strategy"""
        try:
            if len(df) < 60:
                logger.warning(f"Insufficient data for indicators: {len(df)} rows")
                return pd.DataFrame()

            df = df.copy()

            # Exponential Moving Averages
            df['EMA_8'] = ta.ema(df['Close'], length=8)
            df['EMA_21'] = ta.ema(df['Close'], length=21)
            df['EMA_50'] = ta.ema(df['Close'], length=50)

            # RSI
            df['RSI'] = ta.rsi(df['Close'], length=14)

            # ATR for stop-loss / take-profit calculation
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            # ADX for trend strength
            adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            if adx_data is not None and not adx_data.empty:
                df['ADX'] = adx_data.iloc[:, 0]
            else:
                df['ADX'] = 25.0

            # EMA crossover detection
            df['EMA_Cross_Up'] = (
                (df['EMA_8'] > df['EMA_21']) &
                (df['EMA_8'].shift(1) <= df['EMA_21'].shift(1))
            )
            df['EMA_Cross_Down'] = (
                (df['EMA_8'] < df['EMA_21']) &
                (df['EMA_8'].shift(1) >= df['EMA_21'].shift(1))
            )

            # Clean NaN values
            df = df.dropna()

            logger.info(f"Indicators added: {len(df)} clean rows")
            return df

        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return pd.DataFrame()

    def generate_signals(self, df, pair):
        """Generate buy/sell signals with trend-following filters"""
        try:
            signals = pd.DataFrame(index=df.index)
            signals['Price'] = df['Close']
            signals['ATR'] = df['ATR']
            signals['Signal'] = 0

            # Long entry conditions
            long_condition = (
                df['EMA_Cross_Up'] &
                (df['RSI'] > 40) & (df['RSI'] < 70) &
                (df['ADX'] > 20) &
                (df['Close'] > df['EMA_50'])
            )

            # Short entry conditions
            short_condition = (
                df['EMA_Cross_Down'] &
                (df['RSI'] > 30) & (df['RSI'] < 60) &
                (df['ADX'] > 20) &
                (df['Close'] < df['EMA_50'])
            )

            signals.loc[long_condition, 'Signal'] = 1
            signals.loc[short_condition, 'Signal'] = -1

            return signals

        except Exception as e:
            logger.error(f"Signal generation failed for {pair}: {e}")
            return pd.DataFrame(
                index=df.index, columns=['Signal', 'Price', 'ATR']
            ).fillna(0)

    def calculate_position_size(self, entry_price, stop_loss_distance, capital):
        """Calculate position size based on risk per trade and stop distance"""
        try:
            risk_amount = capital * self.risk_per_trade
            if stop_loss_distance <= 0:
                return 0
            position_size = risk_amount / stop_loss_distance
            return int(position_size)
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0

    def backtest(self, data_dict=None, pairs=None, period='6mo'):
        """
        Backtest the forex strategy with 1:3 RR enforcement.

        Args:
            data_dict: Pre-fetched data dict {pair: DataFrame}. If None,
                       data is fetched using yfinance.
            pairs: List of forex pairs. Defaults to FOREX_PAIRS.
            period: Data period for fetching. Default '6mo'.

        Returns:
            Tuple of (trades_df, performance_dict)
        """
        self.current_capital = self.starting_capital
        all_trades = []

        if data_dict is None:
            if pairs is None:
                pairs = self.FOREX_PAIRS
            data_dict = {}
            for pair in pairs:
                df = self.fetch_forex_data(pair, period)
                if not df.empty:
                    data_dict[pair] = df

        if not data_dict:
            logger.error("No forex data available for backtesting")
            return pd.DataFrame(), self._get_empty_performance()

        for pair, raw_df in data_dict.items():
            logger.info(f"Backtesting {pair}...")

            df = self.add_indicators(raw_df)
            if df.empty:
                logger.warning(f"Skipping {pair}: no indicator data")
                continue

            signals = self.generate_signals(df, pair)
            trades = self._execute_trades(signals, df, pair)
            all_trades.extend(trades)

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            performance = self._calculate_performance(trades_df)
            return trades_df, performance

        return pd.DataFrame(), self._get_empty_performance()

    def _execute_trades(self, signals, df, pair):
        """Execute trades with 1:3 RR stop-loss and take-profit enforcement"""
        trades = []
        in_position = False
        position_direction = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        entry_date = None

        for date, row in signals.iterrows():
            current_price = row['Price']
            current_atr = row['ATR']

            if current_atr <= 0 or current_price <= 0:
                continue

            # Check exit conditions for open positions
            if in_position:
                if position_direction == 1:  # Long position
                    current_high = df.loc[date, 'High']
                    current_low = df.loc[date, 'Low']
                    if current_low <= stop_loss:
                        pnl = stop_loss - entry_price
                        trades.append(self._create_trade_record(
                            pair, entry_date, date, 'LONG', entry_price,
                            stop_loss, pnl, 'Stop Loss Hit'
                        ))
                        in_position = False
                    elif current_high >= take_profit:
                        pnl = take_profit - entry_price
                        trades.append(self._create_trade_record(
                            pair, entry_date, date, 'LONG', entry_price,
                            take_profit, pnl, 'Take Profit Hit'
                        ))
                        in_position = False

                elif position_direction == -1:  # Short position
                    current_high = df.loc[date, 'High']
                    current_low = df.loc[date, 'Low']
                    if current_high >= stop_loss:
                        pnl = entry_price - stop_loss
                        trades.append(self._create_trade_record(
                            pair, entry_date, date, 'SHORT', entry_price,
                            stop_loss, pnl, 'Stop Loss Hit'
                        ))
                        in_position = False
                    elif current_low <= take_profit:
                        pnl = entry_price - take_profit
                        trades.append(self._create_trade_record(
                            pair, entry_date, date, 'SHORT', entry_price,
                            take_profit, pnl, 'Take Profit Hit'
                        ))
                        in_position = False

            # Open new positions only when flat
            if not in_position and row['Signal'] != 0:
                sl_distance = current_atr * self.atr_sl_multiplier
                tp_distance = current_atr * self.atr_tp_multiplier

                if row['Signal'] == 1:  # Long
                    entry_price = current_price
                    stop_loss = entry_price - sl_distance
                    take_profit = entry_price + tp_distance
                    position_direction = 1
                    in_position = True
                    entry_date = date

                elif row['Signal'] == -1:  # Short
                    entry_price = current_price
                    stop_loss = entry_price + sl_distance
                    take_profit = entry_price - tp_distance
                    position_direction = -1
                    in_position = True
                    entry_date = date

        return trades

    def _create_trade_record(self, pair, entry_date, exit_date, direction,
                             entry_price, exit_price, pnl, exit_reason):
        """Create a standardized trade record"""
        cost = abs(entry_price * self.transaction_cost) + abs(
            exit_price * self.transaction_cost
        )
        net_pnl = pnl - cost
        pnl_percent = (net_pnl / entry_price) * 100 if entry_price else 0

        self.current_capital += net_pnl

        return {
            'Pair': pair,
            'Direction': direction,
            'Entry_Date': entry_date,
            'Exit_Date': exit_date,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Gross_PnL': pnl,
            'Transaction_Cost': cost,
            'Net_PnL': net_pnl,
            'PnL_Percent': pnl_percent,
            'Exit_Reason': exit_reason,
            'Portfolio_Value': self.current_capital,
            'Strategy': 'Forex_EMA_RSI_1to3RR',
            'RR_Ratio': '1:3',
        }

    def _calculate_performance(self, trades_df):
        """Calculate strategy performance metrics"""
        try:
            total_trades = len(trades_df)
            if total_trades == 0:
                return self._get_empty_performance()

            winning = trades_df[trades_df['Net_PnL'] > 0]
            losing = trades_df[trades_df['Net_PnL'] <= 0]
            win_count = len(winning)
            loss_count = len(losing)
            win_rate = (win_count / total_trades) * 100

            total_pnl = trades_df['Net_PnL'].sum()
            avg_win = winning['Net_PnL'].mean() if win_count > 0 else 0
            avg_loss = abs(losing['Net_PnL'].mean()) if loss_count > 0 else 0
            risk_reward = (avg_win / avg_loss) if avg_loss > 0 else 0

            final_value = self.current_capital
            total_return = (
                (final_value - self.starting_capital) / self.starting_capital
            ) * 100

            returns = trades_df['PnL_Percent'].values
            avg_return = np.mean(returns) if len(returns) > 0 else 0
            volatility = np.std(returns) if len(returns) > 1 else 0
            sharpe = (avg_return / volatility) if volatility > 0 else 0

            portfolio_values = trades_df['Portfolio_Value'].values
            max_drawdown = self._calculate_drawdown(portfolio_values)

            tp_trades = len(
                trades_df[trades_df['Exit_Reason'] == 'Take Profit Hit']
            )
            sl_trades = len(
                trades_df[trades_df['Exit_Reason'] == 'Stop Loss Hit']
            )

            return {
                'Total_Trades': total_trades,
                'Winning_Trades': win_count,
                'Losing_Trades': loss_count,
                'Win_Rate': win_rate,
                'Total_PnL': total_pnl,
                'Average_Win': avg_win,
                'Average_Loss': avg_loss,
                'Risk_Reward_Ratio': risk_reward,
                'Target_RR': '1:3',
                'Average_Return_Percent': avg_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe,
                'Max_Drawdown': max_drawdown,
                'ROI': total_return,
                'Final_Portfolio_Value': final_value,
                'Starting_Capital': self.starting_capital,
                'Take_Profit_Exits': tp_trades,
                'Stop_Loss_Exits': sl_trades,
            }

        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return self._get_empty_performance()

    def _calculate_drawdown(self, portfolio_values):
        """Calculate maximum drawdown from portfolio values"""
        if len(portfolio_values) < 2:
            return 0
        series = pd.Series(portfolio_values)
        running_max = series.expanding().max()
        drawdown = (series - running_max) / running_max * 100
        return drawdown.min()

    def _get_empty_performance(self):
        """Return empty performance metrics"""
        return {
            'Total_Trades': 0,
            'Winning_Trades': 0,
            'Losing_Trades': 0,
            'Win_Rate': 0,
            'Total_PnL': 0,
            'Average_Win': 0,
            'Average_Loss': 0,
            'Risk_Reward_Ratio': 0,
            'Target_RR': '1:3',
            'Average_Return_Percent': 0,
            'Volatility': 0,
            'Sharpe_Ratio': 0,
            'Max_Drawdown': 0,
            'ROI': 0,
            'Final_Portfolio_Value': self.starting_capital,
            'Starting_Capital': self.starting_capital,
            'Take_Profit_Exits': 0,
            'Stop_Loss_Exits': 0,
        }

    def print_summary(self, performance):
        """Print strategy performance summary"""
        print("\n" + "=" * 60)
        print("📊 FOREX STRATEGY - PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"🎯 Strategy: EMA Crossover + RSI + ADX (1:3 RR)")
        print(f"   • Total Trades: {performance.get('Total_Trades', 0)}")
        print(f"   • Win Rate: {performance.get('Win_Rate', 0):.1f}%")
        print(f"   • Risk-Reward Ratio: {performance.get('Risk_Reward_Ratio', 0):.2f}")
        print(f"   • Target RR: {performance.get('Target_RR', '1:3')}")
        print(f"   • Winning Trades: {performance.get('Winning_Trades', 0)}")
        print(f"   • Losing Trades: {performance.get('Losing_Trades', 0)}")
        print(f"💰 P&L:")
        print(f"   • Total P&L: ${performance.get('Total_PnL', 0):,.2f}")
        print(f"   • Average Win: ${performance.get('Average_Win', 0):,.2f}")
        print(f"   • Average Loss: ${performance.get('Average_Loss', 0):,.2f}")
        print(f"   • ROI: {performance.get('ROI', 0):.2f}%")
        print(f"📈 Risk Metrics:")
        print(f"   • Sharpe Ratio: {performance.get('Sharpe_Ratio', 0):.2f}")
        print(f"   • Max Drawdown: {performance.get('Max_Drawdown', 0):.2f}%")
        print(f"   • Take Profit Exits: {performance.get('Take_Profit_Exits', 0)}")
        print(f"   • Stop Loss Exits: {performance.get('Stop_Loss_Exits', 0)}")
        print(f"💼 Portfolio:")
        print(f"   • Starting Capital: ${performance.get('Starting_Capital', 0):,.2f}")
        print(f"   • Final Value: ${performance.get('Final_Portfolio_Value', 0):,.2f}")
        print("=" * 60)
