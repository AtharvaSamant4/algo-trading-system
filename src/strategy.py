# src/strategy.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RSIMAStrategy:
    def __init__(self):
        self.starting_capital = 100000
        self.current_capital = 100000
        self.positions = {}
        self.trades = []
        self.max_position_percent = 0.1
        self.transaction_cost = 0.002
        self.current_symbol = None
        self.ml_engine = None
        
    def set_ml_engine(self, ml_engine):
        """Set the ML engine for predictions during backtesting"""
        self.ml_engine = ml_engine
        
    def generate_signals(self, df, symbol):
        """Generate buy/sell signals based on RSI + MA crossover"""
        try:
            signals = pd.DataFrame(index=df.index)
            signals['Price'] = df['Close']
            signals['RSI'] = df['RSI']
            signals['MA_20'] = df['MA_20']
            signals['MA_50'] = df['MA_50']
            signals['Volume_Ratio'] = df['Volume_Ratio']
            
            signals['Signal'] = 0
            signals['Position'] = 0
            
            # Buy condition: RSI < 35 + MA crossover + Volume confirmation
            buy_condition = (
                (df['RSI'] < 30) &
                (df['MA_20'] > df['MA_50']) &
                (df['Volume_Ratio'] > 1.2) &
                (df['RSI'].shift(1) >= 30))
            
            # Sell conditions: RSI > 65 or MA bearish crossover
            sell_condition = (
                (df['RSI'] > 65) |
                (df['MA_20'] < df['MA_50']))
            
            signals.loc[buy_condition, 'Signal'] = 1
            signals.loc[sell_condition, 'Signal'] = -1
            signals['Position'] = signals['Signal'].fillna(0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {str(e)}")
            return pd.DataFrame(index=df.index, columns=['Signal', 'Position']).fillna(0)
    
    def calculate_position_size(self, price, available_capital):
        """Calculate position size based on available capital"""
        try:
            max_investment = available_capital * self.max_position_percent
            quantity = int(max_investment / price)
            
            if quantity < 1:
                quantity = 0
                
            return quantity
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {str(e)}")
            return 0
    
    def get_ml_prediction_for_date(self, df, target_date, symbol):
        """Get ML prediction using the pre-trained model"""
        try:
            if target_date not in df.index:
                return 1, 0.5
                
            date_loc = df.index.get_loc(target_date)
            if date_loc < 50:
                return 1, 0.5
                
            historical_data = df.iloc[:date_loc].copy()
            
            if len(historical_data) < 50:
                return 1, 0.5
            
            # Use the pre-trained ML engine
            if self.ml_engine:
                try:
                    prediction, confidence = self.ml_engine.predict_next_day(historical_data)
                    confidence = max(0.5, min(0.95, float(confidence)))
                    return int(prediction), confidence
                except Exception as e:
                    logger.warning(f"ML prediction failed for {symbol} on {target_date}: {str(e)}")
            
            # Fallback: Use RSI-based prediction
            current_rsi = df.loc[target_date, 'RSI']
            if current_rsi < 35:
                prediction = 1
                confidence = min(0.9, (35 - current_rsi) / 35 * 0.4 + 0.5)
            elif current_rsi > 65:
                prediction = 0
                confidence = min(0.9, (current_rsi - 65) / 35 * 0.4 + 0.5)
            else:
                prediction = 1
                confidence = 0.5
                
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol} on {target_date}: {str(e)}")
            return 1, 0.5
    
    def backtest_strategy(self, data_dict, ml_engine=None):
        """Backtest strategy with realistic position sizing and ML integration"""
        self.ml_engine = ml_engine
        
        all_trades = []
        self.current_capital = self.starting_capital
        
        portfolio_values = []
        dates = []
        
        if self.ml_engine:
            logger.info("Using provided ML engine for backtesting")
        else:
            logger.warning("No ML engine provided, using RSI-based predictions")
        
        for symbol, df in data_dict.items():
            self.current_symbol = symbol
            logger.info(f"Backtesting {symbol} with ML predictions...")
            
            signals = self.generate_signals(df, symbol)
            if signals.empty:
                logger.warning(f"No signals generated for {symbol}")
                continue
                
            trades, portfolio_history = self._execute_trades_realistic(signals, symbol, df)
            all_trades.extend(trades)
            
            for date, value in portfolio_history:
                portfolio_values.append(value)
                dates.append(date)
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            performance = self._calculate_realistic_performance(trades_df, portfolio_values, dates)
            return trades_df, performance
        
        return pd.DataFrame(), {}
    
    def _execute_trades_realistic(self, signals, symbol, df):
        """Execute trades with realistic capital management and ML predictions"""
        trades = []
        portfolio_history = []
        position_qty = 0
        position_value = 0
        entry_price = 0
        
        for date, row in signals.iterrows():
            try:
                current_price = row['Price']
                
                ml_prediction, ml_confidence = self.get_ml_prediction_for_date(df, date, symbol)
                ml_prediction_text = 'UP' if ml_prediction == 1 else 'DOWN'
                
                total_portfolio_value = self.current_capital + position_value
                portfolio_history.append((date, total_portfolio_value))
                
                if row['Signal'] == 1 and position_qty == 0:  # Buy signal
                    quantity = self.calculate_position_size(current_price, self.current_capital)
                    
                    if quantity > 0:
                        trade_value = quantity * current_price
                        transaction_cost_amount = trade_value * self.transaction_cost
                        total_cost = trade_value + transaction_cost_amount
                        
                        if total_cost <= self.current_capital:
                            position_qty = quantity
                            entry_price = current_price
                            position_value = trade_value
                            self.current_capital -= total_cost
                            
                            trades.append({
                                'Symbol': symbol,
                                'Date': date,
                                'Action': 'BUY',
                                'Price': entry_price,
                                'Quantity': quantity,
                                'Trade_Value': trade_value,
                                'Transaction_Cost': transaction_cost_amount,
                                'RSI': row['RSI'],
                                'MA_Signal': 'Bullish' if row['MA_20'] > row['MA_50'] else 'Bearish',
                                'ML_Prediction': ml_prediction_text,
                                'ML_Confidence': round(ml_confidence * 100, 1),
                                'P&L': 0,
                                'P&L_Percent': 0,
                                'Portfolio_Value': self.current_capital + position_value,
                                'Strategy': 'RSI_MA_ML',
                                'Notes': f'RSI:{row["RSI"]:.1f}, ML:{ml_prediction_text}({ml_confidence:.2f})'
                            })
                
                elif row['Signal'] == -1 and position_qty > 0:  # Sell signal
                    trade_value = position_qty * current_price
                    transaction_cost_amount = trade_value * self.transaction_cost
                    net_proceeds = trade_value - transaction_cost_amount
                    
                    gross_pnl = (current_price - entry_price) * position_qty
                    net_pnl = gross_pnl - (transaction_cost_amount + (position_value * self.transaction_cost))
                    pnl_percent = (net_pnl / position_value) * 100
                    
                    self.current_capital += net_proceeds
                    
                    trades.append({
                        'Symbol': symbol,
                        'Date': date,
                        'Action': 'SELL',
                        'Price': current_price,
                        'Quantity': position_qty,
                        'Trade_Value': trade_value,
                        'Transaction_Cost': transaction_cost_amount,
                        'RSI': row['RSI'],
                        'MA_Signal': 'Bullish' if row['MA_20'] > row['MA_50'] else 'Bearish',
                        'ML_Prediction': ml_prediction_text,
                        'ML_Confidence': round(ml_confidence * 100, 1),
                        'P&L': net_pnl,
                        'P&L_Percent': pnl_percent,
                        'Portfolio_Value': self.current_capital,
                        'Strategy': 'RSI_MA_ML',
                        'Notes': f'P&L:₹{net_pnl:.0f}, ROI:{pnl_percent:.1f}%'
                    })
                    
                    position_qty = 0
                    position_value = 0
                    
            except Exception as e:
                logger.error(f"Trade execution failed at {date} for {symbol}: {str(e)}")
                
        return trades, portfolio_history
    
    def _calculate_realistic_performance(self, trades_df, portfolio_values, dates):
        """Calculate realistic performance metrics"""
        total_trades = len(trades_df)
        sell_trades = trades_df[trades_df['Action'] == 'SELL']
        
        if sell_trades.empty:
            return self._get_empty_performance()
        
        try:
            total_pnl = sell_trades['P&L'].sum()
            winning_trades = len(sell_trades[sell_trades['P&L'] > 0])
            losing_trades = len(sell_trades[sell_trades['P&L'] < 0])
            win_rate = (winning_trades / len(sell_trades)) * 100
            
            final_value = self.current_capital
            total_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
            
            if len(dates) > 252:  # More than 1 year of data
                years = len(dates) / 252
                cagr = ((final_value / self.starting_capital) ** (1/years) - 1) * 100
            else:
                cagr = total_return
            
            returns = sell_trades['P&L_Percent'].values
            avg_return_percent = np.mean(returns) if len(returns) > 0 else 0
            volatility = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = (avg_return_percent / volatility) if volatility > 0 else 0
            
            max_drawdown = self._calculate_realistic_drawdown(portfolio_values)
            ml_accuracy = self._calculate_ml_accuracy(sell_trades) if not sell_trades.empty else 0
            
            return {
                'Total_Trades': total_trades,
                'Total_PnL': total_pnl,
                'Win_Rate': win_rate,
                'Winning_Trades': winning_trades,
                'Losing_Trades': losing_trades,
                'Average_Return_Percent': avg_return_percent,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'ROI': total_return,
                'CAGR': cagr,
                'Final_Portfolio_Value': final_value,
                'Starting_Capital': self.starting_capital,
                'Total_Transaction_Costs': sell_trades['Transaction_Cost'].sum() * 2,
                'ML_Accuracy': ml_accuracy
            }
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {str(e)}")
            return self._get_empty_performance()
    
    def _calculate_ml_accuracy(self, sell_trades):
        """Calculate ML prediction accuracy based on actual trade outcomes"""
        try:
            if len(sell_trades) == 0:
                return 0
            
            correct_predictions = 0
            total_predictions = 0
            
            for _, trade in sell_trades.iterrows():
                if pd.notna(trade['ML_Prediction']) and trade['ML_Prediction'] != '':
                    total_predictions += 1
                    
                    trade_profitable = trade['P&L'] > 0
                    ml_predicted_up = trade['ML_Prediction'] == 'UP'
                    
                    if (ml_predicted_up and trade_profitable) or (not ml_predicted_up and not trade_profitable):
                        correct_predictions += 1
            
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                return round(accuracy, 1)
            
        except Exception as e:
            logger.error(f"ML accuracy calculation failed: {str(e)}")
        
        return 0
    
    def _calculate_realistic_drawdown(self, portfolio_values):
        """Calculate realistic maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0
            
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max * 100
        
        return drawdown.min()
    
    def _get_empty_performance(self):
        """Return empty performance metrics"""
        return {
            'Total_Trades': 0,
            'Total_PnL': 0,
            'Win_Rate': 0,
            'Winning_Trades': 0,
            'Losing_Trades': 0,
            'Average_Return_Percent': 0,
            'Volatility': 0,
            'Sharpe_Ratio': 0,
            'Max_Drawdown': 0,
            'ROI': 0,
            'CAGR': 0,
            'Final_Portfolio_Value': self.starting_capital,
            'Starting_Capital': self.starting_capital,
            'Total_Transaction_Costs': 0,
            'ML_Accuracy': 0
        }
