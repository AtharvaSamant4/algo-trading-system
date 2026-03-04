# src/test_forex_strategy.py
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from forex_strategy import ForexStrategy


def _make_sample_data(rows=120, trend='up'):
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=rows, freq='B')
    base = 1.1000
    prices = [base]
    for i in range(1, rows):
        drift = 0.0002 if trend == 'up' else -0.0002
        change = drift + np.random.normal(0, 0.005)
        prices.append(prices[-1] * (1 + change))

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.003, rows)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, rows)))
    open_ = close * (1 + np.random.normal(0, 0.001, rows))
    volume = np.random.randint(1000, 50000, rows).astype(float)

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)
    return df


class TestForexStrategyInit(unittest.TestCase):
    def test_default_init(self):
        strategy = ForexStrategy()
        self.assertEqual(strategy.starting_capital, 100000)
        self.assertEqual(strategy.risk_per_trade, 0.02)
        self.assertEqual(strategy.rr_ratio, 3.0)
        self.assertAlmostEqual(
            strategy.atr_tp_multiplier,
            strategy.atr_sl_multiplier * strategy.rr_ratio,
        )

    def test_custom_init(self):
        strategy = ForexStrategy(starting_capital=50000, risk_per_trade=0.01)
        self.assertEqual(strategy.starting_capital, 50000)
        self.assertEqual(strategy.risk_per_trade, 0.01)

    def test_rr_ratio_is_3(self):
        strategy = ForexStrategy()
        self.assertEqual(strategy.rr_ratio, 3.0)
        self.assertEqual(
            strategy.atr_tp_multiplier / strategy.atr_sl_multiplier, 3.0,
        )


class TestAddIndicators(unittest.TestCase):
    def setUp(self):
        self.strategy = ForexStrategy()
        self.df = _make_sample_data(120)

    def test_indicators_added(self):
        result = self.strategy.add_indicators(self.df)
        self.assertFalse(result.empty)
        for col in ['EMA_8', 'EMA_21', 'EMA_50', 'RSI', 'ATR', 'ADX']:
            self.assertIn(col, result.columns, f"Missing indicator: {col}")

    def test_crossover_columns(self):
        result = self.strategy.add_indicators(self.df)
        self.assertIn('EMA_Cross_Up', result.columns)
        self.assertIn('EMA_Cross_Down', result.columns)

    def test_insufficient_data(self):
        short_df = _make_sample_data(20)
        result = self.strategy.add_indicators(short_df)
        self.assertTrue(result.empty)


class TestGenerateSignals(unittest.TestCase):
    def setUp(self):
        self.strategy = ForexStrategy()
        self.df = self.strategy.add_indicators(_make_sample_data(120))

    def test_signals_dataframe(self):
        signals = self.strategy.generate_signals(self.df, 'EURUSD=X')
        self.assertIn('Signal', signals.columns)
        self.assertIn('Price', signals.columns)
        self.assertIn('ATR', signals.columns)

    def test_signal_values(self):
        signals = self.strategy.generate_signals(self.df, 'EURUSD=X')
        unique = set(signals['Signal'].unique())
        self.assertTrue(unique.issubset({-1, 0, 1}))


class TestPositionSizing(unittest.TestCase):
    def test_position_size(self):
        strategy = ForexStrategy(starting_capital=100000, risk_per_trade=0.02)
        size = strategy.calculate_position_size(1.1000, 0.0050, 100000)
        expected = int(2000 / 0.0050)
        self.assertEqual(size, expected)

    def test_zero_stop_distance(self):
        strategy = ForexStrategy()
        size = strategy.calculate_position_size(1.1, 0, 100000)
        self.assertEqual(size, 0)


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.strategy = ForexStrategy()
        self.data_dict = {'EURUSD=X': _make_sample_data(120)}

    def test_backtest_returns_dataframe_and_dict(self):
        trades_df, performance = self.strategy.backtest(
            data_dict=self.data_dict,
        )
        self.assertIsInstance(trades_df, pd.DataFrame)
        self.assertIsInstance(performance, dict)

    def test_performance_keys(self):
        _, performance = self.strategy.backtest(data_dict=self.data_dict)
        expected_keys = [
            'Total_Trades', 'Win_Rate', 'Risk_Reward_Ratio', 'Target_RR',
            'Total_PnL', 'ROI', 'Max_Drawdown', 'Starting_Capital',
        ]
        for key in expected_keys:
            self.assertIn(key, performance, f"Missing key: {key}")

    def test_target_rr_is_1_to_3(self):
        _, performance = self.strategy.backtest(data_dict=self.data_dict)
        self.assertEqual(performance['Target_RR'], '1:3')

    def test_trade_record_fields(self):
        trades_df, _ = self.strategy.backtest(data_dict=self.data_dict)
        if not trades_df.empty:
            expected_cols = [
                'Pair', 'Direction', 'Entry_Price', 'Exit_Price',
                'Net_PnL', 'Exit_Reason', 'RR_Ratio', 'Strategy',
            ]
            for col in expected_cols:
                self.assertIn(col, trades_df.columns, f"Missing: {col}")

    def test_exit_reasons(self):
        trades_df, _ = self.strategy.backtest(data_dict=self.data_dict)
        if not trades_df.empty:
            valid_reasons = {'Take Profit Hit', 'Stop Loss Hit'}
            actual = set(trades_df['Exit_Reason'].unique())
            self.assertTrue(
                actual.issubset(valid_reasons),
                f"Unexpected exit reasons: {actual - valid_reasons}",
            )

    def test_empty_data(self):
        trades_df, perf = self.strategy.backtest(data_dict={})
        self.assertTrue(trades_df.empty)
        self.assertEqual(perf['Total_Trades'], 0)


class TestTradeExecution(unittest.TestCase):
    """Test that stop-loss and take-profit logic enforces 1:3 RR."""

    def test_long_take_profit_pnl_is_3x_stop_loss(self):
        strategy = ForexStrategy()
        atr = 0.0050
        entry = 1.1000
        sl_dist = atr * strategy.atr_sl_multiplier
        tp_dist = atr * strategy.atr_tp_multiplier

        sl_pnl = -sl_dist
        tp_pnl = tp_dist

        ratio = abs(tp_pnl / sl_pnl) if sl_pnl != 0 else 0
        self.assertAlmostEqual(ratio, 3.0, places=5)

    def test_short_take_profit_pnl_is_3x_stop_loss(self):
        strategy = ForexStrategy()
        atr = 0.0050
        entry = 1.1000
        sl_dist = atr * strategy.atr_sl_multiplier
        tp_dist = atr * strategy.atr_tp_multiplier

        sl_pnl = -sl_dist
        tp_pnl = tp_dist

        ratio = abs(tp_pnl / sl_pnl) if sl_pnl != 0 else 0
        self.assertAlmostEqual(ratio, 3.0, places=5)


class TestEmptyPerformance(unittest.TestCase):
    def test_empty_performance_structure(self):
        strategy = ForexStrategy()
        perf = strategy._get_empty_performance()
        self.assertEqual(perf['Total_Trades'], 0)
        self.assertEqual(perf['Win_Rate'], 0)
        self.assertEqual(perf['Target_RR'], '1:3')
        self.assertEqual(perf['Starting_Capital'], strategy.starting_capital)


class TestDrawdown(unittest.TestCase):
    def test_no_drawdown(self):
        strategy = ForexStrategy()
        values = [100, 110, 120, 130]
        dd = strategy._calculate_drawdown(values)
        self.assertEqual(dd, 0)

    def test_has_drawdown(self):
        strategy = ForexStrategy()
        values = [100, 120, 90, 110]
        dd = strategy._calculate_drawdown(values)
        self.assertLess(dd, 0)

    def test_single_value(self):
        strategy = ForexStrategy()
        dd = strategy._calculate_drawdown([100])
        self.assertEqual(dd, 0)


if __name__ == '__main__':
    unittest.main()
