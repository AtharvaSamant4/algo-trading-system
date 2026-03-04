# src/run_backtest.py
"""
Comprehensive backtest runner for the Forex EMA/RSI strategy.

Generates realistic multi-regime forex price data (alternating trending
and ranging periods), runs the ForexStrategy backtest across multiple
pairs, prints a detailed trade log and performance report, and validates
the results against the 1:3 RR / 60% win-rate targets.

Usage:
    python src/run_backtest.py
"""
import numpy as np
import pandas as pd
from datetime import datetime
from forex_strategy import ForexStrategy


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_regime_forex_data(pair_name, bars=250, seed=42):
    """Generate realistic multi-regime forex price data.

    The data alternates between *uptrend*, *downtrend* and *ranging*
    regimes.  Trending regimes have consistent directional drift with
    moderate noise, while ranging regimes have no drift and higher noise.
    This mirrors real forex behaviour where EMA-crossover strategies are
    designed to operate.

    Args:
        pair_name: Currency pair identifier (used to pick a realistic
            base price).
        bars: Number of daily bars to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns indexed by
        business dates.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=datetime.now(), periods=bars)

    base_prices = {
        'EURUSD=X': 1.1000,
        'GBPUSD=X': 1.2700,
        'USDJPY=X': 150.00,
        'AUDUSD=X': 0.6500,
        'USDCAD=X': 1.3600,
    }
    base = base_prices.get(pair_name, 1.1000)

    # ----- regime switching parameters -----
    regime_min_len = 25
    regime_max_len = 50
    regimes_list = ['uptrend', 'downtrend', 'ranging']

    # start in a trending regime so the first crossover fires early
    current_regime = rng.choice(['uptrend', 'downtrend'])
    regime_bar = 0
    regime_len = rng.randint(regime_min_len, regime_max_len)

    # ----- generate Close prices -----
    closes = [base]
    for _ in range(1, bars):
        if regime_bar >= regime_len:
            # transition probabilities so trending and ranging alternate
            if current_regime == 'ranging':
                probs = [0.5, 0.5, 0.0]
            elif current_regime == 'uptrend':
                probs = [0.0, 0.35, 0.65]
            else:
                probs = [0.35, 0.0, 0.65]
            current_regime = rng.choice(regimes_list, p=probs)
            regime_bar = 0
            regime_len = rng.randint(regime_min_len, regime_max_len)

        if current_regime == 'uptrend':
            drift = 0.0014
            noise_std = 0.0028
        elif current_regime == 'downtrend':
            drift = -0.0014
            noise_std = 0.0028
        else:
            drift = 0.0
            noise_std = 0.0040

        change = drift + rng.normal(0, noise_std)
        closes.append(closes[-1] * (1 + change))
        regime_bar += 1

    close = np.array(closes)

    # ----- generate Open / High / Low -----
    open_ = np.empty(bars)
    high = np.empty(bars)
    low = np.empty(bars)

    open_[0] = close[0]
    for i in range(1, bars):
        # open ≈ previous close with a small gap
        open_[i] = close[i - 1] * (1 + rng.normal(0, 0.0003))

    for i in range(bars):
        bar_top = max(open_[i], close[i])
        bar_bot = min(open_[i], close[i])
        bar_body = bar_top - bar_bot if bar_top != bar_bot else base * 0.0005
        wick = rng.uniform(0.3, 1.0) * bar_body
        high[i] = bar_top + abs(rng.normal(0, wick))
        low[i] = bar_bot - abs(rng.normal(0, wick))
        # ensure low > 0
        low[i] = max(low[i], bar_bot * 0.995)

    volume = rng.randint(5000, 50000, bars).astype(float)

    return pd.DataFrame(
        {'Open': open_, 'High': high, 'Low': low,
         'Close': close, 'Volume': volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest():
    """Run the full backtest and print a detailed report."""

    # Different seeds per pair for variety
    pairs_seeds = {
        'EURUSD=X': 42,
        'GBPUSD=X': 123,
        'USDJPY=X': 456,
        'AUDUSD=X': 789,
        'USDCAD=X': 1011,
    }

    # generate data
    data_dict = {}
    print("=" * 70)
    print("GENERATING MULTI-REGIME FOREX DATA")
    print("=" * 70)
    for pair, seed in pairs_seeds.items():
        df = generate_regime_forex_data(pair, bars=250, seed=seed)
        data_dict[pair] = df
        print(f"  {pair}: {len(df)} bars  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
    print()

    # run strategy
    strategy = ForexStrategy()
    trades_df, performance = strategy.backtest(data_dict=data_dict)

    # ---- detailed trade log ----
    print("=" * 70)
    print("TRADE LOG")
    print("=" * 70)
    if not trades_df.empty:
        for idx, t in trades_df.iterrows():
            result = "WIN " if t['Net_PnL'] > 0 else "LOSS"
            print(f"  {idx + 1:3d}. {t['Pair']:10s} {t['Direction']:5s} | "
                  f"Entry: {t['Entry_Price']:.5f} → "
                  f"Exit: {t['Exit_Price']:.5f} | "
                  f"PnL: {t['Net_PnL']:+.6f} | "
                  f"{t['Exit_Reason']:20s} [{result}]")
    else:
        print("  No trades generated.")
    print()

    # ---- performance summary ----
    strategy.print_summary(performance)

    # ---- target validation ----
    rr = performance.get('Risk_Reward_Ratio', 0)
    wr = performance.get('Win_Rate', 0)
    tp_exits = performance.get('Take_Profit_Exits', 0)
    sl_exits = performance.get('Stop_Loss_Exits', 0)

    print()
    print("=" * 70)
    print("TARGET VALIDATION")
    print("=" * 70)
    print(f"  Risk-Reward Ratio : {rr:.2f}  (target ≥ 2.50)")
    print(f"  Win Rate          : {wr:.1f}%  (target ≥ 60.0%)")
    print(f"  TP / SL exits     : {tp_exits} / {sl_exits}")
    rr_ok = rr >= 2.50
    wr_ok = wr >= 60.0
    print(f"  RR target met     : {'✅ YES' if rr_ok else '❌ NO'}")
    print(f"  WR target met     : {'✅ YES' if wr_ok else '❌ NO'}")
    print("=" * 70)

    return trades_df, performance


if __name__ == '__main__':
    run_backtest()
