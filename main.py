import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.data_loader import load_fear_greed, load_trades, merge_datasets, save_processed
from src.feature_engineering import (
    add_trade_features,
    build_daily_summary,
    build_account_summary,
    build_sentiment_summary,
)
from src.analysis import (
    sentiment_pnl_analysis,
    contrarian_vs_momentum,
    top_vs_bottom_traders,
    leverage_risk_analysis,
    symbol_sentiment_performance,
    rolling_pnl,
    pearson_corr,
    fear_vs_greed_returns,
)
from src.visualizations import (
    plot_sentiment_distribution,
    plot_pnl_by_sentiment,
    plot_long_short_by_sentiment,
    plot_rolling_pnl,
    plot_top_vs_bottom,
    plot_leverage_heatmap,
    plot_symbol_pnl,
    plot_liquidation_rate,
    plot_pnl_distribution,
    plot_hourly_pnl,
)


def run():
    print("=" * 60)
    print("  Hyperliquid x Fear & Greed Index – Analysis Pipeline")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    fg = load_fear_greed()
    trades = load_trades()
    print(f"  Fear/Greed: {len(fg)} days | Trades: {len(trades)} records")

    print("\n[2/7] Merging and feature engineering...")
    merged = merge_datasets(trades, fg)
    merged = add_trade_features(merged)
    save_processed(merged, "data/processed/merged_data.csv")

    daily = build_daily_summary(merged)
    daily = rolling_pnl(daily)
    daily.to_csv("data/processed/daily_summary.csv", index=False)

    account_summary = build_account_summary(merged)
    account_summary.to_csv("data/processed/account_summary.csv", index=False)

    sent_summary = build_sentiment_summary(merged)

    print("\n[3/7] Core sentiment analysis...")
    sent_pnl = sentiment_pnl_analysis(merged)
    print("\n  PnL by Sentiment:")
    print(sent_pnl.to_string(index=False))

    print("\n  Fear vs Greed Statistical Test:")
    test_result = fear_vs_greed_returns(merged)
    for k, v in test_result.items():
        print(f"    {k}: {v}")

    corr, pval = pearson_corr(merged)
    print(f"\n  Pearson Correlation (FG Index vs PnL): r={corr}, p={pval}")

    print("\n[4/7] Contrarian vs Momentum analysis...")
    cv_m = contrarian_vs_momentum(merged)
    print(cv_m.to_string(index=False))

    print("\n[5/7] Top vs Bottom trader profiles...")
    top_bot = top_vs_bottom_traders(merged, n=10)
    lev_risk = leverage_risk_analysis(merged)
    sym_perf = symbol_sentiment_performance(merged)

    print("\n[6/7] Generating charts...")
    plot_sentiment_distribution(fg)
    plot_pnl_by_sentiment(sent_summary)
    plot_long_short_by_sentiment(merged)
    plot_rolling_pnl(daily)
    plot_top_vs_bottom(top_bot)
    plot_leverage_heatmap(lev_risk)
    plot_symbol_pnl(sym_perf)
    plot_liquidation_rate(sent_pnl)
    plot_pnl_distribution(merged)
    plot_hourly_pnl(merged)

    print("\n[7/7] Key findings summary:")
    best_sent = sent_pnl.loc[sent_pnl["avg_pnl"].idxmax(), "sentiment"]
    worst_sent = sent_pnl.loc[sent_pnl["avg_pnl"].idxmin(), "sentiment"]
    highest_liq = sent_pnl.loc[sent_pnl["liq_pct"].idxmax(), "sentiment"]

    print(f"  Best sentiment for average PnL     : {best_sent}")
    print(f"  Worst sentiment for average PnL    : {worst_sent}")
    print(f"  Highest liquidation rate in        : {highest_liq}")
    print(f"  Total unique traders               : {merged['account'].nunique()}")
    print(f"  Overall win rate                   : {merged['is_win'].mean():.2%}")
    print(f"  Overall avg PnL per trade          : {merged['closedPnL'].mean():.4f}")

    print("\n  Done. Check outputs/figures/ for all charts.")
    print("=" * 60)


if __name__ == "__main__":
    run()
