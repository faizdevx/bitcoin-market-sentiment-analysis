import pandas as pd
import numpy as np


SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def add_trade_features(df):
    df = df.copy()
    df["is_win"] = df["closedPnL"] > 0
    df["abs_pnl"] = df["closedPnL"].abs()
    df["notional"] = df["execution_price"] * df["size"]
    df["pnl_pct"] = df["closedPnL"] / (df["notional"] / df["leverage"]).replace(0, np.nan)
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.day_name()
    df["month"] = df["time"].dt.to_period("M").astype(str)
    df["quarter"] = df["time"].dt.to_period("Q").astype(str)
    df["sentiment"] = pd.Categorical(df["sentiment"], categories=SENTIMENT_ORDER, ordered=True)
    return df


def build_daily_summary(df):
    daily = df.groupby(["date", "sentiment", "fg_value"]).agg(
        total_trades=("closedPnL", "count"),
        total_pnl=("closedPnL", "sum"),
        avg_pnl=("closedPnL", "mean"),
        win_rate=("is_win", "mean"),
        avg_leverage=("leverage", "mean"),
        long_ratio=("side", lambda x: (x == "LONG").mean()),
        unique_traders=("account", "nunique"),
        liquidations=("event", lambda x: (x == "LIQUIDATION").sum()),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date")


def build_account_summary(df):
    acct = df.groupby("account").agg(
        total_trades=("closedPnL", "count"),
        total_pnl=("closedPnL", "sum"),
        win_rate=("is_win", "mean"),
        avg_leverage=("leverage", "mean"),
        avg_size=("size", "mean"),
        symbols_traded=("symbol", "nunique"),
        most_traded_symbol=("symbol", lambda x: x.mode()[0]),
    ).reset_index()
    acct["trader_type"] = pd.cut(
        acct["total_pnl"],
        bins=[-np.inf, -500, 0, 500, 2000, np.inf],
        labels=["Big Loser", "Slight Loser", "Break Even", "Moderate Winner", "Top Performer"],
    )
    return acct.sort_values("total_pnl", ascending=False)


def build_sentiment_summary(df):
    sent = df.groupby("sentiment", observed=True).agg(
        total_trades=("closedPnL", "count"),
        total_pnl=("closedPnL", "sum"),
        avg_pnl=("closedPnL", "mean"),
        median_pnl=("closedPnL", "median"),
        win_rate=("is_win", "mean"),
        avg_leverage=("leverage", "mean"),
        long_ratio=("side", lambda x: (x == "LONG").mean()),
        liquidation_rate=("event", lambda x: (x == "LIQUIDATION").mean()),
    ).reset_index()
    return sent
