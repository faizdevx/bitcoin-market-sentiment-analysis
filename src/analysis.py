import pandas as pd
import numpy as np
from scipy import stats


def sentiment_pnl_analysis(df):
    result = df.groupby("sentiment", observed=True).agg(
        trades=("closedPnL", "count"),
        total_pnl=("closedPnL", "sum"),
        avg_pnl=("closedPnL", "mean"),
        win_rate=("is_win", "mean"),
        avg_leverage=("leverage", "mean"),
        long_pct=("side", lambda x: round((x == "LONG").mean() * 100, 1)),
        liq_pct=("event", lambda x: round((x == "LIQUIDATION").mean() * 100, 2)),
    ).reset_index()
    return result


def contrarian_vs_momentum(df):
    long_sentiment = df[df["side"] == "LONG"].groupby("sentiment", observed=True)["closedPnL"].mean()
    short_sentiment = df[df["side"] == "SHORT"].groupby("sentiment", observed=True)["closedPnL"].mean()

    comp = pd.DataFrame({"long_avg_pnl": long_sentiment, "short_avg_pnl": short_sentiment}).reset_index()
    comp["better_side"] = comp.apply(
        lambda r: "LONG" if r["long_avg_pnl"] > r["short_avg_pnl"] else "SHORT", axis=1
    )
    return comp


def top_vs_bottom_traders(df, n=10):
    acct_pnl = df.groupby("account")["closedPnL"].sum().reset_index()
    top = acct_pnl.nlargest(n, "closedPnL")["account"].tolist()
    bot = acct_pnl.nsmallest(n, "closedPnL")["account"].tolist()

    top_df = df[df["account"].isin(top)].copy()
    bot_df = df[df["account"].isin(bot)].copy()

    top_profile = top_df.groupby("sentiment", observed=True).agg(
        win_rate=("is_win", "mean"),
        avg_leverage=("leverage", "mean"),
        long_pct=("side", lambda x: (x == "LONG").mean()),
    ).reset_index()
    top_profile["group"] = "Top 10"

    bot_profile = bot_df.groupby("sentiment", observed=True).agg(
        win_rate=("is_win", "mean"),
        avg_leverage=("leverage", "mean"),
        long_pct=("side", lambda x: (x == "LONG").mean()),
    ).reset_index()
    bot_profile["group"] = "Bottom 10"

    return pd.concat([top_profile, bot_profile], ignore_index=True)


def leverage_risk_analysis(df):
    bins = [0, 2, 5, 10, 15, 25]
    labels = ["1-2x", "3-5x", "6-10x", "11-15x", "16-25x"]
    df = df.copy()
    df["lev_bucket"] = pd.cut(df["leverage"], bins=bins, labels=labels)
    result = df.groupby(["lev_bucket", "sentiment"], observed=True).agg(
        win_rate=("is_win", "mean"),
        avg_pnl=("closedPnL", "mean"),
        liq_rate=("event", lambda x: (x == "LIQUIDATION").mean()),
        trades=("closedPnL", "count"),
    ).reset_index()
    return result


def symbol_sentiment_performance(df):
    result = df.groupby(["symbol", "sentiment"], observed=True).agg(
        win_rate=("is_win", "mean"),
        avg_pnl=("closedPnL", "mean"),
        trades=("closedPnL", "count"),
    ).reset_index()
    return result


def rolling_pnl(daily_df, window=30):
    daily_df = daily_df.sort_values("date").copy()
    daily_df["rolling_pnl"] = daily_df["total_pnl"].rolling(window=window).mean()
    daily_df["rolling_win_rate"] = daily_df["win_rate"].rolling(window=window).mean()
    return daily_df


def pearson_corr(df):
    corr, pval = stats.pearsonr(df["fg_value"].dropna(), df["closedPnL"][df["fg_value"].notna()])
    return round(corr, 4), round(pval, 6)


def fear_vs_greed_returns(df):
    fear_days = df[df["sentiment"].isin(["Fear", "Extreme Fear"])]["closedPnL"]
    greed_days = df[df["sentiment"].isin(["Greed", "Extreme Greed"])]["closedPnL"]
    t_stat, p_val = stats.ttest_ind(fear_days, greed_days)
    return {
        "fear_avg_pnl": round(fear_days.mean(), 4),
        "greed_avg_pnl": round(greed_days.mean(), 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "significant": p_val < 0.05,
    }
