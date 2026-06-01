import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_COLORS = {
    "Extreme Fear": "#d62728",
    "Fear": "#ff7f0e",
    "Neutral": "#7f7f7f",
    "Greed": "#2ca02c",
    "Extreme Greed": "#1a7a1a",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def plot_sentiment_distribution(fg_df, save_path="outputs/figures/01_sentiment_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = fg_df["sentiment"].value_counts().reindex(SENTIMENT_ORDER)
    colors = [SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]
    axes[0].bar(SENTIMENT_ORDER, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Sentiment Classification Count (2022–2025)")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Days")
    axes[0].tick_params(axis="x", rotation=15)

    fg_df_plot = fg_df.copy()
    fg_df_plot["date"] = pd.to_datetime(fg_df_plot["date"])
    axes[1].plot(fg_df_plot["date"], fg_df_plot["fg_value"], color="#444", linewidth=0.8, alpha=0.8)
    axes[1].axhline(25, color="#d62728", linestyle="--", alpha=0.5, label="Extreme Fear <25")
    axes[1].axhline(45, color="#ff7f0e", linestyle="--", alpha=0.5, label="Fear <45")
    axes[1].axhline(55, color="#2ca02c", linestyle="--", alpha=0.5, label="Greed >55")
    axes[1].axhline(75, color="#1a7a1a", linestyle="--", alpha=0.5, label="Extreme Greed >75")
    axes[1].set_title("Fear & Greed Index Over Time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Index Value")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pnl_by_sentiment(sent_summary, save_path="outputs/figures/02_pnl_by_sentiment.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sent_summary = sent_summary.set_index("sentiment").reindex(SENTIMENT_ORDER).reset_index()
    colors = [SENTIMENT_COLORS[s] for s in sent_summary["sentiment"]]

    axes[0].bar(sent_summary["sentiment"], sent_summary["avg_pnl"], color=colors, edgecolor="white")
    axes[0].set_title("Average PnL per Trade by Sentiment")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Avg PnL (USD)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].axhline(0, color="black", linewidth=0.8)

    axes[1].bar(sent_summary["sentiment"], sent_summary["win_rate"] * 100, color=colors, edgecolor="white")
    axes[1].set_title("Win Rate % by Sentiment")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].axhline(50, color="black", linewidth=0.8, linestyle="--")

    axes[2].bar(sent_summary["sentiment"], sent_summary["avg_leverage"], color=colors, edgecolor="white")
    axes[2].set_title("Avg Leverage Used by Sentiment")
    axes[2].set_xlabel("Sentiment")
    axes[2].set_ylabel("Average Leverage")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_long_short_by_sentiment(df, save_path="outputs/figures/03_long_short_sentiment.png"):
    side_sent = df.groupby(["sentiment", "side"], observed=True).size().unstack(fill_value=0)
    side_sent = side_sent.reindex(SENTIMENT_ORDER)
    side_sent_pct = side_sent.div(side_sent.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(side_sent_pct))
    ax.bar(side_sent_pct.index, side_sent_pct["LONG"], label="LONG", color="#2ca02c", edgecolor="white")
    ax.bar(side_sent_pct.index, side_sent_pct["SHORT"], bottom=side_sent_pct["LONG"],
           label="SHORT", color="#d62728", edgecolor="white")
    ax.axhline(50, color="black", linewidth=1, linestyle="--")
    ax.set_title("Long vs Short Trade Distribution by Sentiment")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Sentiment")
    ax.legend()
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_rolling_pnl(daily_df, save_path="outputs/figures/04_rolling_pnl.png"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(daily_df["date"], daily_df["rolling_pnl"], color="#1f77b4", linewidth=1.5)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("30-Day Rolling Average Daily PnL")
    axes[0].set_ylabel("Avg PnL (USD)")

    axes[1].plot(daily_df["date"], daily_df["rolling_win_rate"] * 100, color="#ff7f0e", linewidth=1.5)
    axes[1].axhline(50, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("30-Day Rolling Win Rate")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_vs_bottom(comparison_df, save_path="outputs/figures/05_top_vs_bottom_traders.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["win_rate", "avg_leverage", "long_pct"]
    titles = ["Win Rate", "Avg Leverage", "Long Position %"]
    ylabels = ["Win Rate", "Leverage", "Long %"]

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        for group, color in [("Top 10", "#2ca02c"), ("Bottom 10", "#d62728")]:
            subset = comparison_df[comparison_df["group"] == group]
            subset = subset.set_index("sentiment").reindex(SENTIMENT_ORDER).reset_index()
            vals = subset[metric].values
            if metric == "win_rate":
                vals = vals * 100
            elif metric == "long_pct":
                vals = vals * 100
            ax.plot(SENTIMENT_ORDER, vals, marker="o", label=group, color=color, linewidth=2)
        ax.set_title(f"{title} – Top vs Bottom Traders")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_leverage_heatmap(lev_df, save_path="outputs/figures/06_leverage_heatmap.png"):
    pivot = lev_df.pivot_table(index="lev_bucket", columns="sentiment", values="win_rate", observed=True)
    pivot = pivot.reindex(columns=SENTIMENT_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Win Rate"})
    ax.set_title("Win Rate by Leverage Bucket × Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Leverage Bucket")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_symbol_pnl(sym_df, save_path="outputs/figures/07_symbol_pnl.png"):
    sym_overall = sym_df.groupby("symbol")["avg_pnl"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in sym_overall.values]
    ax.barh(sym_overall.index, sym_overall.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Average PnL per Trade by Symbol")
    ax.set_xlabel("Avg PnL (USD)")
    ax.set_ylabel("Symbol")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_liquidation_rate(sent_summary, save_path="outputs/figures/08_liquidation_rate.png"):
    sent_summary = sent_summary.copy().set_index("sentiment").reindex(SENTIMENT_ORDER).reset_index()
    colors = [SENTIMENT_COLORS[s] for s in sent_summary["sentiment"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sent_summary["sentiment"], sent_summary["liq_pct"] * 100, color=colors, edgecolor="white")
    ax.set_title("Liquidation Rate by Market Sentiment")
    ax.set_ylabel("Liquidation Rate (%)")
    ax.set_xlabel("Sentiment")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pnl_distribution(df, save_path="outputs/figures/09_pnl_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    clipped = df["closedPnL"].clip(-200, 200)
    axes[0].hist(clipped, bins=80, color="#1f77b4", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_title("PnL Distribution (clipped at ±200)")
    axes[0].set_xlabel("PnL (USD)")
    axes[0].set_ylabel("Count")

    for sentiment in SENTIMENT_ORDER:
        subset = df[df["sentiment"] == sentiment]["closedPnL"].clip(-200, 200)
        axes[1].hist(subset, bins=40, alpha=0.5, label=sentiment,
                     color=SENTIMENT_COLORS[sentiment], density=True)
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("PnL Distribution by Sentiment")
    axes[1].set_xlabel("PnL (USD)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_hourly_pnl(df, save_path="outputs/figures/10_hourly_pnl.png"):
    hourly = df.groupby("hour").agg(
        avg_pnl=("closedPnL", "mean"),
        win_rate=("is_win", "mean"),
        trades=("closedPnL", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(hourly["hour"], hourly["avg_pnl"], color="#1f77b4", edgecolor="white")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Average PnL by Hour of Day (UTC)")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Avg PnL (USD)")

    axes[1].plot(hourly["hour"], hourly["win_rate"] * 100, marker="o", color="#ff7f0e", linewidth=2)
    axes[1].axhline(50, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Win Rate by Hour of Day (UTC)")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Win Rate (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
