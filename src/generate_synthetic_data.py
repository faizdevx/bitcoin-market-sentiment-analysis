import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

fg = pd.read_csv("data/raw/fear_greed_index.csv")
fg["date"] = pd.to_datetime(fg["date"])
fg = fg[(fg["date"] >= "2022-01-01") & (fg["date"] <= "2025-04-30")].reset_index(drop=True)

date_sentiment = dict(zip(fg["date"], fg["classification"]))
date_value = dict(zip(fg["date"], fg["value"]))

symbols = ["BTC", "ETH", "SOL", "AVAX", "ARB", "OP", "DOGE", "MATIC", "LINK", "APT"]
n_accounts = 80
accounts = [f"0x{i:040x}" for i in range(1, n_accounts + 1)]

leverage_map = {
    "Extreme Fear": (1, 5),
    "Fear": (1, 10),
    "Neutral": (2, 15),
    "Greed": (3, 20),
    "Extreme Greed": (5, 25),
}

long_prob_map = {
    "Extreme Fear": 0.30,
    "Fear": 0.40,
    "Neutral": 0.50,
    "Greed": 0.65,
    "Extreme Greed": 0.78,
}

symbol_price = {
    "BTC": 35000,
    "ETH": 2000,
    "SOL": 80,
    "AVAX": 25,
    "ARB": 1.2,
    "OP": 1.5,
    "DOGE": 0.08,
    "MATIC": 0.9,
    "LINK": 12,
    "APT": 8,
}

rows = []
dates = pd.date_range("2022-01-01", "2025-04-30", freq="D")

for date in dates:
    sentiment = date_sentiment.get(pd.Timestamp(date), "Neutral")
    fg_val = date_value.get(pd.Timestamp(date), 50)

    trades_today = np.random.poisson(lam=35)

    for _ in range(trades_today):
        account = np.random.choice(accounts)
        symbol = np.random.choice(symbols, p=[0.30, 0.25, 0.15, 0.06, 0.05, 0.04, 0.05, 0.04, 0.03, 0.03])
        base_price = symbol_price[symbol]
        exec_price = base_price * np.random.uniform(0.92, 1.08)

        lev_low, lev_high = leverage_map[sentiment]
        leverage = np.random.randint(lev_low, lev_high + 1)

        long_prob = long_prob_map[sentiment]
        side = "LONG" if np.random.random() < long_prob else "SHORT"

        size = np.random.lognormal(mean=2.5, sigma=1.2)
        size = round(size, 4)

        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        trade_time = date + timedelta(hours=int(hour), minutes=int(minute))

        start_position = np.random.choice(["0", str(round(np.random.uniform(-5, 5), 4))], p=[0.4, 0.6])

        event = np.random.choice(["ORDER_FILLED", "LIQUIDATION", "TAKE_PROFIT", "STOP_LOSS"],
                                  p=[0.72, 0.04, 0.14, 0.10])

        win_prob_base = 0.48
        sentiment_modifier = (fg_val - 50) / 200
        if side == "LONG":
            win_prob = win_prob_base + sentiment_modifier
        else:
            win_prob = win_prob_base - sentiment_modifier

        win_prob = np.clip(win_prob, 0.30, 0.70)

        if event == "LIQUIDATION":
            pnl = -abs(np.random.lognormal(mean=2.0, sigma=0.8)) * size * 0.1
        elif np.random.random() < win_prob:
            pnl = abs(np.random.lognormal(mean=1.2, sigma=0.9)) * size * 0.05
        else:
            pnl = -abs(np.random.lognormal(mean=1.0, sigma=0.7)) * size * 0.04

        pnl = round(pnl, 4)

        rows.append({
            "account": account,
            "symbol": symbol,
            "execution_price": round(exec_price, 4),
            "size": size,
            "side": side,
            "time": trade_time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_position": start_position,
            "event": event,
            "closedPnL": pnl,
            "leverage": leverage,
            "date": date.strftime("%Y-%m-%d"),
        })

df = pd.DataFrame(rows)
df.to_csv("data/raw/hyperliquid_trades.csv", index=False)
print(f"Generated {len(df)} trades across {df['account'].nunique()} accounts")
print(df.head())
