import pandas as pd


def load_fear_greed(path="data/raw/fear_greed_index.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"value": "fg_value", "classification": "sentiment"})
    df = df[["date", "fg_value", "sentiment"]].sort_values("date").reset_index(drop=True)
    return df


def load_trades(path="data/raw/hyperliquid_trades.csv"):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = pd.to_datetime(df["date"])
    df["closedPnL"] = pd.to_numeric(df["closedPnL"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["leverage"] = pd.to_numeric(df["leverage"], errors="coerce")
    df["execution_price"] = pd.to_numeric(df["execution_price"], errors="coerce")
    return df


def merge_datasets(trades, fear_greed):
    merged = trades.merge(fear_greed, on="date", how="left")
    return merged


def save_processed(df, path="data/processed/merged_data.csv"):
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
