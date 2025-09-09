import pandas as pd
import numpy as np

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["tx_id"]).copy()
    df = df[df["amount"] > -1000].copy()
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df = df.sort_values(["customer_id", "date"])
    df["roll_mean_7"] = df.groupby("customer_id")["amount"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["roll_std_7"]  = df.groupby("customer_id")["amount"].transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0.0)
    df["zscore_7"] = (df["amount"] - df["roll_mean_7"]) / df["roll_std_7"].replace(0, 1e-9)
    return df

def zscore_flags(df: pd.DataFrame, th=3.5) -> pd.Series:
    return (df["zscore_7"].abs() >= th) | (df["amount"] <= 0)
