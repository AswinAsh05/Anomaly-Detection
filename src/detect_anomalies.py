import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import clean, feature_engineer, zscore_flags

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def run_models(df: pd.DataFrame, contamination: float = 0.02, random_state: int = 42):
    feats = df[["amount", "dayofweek", "month", "zscore_7"]].to_numpy()

    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state)
    iso_labels = iso.fit_predict(feats)
    iso_score = -iso.decision_function(feats)

    lof = LocalOutlierFactor(n_neighbors=35, contamination=contamination)
    lof_labels = lof.fit_predict(feats)
    lof_score = -lof.negative_outlier_factor_

    z_flags = zscore_flags(df).astype(int)
    return iso_labels, iso_score, lof_labels, lof_score, z_flags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to transactions.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--contamination", type=float, default=0.02)
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    df = load(args.input)
    df = clean(df)
    df = feature_engineer(df)

    iso_labels, iso_score, lof_labels, lof_score, z_flags = run_models(df, contamination=args.contamination)

    out = df.copy()
    out["iso_label"] = (iso_labels == -1).astype(int)
    out["lof_label"] = (lof_labels == -1).astype(int)
    out["zscore_label"] = z_flags
    out["votes"] = out[["iso_label", "lof_label", "zscore_label"]].sum(axis=1)
    s_iso = (iso_score - iso_score.min()) / max(1e-9, (iso_score.max() - iso_score.min()))
    s_lof = (lof_score - lof_score.min()) / max(1e-9, (lof_score.max() - lof_score.min()))
    out["severity"] = (s_iso + s_lof) / 2.0

    anomalies = out[out["votes"] >= 2].sort_values(["severity"], ascending=False)
    anomalies.to_csv(os.path.join(args.outdir, "anomalies.csv"), index=False)

    fig1, ax1 = plt.subplots(figsize=(12,4))
    df_plot = out.sample(n=min(20000, len(out)), random_state=42).sort_values("date")
    ax1.plot(df_plot["date"], df_plot["amount"])
    ax1.set_title("Transaction Amount over Time (sample)")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Amount")
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.outdir, "fig_amount_time.png"), dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.hist(out["amount"], bins=60)
    ax2.set_title("Amount Distribution")
    ax2.set_xlabel("Amount"); ax2.set_ylabel("Count")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "fig_amount_hist.png"), dpi=160)
    plt.close(fig2)

    print("[OK] Anomaly detection complete.")
    print(f"Flagged anomalies: {len(anomalies):,}")
    print(f"Outputs saved to: {args.outdir}")

if __name__ == "__main__":
    main()
