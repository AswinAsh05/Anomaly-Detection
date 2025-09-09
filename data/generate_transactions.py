import argparse
import numpy as np
import pandas as pd

CATEGORIES = ["Electronics", "Home", "Beauty", "Toys", "Grocery"]

def generate(start: str, end: str, seed: int, n_customers: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    rows = []
    for d in dates:
        base_n = rng.poisson(lam=300)
        for _ in range(base_n):
            tx_id = f"{d.strftime('%Y%m%d')}-{rng.integers(1_000_000, 9_999_999)}"
            cust = int(rng.integers(1, n_customers+1))
            cat = rng.choice(CATEGORIES, p=[0.25, 0.22, 0.2, 0.18, 0.15])
            mu_sigma = {
                "Electronics": (4.5, 0.6),
                "Home": (3.5, 0.5),
                "Beauty": (3.0, 0.4),
                "Toys": (3.1, 0.45),
                "Grocery": (2.2, 0.3),
            }[cat]
            amount = float(np.round(np.exp(rng.normal(*mu_sigma)), 2))
            rows.append([tx_id, d.date(), cust, cat, amount])

        if rng.random() < 0.05:
            for _ in range(rng.integers(30, 80)):
                tx_id = f"{d.strftime('%Y%m%d')}-{rng.integers(1_000_000, 9_999_999)}"
                cust = int(rng.integers(1, n_customers+1))
                cat = "Grocery"
                amount = float(np.round(np.exp(rng.normal(1.9, 0.25)), 2))
                rows.append([tx_id, d.date(), cust, cat, amount])

        if rng.random() < 0.03:
            for _ in range(rng.integers(3, 10)):
                tx_id = f"{d.strftime('%Y%m%d')}-{rng.integers(1_000_000, 9_999_999)}"
                cust = int(rng.integers(1, n_customers+1))
                cat = "Electronics"
                amount = float(np.round(np.random.uniform(1500, 5000), 2))
                rows.append([tx_id, d.date(), cust, cat, amount])

        if rng.random() < 0.02:
            for _ in range(rng.integers(2, 6)):
                tx_id = f"{d.strftime('%Y%m%d')}-{rng.integers(1_000_000, 9_999_999)}"
                cust = int(rng.integers(1, n_customers+1))
                cat = rng.choice(CATEGORIES)
                amount = float(rng.choice([0.0, -rng.uniform(1, 50)]))
                rows.append([tx_id, d.date(), cust, cat, amount])

    df = pd.DataFrame(rows, columns=["tx_id", "date", "customer_id", "category", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-customers", type=int, default=500)
    ap.add_argument("--out", default="data/transactions.csv")
    args = ap.parse_args()

    df = generate(args.start, args.end, args.seed, args.n_customers)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(df):,} rows")

if __name__ == "__main__":
    main()
