import pandas as pd


def load_clean(path="../data/telco_churn.csv"):
    """Load the Telco churn data and apply the cleaning decided in 01_eda."""
    df = pd.read_csv(path)
    # 11 blank TotalCharges rows are brand-new customers (tenure = 0), so 0 is the correct value
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])
    return df
