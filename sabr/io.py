import pandas as pd
def read_smiles(csv_path: str) -> pd.DataFrame:
    required = {"expiry", "forward", "discount", "strike", "market_iv", "option_type"}
    df = pd.read_csv(csv_path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    df["option_type"] = df["option_type"].astype(str).str.upper().map({"C":"C","CALL":"C","P":"P","PUT":"P"})
    if "tenor" not in df.columns: df["tenor"] = ""
    if "notional" not in df.columns: df["notional"] = 1.0
    return df
def group_smiles(df: pd.DataFrame):
    for (expiry, tenor), g in df.groupby(["expiry", "tenor"], sort=True):
        g = g.sort_values("strike").reset_index(drop=True)
        yield (expiry, tenor), g
