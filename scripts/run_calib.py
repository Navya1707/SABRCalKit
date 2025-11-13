#!/usr/bin/env python3
import argparse
import numpy as np, pandas as pd
from pathlib import Path
from sabr.io import read_smiles, group_smiles
from sabr.calibrator import calibrate_smile
def main():
    ap = argparse.ArgumentParser(description="Calibrate SABR (Hagan IV) per smile from CSV.")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--outdir", default="results", help="Output directory")
    ap.add_argument("--beta", type=float, default=0.5, help="Fixed beta in (0,1]")
    ap.add_argument("--vega-weighted", action="store_true", help="Use vega-weighted LS")
    ap.add_argument("--multistart", type=int, default=3, help="Number of random starts")
    ap.add_argument("--max-nfev", type=int, default=300, help="Max function evaluations")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = read_smiles(args.inp)
    rows = []
    min_expiry = df["expiry"].min()
    for (expiry, tenor), g in group_smiles(df):
        F = float(g["forward"].iloc[0]); Df = float(g["discount"].iloc[0])
        T = max((expiry - min_expiry).days / 365.0, 0.25)
        ot = g["option_type"].iloc[0]
        Ks = g["strike"].to_numpy(float); ivs = g["market_iv"].to_numpy(float)
        res = calibrate_smile(F, T, Ks, ivs, Df=Df, beta=args.beta,
                              vega_weighted=args.vega_weighted, multistart=args.multistart,
                              random_state=args.seed, price_side="call" if ot=="C" else "put",
                              max_nfev=args.max_nfev)
        rows.append({
            "expiry": expiry.strftime("%Y-%m-%d"),
            "tenor": tenor,
            "alpha": res.alpha, "beta": res.beta, "rho": res.rho, "nu": res.nu,
            "success": res.success, "nfev": res.nfev,
            "rmse_iv": res.rmse_iv, "rmse_iv_bps": res.rmse_iv_bps,
            "rmse_price_bps": res.rmse_price_bps, "runtime_ms": res.runtime_ms
        })
    out_csv = outdir / "calibration_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
if __name__ == "__main__":
    main()
