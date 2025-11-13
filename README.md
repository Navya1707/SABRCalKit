# SABR Calibration Mini‑Suite
A Python mini-suite for calibration & analysis of the SABR volatility smile model_

## Overview  
**SABRCalKit** is a compact, yet powerful tool for calibrating the SABR model (Stochastic Alpha Beta Rho) volatility-smile model widely used in derivatives markets (caps/floors, swaptions, equity/FX options).  
It lets users:  
- load market volatility smile data (strikes vs implied volatilities)  
- fit the SABR parameters (α, β, ρ, ν / vol-vol) under the popular Hagan (2002) / shifted-lognormal specification  
- compute model-implied volatilities across strikes & maturities  
- compare model vs market and track calibration error  
- integrate into quant workflows (ETL / batch-calibration / dashboards)  

## Key Features  
- Lean, modular Python codebase (no heavy dependencies beyond numpy, scipy, pandas)  
- Support for both log-normal and shifted log-normal SABR frames  
- Scriptable CLI entry point (`scripts/run_calib.py`) for batch runs / output logs  
- Simple input data folder under `data/` for user-supplied market quotes  
- Easily extended to additional underlying types (equity/FX/caps) and surfaces  

## Why Use It?  
- Perform fast calibration of SABR parameters for a single maturity / slice, or across multiple slices  
- Generate model vol‐smile plots & error diagnostics (difference between market and model)  
- Ideal for quant research, risk analytics, prototype trading desk tooling, or teaching derivatives calibration  
- Lightweight compared to full production libraries — great for prototyping, academic work, and teaching  

## Installation  
```bash
git clone https://github.com/Navya1707/SABRCalKit.git
cd SABRCalKit  
pip install -r requirements.txt  
