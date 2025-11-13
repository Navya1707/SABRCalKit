import numpy as np
def _n_cdf(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _n_pdf(x: float) -> float:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)
def black_price(F: float, K: float, T: float, sigma: float, call: bool = True, Df: float = 1.0) -> float:
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        intrinsic = max(F-K, 0.0) if call else max(K-F, 0.0); return Df * intrinsic
    vol_sqrtT = sigma * np.sqrt(T); d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrtT; d2 = d1 - vol_sqrtT
    price = F * _n_cdf(d1) - K * _n_cdf(d2) if call else K * _n_cdf(-d2) - F * _n_cdf(-d1)
    return float(Df * price)
def black_vega(F: float, K: float, T: float, sigma: float, Df: float = 1.0) -> float:
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0: return 0.0
    d1 = (np.log(F/K) + 0.5*sigma*sigma*T) / (sigma*np.sqrt(T))
    return float(Df * F * np.sqrt(T) * _n_pdf(d1))
