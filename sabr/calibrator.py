from dataclasses import dataclass
import numpy as np, time
from typing import Optional, Tuple
from scipy.optimize import least_squares
from .hagan import hagan_iv
from .black import black_price, black_vega
@dataclass
class CalibResult:
    alpha: float; beta: float; rho: float; nu: float; success: bool; nfev: int
    rmse_iv: float; rmse_iv_bps: float; rmse_price_bps: float; runtime_ms: float
def _to_unconstrained(alpha: float, rho: float, nu: float) -> np.ndarray:
    return np.array([np.log(max(alpha, 1e-12)), np.arctanh(np.clip(rho, -0.9999, 0.9999)), np.log(max(nu, 1e-12))], dtype=float)
def _from_unconstrained(x: np.ndarray) -> Tuple[float, float, float]:
    a, r, v = x; return float(np.exp(a)), float(np.tanh(r)), float(np.exp(v))
def _initial_guess(F, Ks, iv_mkt, beta):
    atm_idx = np.argmin(np.abs(Ks - F)); atm_iv = float(iv_mkt[atm_idx])
    alpha0 = max(1e-4, atm_iv * (F ** (1.0 - beta))); rho0 = -0.2; nu0 = 0.5
    return alpha0, rho0, nu0
def calibrate_smile(F: float, T: float, Ks: np.ndarray, iv_mkt: np.ndarray, Df: float = 1.0,
                    beta: float = 0.5, vega_weighted: bool = True, multistart: int = 3,
                    random_state: Optional[int] = 42, price_side: str = "call",
                    max_nfev: int = 300) -> CalibResult:
    rng = np.random.default_rng(random_state); call = True if price_side.lower() == "call" else False
    if vega_weighted:
        vegas = np.array([black_vega(F, k, T, sigma) for k, sigma in zip(Ks, iv_mkt)], dtype=float)
        w = vegas / (np.sum(vegas) + 1e-12); w = np.maximum(w, 1e-6)
    else:
        w = np.ones_like(iv_mkt, dtype=float) / max(len(iv_mkt), 1)
    def objective(x):
        alpha, rho, nu = _from_unconstrained(x)
        model_ivs = np.array([hagan_iv(F, k, T, alpha, beta, rho, nu) for k in Ks], dtype=float)
        return np.sqrt(w) * (model_ivs - iv_mkt)
    best = None; t0 = time.time()
    alpha0, rho0, nu0 = _initial_guess(F, Ks, iv_mkt, beta)
    seeds = [_to_unconstrained(alpha0, rho0, nu0)]
    for _ in range(max(multistart - 1, 0)):
        a = alpha0 * float(np.exp(rng.normal(0, 0.2)))
        r = float(np.clip(rho0 + rng.normal(0, 0.15), -0.85, 0.85))
        v = nu0 * float(np.exp(rng.normal(0, 0.3)))
        seeds.append(_to_unconstrained(a, r, v))
    for s in seeds:
        res = least_squares(objective, s, method="trf", max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        alpha, rho, nu = _from_unconstrained(res.x)
        model_ivs = np.array([hagan_iv(F, k, T, alpha, beta, rho, nu) for k in Ks], dtype=float)
        iv_err = model_ivs - iv_mkt; rmse_iv = float(np.sqrt(np.mean(iv_err**2))); rmse_iv_bps = float(1e4 * rmse_iv)
        model_prices = np.array([black_price(F, k, T, s, call=call, Df=Df) for k, s in zip(Ks, model_ivs)], dtype=float)
        mkt_prices = np.array([black_price(F, k, T, s, call=call, Df=Df) for k, s in zip(Ks, iv_mkt)], dtype=float)
        rmse_price_bps = float(1e4 * np.sqrt(np.mean((model_prices - mkt_prices) ** 2)))
        out = (rmse_iv, {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu,
                         'success': bool(res.success), 'nfev': int(res.nfev),
                         'rmse_iv': rmse_iv, 'rmse_iv_bps': rmse_iv_bps,
                         'rmse_price_bps': rmse_price_bps})
        if (best is None) or (out[0] < best[0]): best = out
    t1 = time.time(); info = best[1]
    return CalibResult(alpha=info['alpha'], beta=info['beta'], rho=info['rho'], nu=info['nu'],
                       success=info['success'], nfev=info['nfev'],
                       rmse_iv=info['rmse_iv'], rmse_iv_bps=info['rmse_iv_bps'],
                       rmse_price_bps=info['rmse_price_bps'],
                       runtime_ms=float((t1 - t0) * 1000.0))
