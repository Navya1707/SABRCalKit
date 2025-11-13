import numpy as np
def _atm_hagan_iv(F, T, alpha, beta, rho, nu):
    F_pow = F**(1.0 - beta) if F > 0 else 1e-12
    term1 = alpha / F_pow
    term2 = (
        ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2) / (F ** (2.0 - 2.0 * beta) if F > 0 else 1e12)
        + (rho * beta * nu * alpha) / (4.0 * (F_pow if F_pow > 0 else 1e-12))
        + ((2.0 - 3.0 * rho ** 2) * (nu ** 2) / 24.0)
    ) * T
    return term1 * (1.0 + term2)
def hagan_iv(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    eps = 1e-12; K = float(K); F = float(F); T = float(T)
    alpha = max(alpha, 1e-12); nu = max(nu, 1e-12)
    rho = max(min(rho, 0.9999), -0.9999); beta = max(min(beta, 0.9999), 1e-6)
    if T <= 0 or K <= 0 or F <= 0: return 0.0
    if abs(K - F) / F < 1e-6: return _atm_hagan_iv(F, T, alpha, beta, rho, nu)
    logFK = np.log(F / K); one_m_beta = 1.0 - beta; FK_beta = (F * K) ** (one_m_beta / 2.0)
    z = (nu / alpha) * FK_beta * logFK
    sqrt_term = np.sqrt(max(1.0 - 2.0 * rho * z + z * z, 1e-18))
    chi = np.log((sqrt_term + z - rho) / (1.0 - rho))
    z_over_chi = 1.0 if abs(chi) < eps else z / chi
    logFK2 = logFK * logFK; logFK4 = logFK2 * logFK2
    D = FK_beta * (1.0 + (one_m_beta**2 / 24.0) * logFK2 + (one_m_beta**4 / 1920.0) * logFK4)
    if D <= 0.0: return 0.0
    A = alpha / D
    Tterm = ( ((one_m_beta**2) / 24.0) * (alpha**2) / (FK_beta**2) + (rho * beta * nu * alpha) / (4.0 * FK_beta) + ((2.0 - 3.0 * (rho**2)) / 24.0) * (nu**2) ) * T
    return float(A * z_over_chi * (1.0 + Tterm))
