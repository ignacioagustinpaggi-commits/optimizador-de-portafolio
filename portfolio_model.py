"""
portfolio_model.py
==================
Teoría Moderna de Portafolios — Markowitz (1952).
Optimización SLSQP, Monte Carlo y frontera eficiente.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

DIAS_ANIO = 252


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES BASE
# ─────────────────────────────────────────────────────────────────────────────

def port_retorno(pesos: np.ndarray, retornos: pd.DataFrame) -> float:
    return float(np.dot(pesos, retornos.mean()) * DIAS_ANIO)


def port_volatilidad(pesos: np.ndarray, retornos: pd.DataFrame) -> float:
    cov = retornos.cov() * DIAS_ANIO
    return float(np.sqrt(max(np.dot(pesos.T, np.dot(cov.values, pesos)), 0)))


def port_sharpe(pesos: np.ndarray, retornos: pd.DataFrame, rf: float = 0.0) -> float:
    vol = port_volatilidad(pesos, retornos)
    return float((port_retorno(pesos, retornos) - rf) / vol) if vol > 0 else 0.0


def portafolio_equiponderado(retornos: pd.DataFrame) -> np.ndarray:
    n = len(retornos.columns)
    return np.array([1 / n] * n)


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS COMPLETAS DE UN PORTAFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def metricas_portafolio(
    pesos: np.ndarray,
    retornos: pd.DataFrame,
    precios: pd.DataFrame,
    rf: float = 0.0,
) -> dict:
    """Calcula retorno, volatilidad, Sharpe, drawdown, VaR, evolución."""
    if pesos is None:
        return {}
    w = np.array(pesos) / np.sum(pesos)
    tickers = retornos.columns.tolist()

    ret = port_retorno(w, retornos)
    vol = port_volatilidad(w, retornos)
    shr = (ret - rf) / vol if vol > 0 else 0.0

    ret_d    = (retornos * w).sum(axis=1)
    evol     = (1 + ret_d).cumprod() * 100
    mdd      = float(((evol - evol.cummax()) / evol.cummax()).min())
    var_95   = float(np.percentile(ret_d.dropna(), 5))

    ret_neg  = ret_d[ret_d < 0]
    down_vol = float(ret_neg.std() * np.sqrt(DIAS_ANIO)) if len(ret_neg) > 0 else 0.0
    sortino  = float((ret - rf) / down_vol) if down_vol > 0 else 0.0

    return {
        "pesos":         dict(zip(tickers, w.round(6))),
        "retorno_anual": round(ret, 6),
        "volatilidad":   round(vol, 6),
        "sharpe":        round(shr, 4),
        "sortino":       round(sortino, 4),
        "max_drawdown":  round(mdd, 4),
        "var_95_diario": round(var_95, 6),
        "evolucion":     evol,
        "ret_diarios":   ret_d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZACIÓN SLSQP
# ─────────────────────────────────────────────────────────────────────────────

def _bounds(n, peso_min=0.0, peso_max=1.0):
    return tuple((peso_min, peso_max) for _ in range(n))


def _constraints(n):
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]


def optimizar_max_sharpe(
    retornos: pd.DataFrame,
    rf: float = 0.0,
    peso_min: float = 0.0,
    peso_max: float = 1.0,
) -> Optional[np.ndarray]:
    n  = len(retornos.columns)
    w0 = np.array([1 / n] * n)

    def neg_sharpe(w):
        r = port_retorno(w, retornos)
        v = port_volatilidad(w, retornos)
        return -(r - rf) / v if v > 0 else 0.0

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=_bounds(n, peso_min, peso_max),
                   constraints=_constraints(n),
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else None


def optimizar_min_varianza(
    retornos: pd.DataFrame,
    peso_min: float = 0.0,
    peso_max: float = 1.0,
) -> Optional[np.ndarray]:
    n  = len(retornos.columns)
    w0 = np.array([1 / n] * n)

    res = minimize(lambda w: port_volatilidad(w, retornos), w0,
                   method="SLSQP",
                   bounds=_bounds(n, peso_min, peso_max),
                   constraints=_constraints(n),
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else None


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def simulacion_monte_carlo(
    retornos: pd.DataFrame,
    n_sim: int = 5_000,
    rf: float = 0.0,
    semilla: int = 42,
) -> pd.DataFrame:
    np.random.seed(semilla)
    n = len(retornos.columns)
    tickers = retornos.columns.tolist()
    filas = []

    for _ in range(n_sim):
        w = np.random.uniform(0, 1, n)
        w /= w.sum()
        r  = port_retorno(w, retornos)
        v  = port_volatilidad(w, retornos)
        s  = (r - rf) / v if v > 0 else 0.0
        fila = {"retorno": r, "volatilidad": v, "sharpe": s}
        fila.update({f"w_{t}": wi for t, wi in zip(tickers, w)})
        filas.append(fila)

    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# FRONTERA EFICIENTE
# ─────────────────────────────────────────────────────────────────────────────

def calcular_frontera_eficiente(
    retornos: pd.DataFrame,
    n_puntos: int = 80,
    peso_min: float = 0.0,
    peso_max: float = 1.0,
) -> pd.DataFrame:
    n  = len(retornos.columns)
    w0 = np.array([1 / n] * n)

    ret_min = retornos.mean().min() * DIAS_ANIO
    ret_max = retornos.mean().max() * DIAS_ANIO
    targets = np.linspace(ret_min, ret_max, n_puntos)

    puntos = []
    for r_obj in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, r=r_obj: port_retorno(w, retornos) - r},
        ]
        res = minimize(lambda w: port_volatilidad(w, retornos), w0,
                       method="SLSQP",
                       bounds=_bounds(n, peso_min, peso_max),
                       constraints=cons,
                       options={"ftol": 1e-10, "maxiter": 500})
        if res.success:
            v = port_volatilidad(res.x, retornos)
            puntos.append({"retorno_objetivo": r_obj, "volatilidad": v,
                           "sharpe": r_obj / v if v > 0 else 0})

    return pd.DataFrame(puntos)


# ─────────────────────────────────────────────────────────────────────────────
# TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────────

def tabla_comparativa_carteras(
    retornos: pd.DataFrame,
    precios: pd.DataFrame,
    w_ms: Optional[np.ndarray],
    w_mv: Optional[np.ndarray],
    rf: float = 0.0,
) -> pd.DataFrame:
    tickers = retornos.columns.tolist()
    w_eq    = portafolio_equiponderado(retornos)
    carteras = {"Equiponderado": w_eq}
    if w_ms is not None: carteras["Máximo Sharpe"]   = w_ms
    if w_mv is not None: carteras["Mínima Varianza"] = w_mv

    filas = []
    for nombre, w in carteras.items():
        r   = port_retorno(w, retornos)
        v   = port_volatilidad(w, retornos)
        s   = (r - rf) / v if v > 0 else 0.0
        rd  = (retornos * w).sum(axis=1)
        ev  = (1 + rd).cumprod() * 100
        mdd = float(((ev - ev.cummax()) / ev.cummax()).min())
        var = float(np.percentile(rd.dropna(), 5))
        fila = {"Cartera": nombre, "Retorno Anual": r, "Volatilidad": v,
                "Sharpe": s, "Max Drawdown": mdd, "VaR 95%": var}
        for t, wi in zip(tickers, w):
            fila[f"w_{t}"] = round(wi, 4)
        filas.append(fila)

    return pd.DataFrame(filas).set_index("Cartera")
