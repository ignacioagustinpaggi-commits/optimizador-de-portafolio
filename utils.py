"""
utils.py
========
Métricas financieras, VaR, drawdown, rankings y FODA.
"""

import numpy as np
import pandas as pd
from typing import Optional

DIAS_ANIO = 252


def retorno_anualizado(ret: pd.Series) -> float:
    return float(ret.mean() * DIAS_ANIO)


def volatilidad_anualizada(ret: pd.Series) -> float:
    return float(ret.std() * np.sqrt(DIAS_ANIO))


def sharpe_ratio(ret: pd.Series, rf: float = 0.0) -> float:
    r = retorno_anualizado(ret)
    v = volatilidad_anualizada(ret)
    return float((r - rf) / v) if v > 0 else 0.0


def sortino_ratio(ret: pd.Series, rf: float = 0.0) -> float:
    r    = retorno_anualizado(ret)
    neg  = ret[ret < 0]
    dv   = float(neg.std() * np.sqrt(DIAS_ANIO)) if len(neg) > 0 else 0.0
    return float((r - rf) / dv) if dv > 0 else 0.0


def retorno_acumulado(precios: pd.Series) -> float:
    return float((precios.iloc[-1] / precios.iloc[0]) - 1)


def max_drawdown(precios: pd.Series) -> float:
    rm = precios.cummax()
    return float(((precios - rm) / rm).min())


def var_historico(ret: pd.Series, nivel: float = 0.95) -> float:
    return float(np.percentile(ret.dropna(), (1 - nivel) * 100))


def cvar_historico(ret: pd.Series, nivel: float = 0.95) -> float:
    v = var_historico(ret, nivel)
    cola = ret[ret <= v]
    return float(cola.mean()) if not cola.empty else v


def calmar_ratio(precios: pd.Series) -> float:
    mdd = abs(max_drawdown(precios))
    return float(retorno_acumulado(precios) / mdd) if mdd > 0 else 0.0


def tabla_metricas(
    precios: pd.DataFrame,
    retornos: pd.DataFrame,
    rf: float = 0.0,
) -> pd.DataFrame:
    filas = []
    for t in retornos.columns:
        rs = retornos[t].dropna()
        ps = precios[t].dropna()
        filas.append({
            "Ticker":              t,
            "Precio Inicial":      round(ps.iloc[0], 2),
            "Precio Final":        round(ps.iloc[-1], 2),
            "Retorno Acumulado":   retorno_acumulado(ps),
            "Retorno Anualizado":  retorno_anualizado(rs),
            "Volatilidad Anual":   volatilidad_anualizada(rs),
            "Sharpe Ratio":        round(sharpe_ratio(rs, rf), 3),
            "Sortino Ratio":       round(sortino_ratio(rs, rf), 3),
            "Calmar Ratio":        round(calmar_ratio(ps), 3),
            "Max Drawdown":        max_drawdown(ps),
            "VaR 95% (diario)":    var_historico(rs),
            "CVaR 95% (diario)":   cvar_historico(rs),
            "Mejor Día":           float(rs.max()),
            "Peor Día":            float(rs.min()),
            "Fecha Mejor Día":     str(rs.idxmax().date()) if hasattr(rs.idxmax(), 'date') else str(rs.idxmax()),
            "Fecha Peor Día":      str(rs.idxmin().date()) if hasattr(rs.idxmin(), 'date') else str(rs.idxmin()),
        })
    return pd.DataFrame(filas).set_index("Ticker")


def generar_foda(ticker: str, metricas: pd.DataFrame) -> dict:
    if ticker not in metricas.index:
        return {"fortalezas": [], "oportunidades": [], "debilidades": [], "amenazas": []}

    row  = metricas.loc[ticker]
    ret  = row.get("Retorno Anualizado", 0)
    vol  = row.get("Volatilidad Anual", 0)
    shr  = row.get("Sharpe Ratio", 0)
    mdd  = row.get("Max Drawdown", 0)
    var  = row.get("VaR 95% (diario)", 0)

    F, O, D, A = [], [], [], []

    if ret > 0.15:  F.append(f"Alto retorno anualizado ({ret:.1%})")
    elif ret > 0:   F.append(f"Retorno positivo en el período ({ret:.1%})")
    if shr > 1.0:   F.append(f"Excelente Sharpe Ratio ({shr:.2f})")
    elif shr > 0.5: F.append(f"Sharpe Ratio aceptable ({shr:.2f})")
    if vol < 0.20:  F.append(f"Baja volatilidad ({vol:.1%}), perfil defensivo")
    if abs(mdd) < 0.15: F.append(f"Drawdown máximo contenido ({mdd:.1%})")
    if not F:       F.append("Activo listado en mercado regulado internacional")

    if vol > 0.30:  O.append("Alta volatilidad genera oportunidades de entrada en valles")
    O.append("Potencial de diversificación dentro del portafolio")
    O.append("Acceso a sectores o geografías complementarios")

    if vol > 0.40:  D.append(f"Alta volatilidad anualizada ({vol:.1%})")
    elif vol > 0.25:D.append(f"Volatilidad moderada-alta ({vol:.1%})")
    if shr < 0.3:   D.append(f"Bajo Sharpe Ratio ({shr:.2f}): retorno no compensa el riesgo")
    if ret < 0:     D.append(f"Retorno negativo en el período analizado ({ret:.1%})")
    if abs(mdd) > 0.30: D.append(f"Drawdown máximo elevado ({mdd:.1%})")
    if abs(var) > 0.03: D.append(f"VaR diario al 95%: {var:.2%} (pérdida potencial significativa)")
    if not D:       D.append("Sin debilidades destacadas en el período analizado")

    A.append("El desempeño pasado no garantiza rendimientos futuros")
    A.append("Cambios en tasas de interés internacionales afectan la valuación")
    A.append("Riesgo cambiario para activos en moneda distinta al ARS")
    if vol > 0.35:  A.append("Alta sensibilidad a eventos macro y resultados corporativos")

    return {"fortalezas": F, "oportunidades": O, "debilidades": D, "amenazas": A}


def comparacion_grupos(
    metricas: pd.DataFrame,
    tickers_arg: list,
    tickers_int: list,
    retornos: pd.DataFrame,
) -> dict:
    def stats(tg):
        g = [t for t in tg if t in metricas.index]
        if not g: return {}
        sub  = metricas.loc[g]
        rets = retornos[[t for t in g if t in retornos.columns]]
        cv   = rets.corr().values if len(g) > 1 else np.array([[1.0]])
        mask = ~np.eye(cv.shape[0], dtype=bool)
        cp   = float(cv[mask].mean()) if mask.any() else float("nan")
        return {
            "n": len(g),
            "retorno_prom":     float(sub["Retorno Anualizado"].mean()),
            "vol_prom":         float(sub["Volatilidad Anual"].mean()),
            "sharpe_prom":      float(sub["Sharpe Ratio"].mean()),
            "mdd_prom":         float(sub["Max Drawdown"].mean()),
            "corr_interna":     cp,
        }
    return {"arg": stats(tickers_arg), "int": stats(tickers_int)}


def formatear_pct(v: float, dec: int = 1) -> str:
    return f"{'+' if v >= 0 else ''}{v * 100:.{dec}f}%"


def monto_final(monto_inicial: float, retorno_acum: float) -> float:
    return monto_inicial * (1 + retorno_acum)
