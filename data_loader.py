"""
data_loader.py
==============
Descarga y validación de datos financieros via Yahoo Finance.
Incluye activos argentinos (Bolsa de Buenos Aires .BA y ADRs NYSE),
CEDEARs, acciones internacionales y ETFs.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE ACTIVOS
# ─────────────────────────────────────────────────────────────────────────────

ACCIONES_BA = {
    "YPF (BYMA)":            "YPFD.BA",
    "Galicia (BYMA)":        "GGAL.BA",
    "Pampa Energía (BYMA)":  "PAMP.BA",
    "Ternium Arg. (BYMA)":   "TXAR.BA",
    "TGS (BYMA)":            "TGSU2.BA",
    "Central Puerto (BYMA)": "CEPU.BA",
    "Aluar (BYMA)":          "ALUA.BA",
    "Loma Negra (BYMA)":     "LOMA.BA",
    "Banco Macro (BYMA)":    "BMA.BA",
    "Supervielle (BYMA)":    "SUPV.BA",
}

ADRS_NYSE = {
    "YPF ADR (NYSE)":        "YPF",
    "Galicia ADR (NYSE)":    "GGAL",
    "Pampa ADR (NYSE)":      "PAM",
    "MercadoLibre (NYSE)":   "MELI",
    "Globant (NYSE)":        "GLOB",
    "Banco Macro ADR":       "BMA",
    "Central Puerto ADR":    "CEPU",
    "Bioceres (NYSE)":       "BIOX",
}

INTERNACIONALES = {
    "Apple (AAPL)":          "AAPL",
    "Microsoft (MSFT)":      "MSFT",
    "NVIDIA (NVDA)":         "NVDA",
    "Amazon (AMZN)":         "AMZN",
    "Alphabet (GOOGL)":      "GOOGL",
    "Meta (META)":           "META",
    "Tesla (TSLA)":          "TSLA",
    "Coca-Cola (KO)":        "KO",
    "JPMorgan (JPM)":        "JPM",
    "Berkshire B (BRK-B)":   "BRK-B",
    "Exxon (XOM)":           "XOM",
}

ETFS = {
    "S&P 500 ETF (SPY)":     "SPY",
    "Nasdaq 100 (QQQ)":      "QQQ",
    "Brasil ETF (EWZ)":      "EWZ",
    "Gold ETF (GLD)":        "GLD",
    "Bonos Tesoro USA (TLT)":"TLT",
    "Emerg. Markets (EEM)":  "EEM",
    "Bonos EM (EMB)":        "EMB",
}

# Diccionarios agrupados para el sidebar
GRUPOS_ACTIVOS = {
    "🇦🇷 Acciones BYMA (en ARS)": ACCIONES_BA,
    "🇦🇷 ADRs argentinos (NYSE)": ADRS_NYSE,
    "🌎 Acciones internacionales": INTERNACIONALES,
    "📦 ETFs y Bonos":             ETFS,
}

# Catálogo unificado nombre → ticker
TODOS_LOS_ACTIVOS = {
    **ACCIONES_BA, **ADRS_NYSE, **INTERNACIONALES, **ETFS
}

# Ticker → nombre corto
TICKER_A_NOMBRE = {v: k for k, v in TODOS_LOS_ACTIVOS.items()}

# Sets por origen
TICKERS_BA  = set(ACCIONES_BA.values())
TICKERS_ADR = set(ADRS_NYSE.values())
TICKERS_INT = set(INTERNACIONALES.values())
TICKERS_ETF = set(ETFS.values())
TICKERS_ARG = TICKERS_BA | TICKERS_ADR


def es_argentino(ticker: str) -> bool:
    return ticker.upper() in TICKERS_ARG


def grupo_ticker(ticker: str) -> str:
    t = ticker.upper()
    if t in TICKERS_BA:  return "BYMA"
    if t in TICKERS_ADR: return "ADR"
    if t in TICKERS_ETF: return "ETF"
    return "Internacional"


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def descargar_precios(
    tickers: list,
    fecha_inicio: str,
    fecha_fin: str,
) -> tuple:
    """
    Descarga precios de cierre ajustados desde Yahoo Finance.

    Returns
    -------
    precios   : DataFrame (fecha × ticker)
    exitosos  : lista de tickers con datos
    fallidos  : lista de tickers sin datos
    """
    if not tickers:
        return pd.DataFrame(), [], []

    try:
        raw = yf.download(
            tickers,
            start=fecha_inicio,
            end=fecha_fin,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
    except Exception:
        return pd.DataFrame(), [], list(tickers)

    # Extraer Close correctamente para 1 o N tickers
    if len(tickers) == 1:
        if "Close" in raw.columns:
            precios = raw[["Close"]].copy()
            precios.columns = tickers
        else:
            return pd.DataFrame(), [], list(tickers)
    else:
        if isinstance(raw.columns, pd.MultiIndex):
            precios = raw["Close"].copy()
        else:
            precios = raw[["Close"]].copy()

    precios = precios.dropna(axis=1, how="all")
    exitosos = [t for t in tickers if t in precios.columns]
    fallidos = [t for t in tickers if t not in precios.columns]
    precios  = precios[exitosos].dropna() if exitosos else pd.DataFrame()

    # Filtrar con menos de 30 filas
    insuf    = [t for t in exitosos if precios[t].count() < 30]
    exitosos = [t for t in exitosos if t not in insuf]
    fallidos = fallidos + insuf
    precios  = precios[exitosos] if exitosos else pd.DataFrame()

    return precios, exitosos, fallidos


def calcular_retornos_log(precios: pd.DataFrame) -> pd.DataFrame:
    """Retornos logarítmicos diarios: ln(P_t / P_{t-1})."""
    return np.log(precios / precios.shift(1)).dropna()


def normalizar_base_100(precios: pd.DataFrame) -> pd.DataFrame:
    """Normaliza precios a base 100 desde la primera observación."""
    return (precios / precios.iloc[0]) * 100


def calcular_periodo_texto(fecha_inicio: str, fecha_fin: str) -> str:
    fmt = "%Y-%m-%d"
    ini = datetime.strptime(fecha_inicio, fmt)
    fin = datetime.strptime(fecha_fin, fmt)
    anios = round((fin - ini).days / 365, 1)
    return (
        f"{ini.strftime('%d %b %Y')} — {fin.strftime('%d %b %Y')} "
        f"({anios} año{'s' if anios != 1 else ''})"
    )


def obtener_info_activo(ticker: str) -> dict:
    """Metadata de un activo via yfinance.info. Nunca lanza excepción."""
    defaults = {"nombre": None, "sector": None, "industria": None,
                "pais": None, "moneda": None, "tipo": None}
    try:
        info = yf.Ticker(ticker).info
        defaults.update({
            "nombre":    info.get("longName") or info.get("shortName"),
            "sector":    info.get("sector"),
            "industria": info.get("industry"),
            "pais":      info.get("country"),
            "moneda":    info.get("currency"),
            "tipo":      info.get("quoteType"),
        })
    except Exception:
        pass
    return defaults
