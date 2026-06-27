"""
plots.py
========
Todas las visualizaciones con Plotly.
Paleta: azul marino #0A2342, azul eléctrico #2196F3,
        teal #00BFA5, naranja #FF6B35, dorado #F9A825.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

# ── Paleta ──────────────────────────────────────────────────────────────────
C = {
    "marino":   "#0A2342",
    "azul":     "#2196F3",
    "teal":     "#00BFA5",
    "naranja":  "#FF6B35",
    "dorado":   "#F9A825",
    "verde":    "#1a7a4a",
    "rojo":     "#c0392b",
    "gris":     "#9E9E9E",
    "grid":     "#E0E0E0",
}

PALETA = ["#2196F3","#00BFA5","#FF6B35","#9C27B0","#F9A825",
          "#1a7a4a","#c0392b","#0A2342","#00ACC1","#FF8F00",
          "#7B1FA2","#388E3C","#E91E63","#795548","#607D8B"]

LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, Arial", color=C["marino"], size=12),
    margin=dict(l=55, r=25, t=55, b=45),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["grid"],
                borderwidth=1, font_size=11),
)


def _fig(titulo, xt="", yt="", h=450):
    fig = go.Figure()
    fig.update_layout(**LAYOUT,
        title=dict(text=titulo, font=dict(size=15, color=C["marino"])),
        xaxis_title=xt, yaxis_title=yt, height=h)
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def grafico_precios_normalizados(precios, tickers_arg=None, titulo="Evolución — Base 100"):
    tickers_arg = tickers_arg or []
    base = (precios / precios.iloc[0]) * 100
    fig = _fig(titulo, yt="Valor (Base 100)", h=460)
    for i, t in enumerate(base.columns):
        fig.add_trace(go.Scatter(
            x=base.index, y=base[t], name=t, mode="lines",
            line=dict(color=PALETA[i % len(PALETA)], width=2.5,
                      dash="solid" if t in tickers_arg else "dot"),
            hovertemplate=f"<b>{t}</b><br>%{{x|%d %b %Y}}<br>Base 100: %{{y:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dash", line_color=C["grid"], line_width=1)
    fig.update_layout(hovermode="x unified")
    return fig


def grafico_retornos_acumulados(retornos, titulo="Retornos Acumulados"):
    acum = (1 + retornos).cumprod() - 1
    fig  = _fig(titulo, yt="Retorno Acumulado", h=430)
    for i, t in enumerate(acum.columns):
        fig.add_trace(go.Scatter(
            x=acum.index, y=acum[t], name=t, mode="lines",
            line=dict(color=PALETA[i % len(PALETA)], width=2),
            hovertemplate=f"<b>{t}</b><br>%{{x|%d %b %Y}}<br>%{{y:.1%}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=C["grid"])
    fig.update_layout(yaxis_tickformat=".0%", hovermode="x unified")
    return fig


def grafico_retorno_riesgo(metricas, tickers_arg=None, titulo="Retorno vs. Riesgo"):
    tickers_arg = tickers_arg or []
    fig = _fig(titulo, xt="Volatilidad Anualizada", yt="Retorno Anualizado", h=480)
    for t in metricas.index:
        row  = metricas.loc[t]
        ret  = row["Retorno Anualizado"]
        vol  = row["Volatilidad Anual"]
        shr  = row["Sharpe Ratio"]
        col  = C["naranja"] if t in tickers_arg else C["azul"]
        sym  = "star" if t in tickers_arg else "circle"
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text", name=t, text=[t],
            textposition="top center",
            marker=dict(size=max(12, abs(shr)*18), color=col,
                        symbol=sym, line=dict(width=1.5, color="white"), opacity=0.85),
            hovertemplate=(f"<b>{t}</b><br>Retorno: {ret:.1%}<br>"
                           f"Volatilidad: {vol:.1%}<br>Sharpe: {shr:.2f}<extra></extra>"),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=C["grid"])
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", showlegend=False)
    return fig


def grafico_correlacion(retornos, titulo="Matriz de Correlación"):
    corr = retornos.corr().round(2)
    n    = len(corr)
    fig  = px.imshow(corr, text_auto=True, color_continuous_scale="RdYlGn",
                     zmin=-1, zmax=1, aspect="auto")
    fig.update_traces(texttemplate="%{text:.2f}", textfont_size=11)
    fig.update_layout(**LAYOUT, title=titulo, height=max(350, n * 50))
    fig.update_layout(coloraxis_colorbar=dict(title="ρ"))
    return fig


def grafico_frontera_eficiente(
    mc_df, frontera_df=None, w_ms=None, w_mv=None, w_eq=None,
    retornos=None, rf=0.0, titulo="Frontera Eficiente de Markowitz"
):
    from portfolio_model import port_retorno, port_volatilidad
    fig = _fig(titulo, xt="Volatilidad Anualizada", yt="Retorno Anualizado", h=530)

    # Nube Monte Carlo
    fig.add_trace(go.Scatter(
        x=mc_df["volatilidad"], y=mc_df["retorno"], mode="markers",
        name="Portafolios simulados",
        marker=dict(size=3, color=mc_df["sharpe"], colorscale="RdYlGn",
                    showscale=True, colorbar=dict(title="Sharpe", thickness=12),
                    opacity=0.45),
        hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
    ))

    # Frontera eficiente
    if frontera_df is not None and not frontera_df.empty:
        fig.add_trace(go.Scatter(
            x=frontera_df["volatilidad"], y=frontera_df["retorno_objetivo"],
            mode="lines", name="Frontera Eficiente",
            line=dict(color=C["marino"], width=3),
        ))

    if retornos is not None:
        # Máximo Sharpe
        if w_ms is not None:
            r, v = port_retorno(w_ms, retornos), port_volatilidad(w_ms, retornos)
            s = (r - rf) / v if v > 0 else 0
            fig.add_trace(go.Scatter(
                x=[v], y=[r], mode="markers", name=f"⭐ Máx. Sharpe ({s:.2f})",
                marker=dict(color=C["dorado"], size=20, symbol="star",
                            line=dict(width=2, color=C["marino"])),
                hovertemplate=f"<b>Máximo Sharpe</b><br>Vol: {v:.1%}<br>Ret: {r:.1%}<br>Sharpe: {s:.2f}<extra></extra>",
            ))
            # CML
            vr = np.linspace(0, mc_df["volatilidad"].max() * 1.1, 100)
            fig.add_trace(go.Scatter(
                x=vr, y=rf + s * vr, mode="lines", name="Capital Market Line",
                line=dict(color=C["dorado"], width=1.5, dash="dash"),
            ))

        # Mínima Varianza
        if w_mv is not None:
            r, v = port_retorno(w_mv, retornos), port_volatilidad(w_mv, retornos)
            fig.add_trace(go.Scatter(
                x=[v], y=[r], mode="markers", name=f"🛡️ Mín. Varianza (Vol {v:.1%})",
                marker=dict(color=C["teal"], size=20, symbol="star",
                            line=dict(width=2, color=C["marino"])),
                hovertemplate=f"<b>Mínima Varianza</b><br>Vol: {v:.1%}<br>Ret: {r:.1%}<extra></extra>",
            ))

        # Equiponderado
        if w_eq is not None:
            r, v = port_retorno(w_eq, retornos), port_volatilidad(w_eq, retornos)
            fig.add_trace(go.Scatter(
                x=[v], y=[r], mode="markers", name="◆ Equiponderado",
                marker=dict(color=C["gris"], size=14, symbol="diamond",
                            line=dict(width=2, color=C["marino"])),
            ))

    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", hovermode="closest")
    return fig


def grafico_evolucion_portafolios(ev_ms, ev_mv=None, ev_eq=None,
                                   titulo="Evolución Histórica (Base 100)"):
    fig = _fig(titulo, yt="Valor (Base 100)", h=430)
    fig.add_trace(go.Scatter(x=ev_ms.index, y=ev_ms, name="⭐ Máx. Sharpe",
        mode="lines", line=dict(color=C["dorado"], width=3),
        fill="tozeroy", fillcolor="rgba(249,168,37,0.05)"))
    if ev_mv is not None:
        fig.add_trace(go.Scatter(x=ev_mv.index, y=ev_mv, name="🛡️ Mín. Varianza",
            mode="lines", line=dict(color=C["teal"], width=2.5, dash="dash")))
    if ev_eq is not None:
        fig.add_trace(go.Scatter(x=ev_eq.index, y=ev_eq, name="◆ Equiponderado",
            mode="lines", line=dict(color=C["gris"], width=2, dash="dot")))
    fig.add_hline(y=100, line_dash="dot", line_color=C["grid"],
                  annotation_text="Base 100", annotation_position="right")
    fig.update_layout(hovermode="x unified")
    return fig


def grafico_barras_comparativo(tabla, titulo="Comparativa de Carteras"):
    metricas = ["Retorno Anual", "Volatilidad", "Sharpe"]
    carteras = tabla.index.tolist()
    colores  = [C["dorado"], C["teal"], C["gris"], C["azul"]]
    fig = _fig(titulo, h=400)
    for i, cartera in enumerate(carteras):
        vals = [
            tabla.loc[cartera, "Retorno Anual"] * 100,
            tabla.loc[cartera, "Volatilidad"]   * 100,
            tabla.loc[cartera, "Sharpe"],
        ]
        fig.add_trace(go.Bar(
            name=cartera, x=metricas, y=vals,
            marker_color=colores[i % len(colores)],
            text=[f"{v:.1f}" for v in vals], textposition="outside",
        ))
    fig.update_layout(barmode="group", yaxis_title="Valor")
    return fig


def grafico_composicion(pesos: dict, titulo="Composición del Portafolio"):
    tickers = list(pesos.keys())
    valores = [v * 100 for v in pesos.values()]
    colores = [PALETA[i % len(PALETA)] for i in range(len(tickers))]
    fig = go.Figure(go.Pie(
        labels=tickers, values=valores, hole=0.52,
        marker=dict(colors=colores, line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(**LAYOUT, title=titulo, height=370,
        annotations=[dict(text="Pesos", x=0.5, y=0.5,
                         font_size=13, showarrow=False, font_color=C["marino"])])
    return fig


def grafico_drawdown(precios, titulo="Drawdown por Activo"):
    fig = _fig(titulo, yt="Drawdown (%)", h=400)
    for i, t in enumerate(precios.columns):
        rm = precios[t].cummax()
        dd = (precios[t] - rm) / rm * 100
        col = PALETA[i % len(PALETA)]
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=t, mode="lines",
            line=dict(color=col, width=1.5),
            fill="tozeroy", fillcolor=f"rgba(33,150,243,0.05)",
            hovertemplate=f"<b>{t}</b><br>%{{x|%d %b %Y}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=C["grid"], line_width=0.5)
    fig.update_layout(hovermode="x unified")
    return fig


def grafico_distribucion(retornos, ticker, var_95=None,
                         titulo=None):
    ret = retornos[ticker].dropna()
    titulo = titulo or f"Distribución de Retornos — {ticker}"
    fig = _fig(titulo, xt="Retorno Diario (%)", yt="Frecuencia", h=380)
    fig.add_trace(go.Histogram(
        x=ret * 100, nbinsx=60, name="Retornos",
        marker_color=C["azul"], opacity=0.75,
    ))
    if var_95 is not None:
        fig.add_vline(x=var_95 * 100, line_dash="dash", line_color=C["rojo"],
            annotation_text=f"VaR 95%: {var_95:.2%}",
            annotation_position="top left",
            annotation_font_color=C["rojo"])
    fig.add_vline(x=0, line_color=C["grid"], line_width=1)
    return fig


def grafico_ranking_barras(metricas, columna, titulo, pct=True, asc=False):
    df = metricas[[columna]].sort_values(columna, ascending=not asc)
    colores = [C["verde"] if v >= 0 else C["rojo"] for v in df[columna]]
    fig = _fig(titulo, h=max(300, len(df) * 42))
    fig.add_trace(go.Bar(
        y=df.index, x=df[columna] * (100 if pct else 1),
        orientation="h", marker_color=colores,
        text=[f"{v:.1%}" if pct else f"{v:.2f}" for v in df[columna]],
        textposition="outside",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def grafico_pesos_comparativo(tabla, tickers, titulo="Pesos por Cartera"):
    fig = _fig(titulo, yt="Peso (%)", h=370)
    carteras = tabla.index.tolist()
    for i, t in enumerate(tickers):
        col = f"w_{t}"
        if col not in tabla.columns: continue
        vals = tabla[col] * 100
        fig.add_trace(go.Bar(
            name=t, x=carteras, y=vals,
            marker_color=PALETA[i % len(PALETA)],
            text=[f"{v:.0f}%" for v in vals], textposition="inside",
        ))
    fig.update_layout(barmode="stack")
    return fig
