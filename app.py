"""
app.py  —  Optimizador de Portafolio Interactivo
=================================================
Ejecutar: streamlit run app.py

Tabs:
  1. Evaluación
  2. Comparador
  3. Frontera Eficiente
  4. Riesgo
  5. Contexto Financiero
  6. Metodología
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

from data_loader import (
    GRUPOS_ACTIVOS, TICKERS_ARG, TICKER_A_NOMBRE,
    descargar_precios, calcular_retornos_log, es_argentino,
    calcular_periodo_texto,
)
from utils import (
    tabla_metricas, generar_foda, monto_final,
    retorno_acumulado, formatear_pct,
)
from portfolio_model import (
    simulacion_monte_carlo, optimizar_max_sharpe, optimizar_min_varianza,
    portafolio_equiponderado, metricas_portafolio,
    calcular_frontera_eficiente, tabla_comparativa_carteras,
)
from plots import (
    grafico_precios_normalizados, grafico_retornos_acumulados,
    grafico_retorno_riesgo, grafico_correlacion,
    grafico_frontera_eficiente, grafico_evolucion_portafolios,
    grafico_barras_comparativo, grafico_composicion,
    grafico_drawdown, grafico_distribucion,
    grafico_ranking_barras, grafico_pesos_comparativo,
)
from ai_interpreter import generar_interpretacion, detectar_api_disponible

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Optimizador de Portafolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #F4F6F9;
    color: #1A1A2E;
}
h1 { font-family:'Space Grotesk',sans-serif; font-weight:700;
     color:#0A2342; font-size:2.1rem; letter-spacing:-0.5px; margin-bottom:.2rem; }
h2,h3,h4 { font-family:'Space Grotesk',sans-serif; color:#0A2342; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0A2342 0%, #0D3B6E 100%) !important;
}
[data-testid="stSidebar"] *          { color: #E8EAF6 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong     { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg,#2196F3,#00BFA5);
    color:white !important; font-family:'Space Grotesk',sans-serif;
    font-weight:600; border:none; border-radius:8px;
    padding:.75rem 1.5rem; width:100%; font-size:.95rem;
    letter-spacing:.5px; margin-top:.5rem;
}
[data-testid="stSidebar"] .stButton>button:hover { opacity:.88; }

/* ── KPI cards ── */
.kpi { background:white; border:1px solid #E0E0E0; border-radius:12px;
       padding:1.1rem 1.4rem; box-shadow:0 1px 4px rgba(0,0,0,.06); margin:.25rem 0; }
.kpi-label { font-size:.68rem; letter-spacing:1.5px; text-transform:uppercase;
             color:#757575; font-weight:500; margin-bottom:.2rem; }
.kpi-value { font-family:'Space Grotesk',sans-serif; font-size:1.85rem;
             font-weight:700; line-height:1.1; }
.pos { color:#1a7a4a; } .neg { color:#c0392b; } .neu { color:#0A2342; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap:3px; background:#EBF0F5; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] {
    border-radius:7px; font-size:.83rem; font-weight:500;
    color:#546E7A; padding:.38rem .9rem; }
.stTabs [aria-selected="true"] {
    background:white !important; color:#0A2342 !important;
    font-weight:600; box-shadow:0 1px 3px rgba(0,0,0,.1); }

/* ── Info / warning boxes ── */
.ibox { background:#E3F2FD; border-left:4px solid #2196F3;
        border-radius:4px; padding:.7rem 1rem; font-size:.85rem;
        color:#0D47A1; margin:.5rem 0; }
.wbox { background:#FFF8E1; border-left:4px solid #F9A825;
        border-radius:4px; padding:.7rem 1rem; font-size:.85rem;
        color:#5D4037; margin:.5rem 0; }
.errbox { background:#FFEBEE; border-left:4px solid #c0392b;
          border-radius:4px; padding:.7rem 1rem; font-size:.85rem;
          color:#b71c1c; margin:.5rem 0; }

/* ── FODA ── */
.foda { border-radius:10px; padding:.9rem 1rem; margin:.25rem 0; }
.foda-f{background:#E8F5E9;border-top:3px solid #1a7a4a;}
.foda-o{background:#E3F2FD;border-top:3px solid #2196F3;}
.foda-d{background:#FFF3E0;border-top:3px solid #FF6B35;}
.foda-a{background:#FFEBEE;border-top:3px solid #c0392b;}

footer { display:none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi(label, valor, cls="neu"):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {cls}">{valor}</div>
    </div>""", unsafe_allow_html=True)

def ibox(txt): st.markdown(f'<div class="ibox">💡 {txt}</div>', unsafe_allow_html=True)
def wbox(txt): st.markdown(f'<div class="wbox">⚠️ {txt}</div>', unsafe_allow_html=True)
def vacío(msg="Ejecutá el análisis para ver resultados."):
    st.markdown(f"""
    <div style="text-align:center;padding:3.5rem;color:#B0BEC5;">
      <p style="font-size:3rem;margin:0;">📊</p>
      <p style="font-family:'Space Grotesk',sans-serif;font-size:1rem;margin-top:.8rem;">{msg}</p>
    </div>""", unsafe_allow_html=True)

def s(k, d=None): return st.session_state.get(k, d)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PERFIL DEL INVERSOR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 👤 Perfil del Inversor")
    st.markdown("---")

    # ── Selección de activos por grupo ──────────────────────────────────────
    tickers_seleccionados = []
    for grupo, activos in GRUPOS_ACTIVOS.items():
        with st.expander(grupo, expanded=(grupo == "🇦🇷 ADRs argentinos (NYSE)")):
            nombres = list(activos.keys())
            defaults = []
            if "ADRs" in grupo:
                defaults = [n for n in nombres if activos[n] in ["YPF","GGAL","PAM","MELI"]]
            elif "internacionales" in grupo:
                defaults = [n for n in nombres if activos[n] in ["AAPL","NVDA"]]
            elif "ETFs" in grupo:
                defaults = [n for n in nombres if activos[n] in ["SPY"]]

            sel = st.multiselect(
                "Seleccioná activos",
                options=nombres,
                default=defaults,
                key=f"sel_{grupo}",
                label_visibility="collapsed",
            )
            for nombre in sel:
                t = activos[nombre]
                if t not in tickers_seleccionados:
                    tickers_seleccionados.append(t)

    # ── Ticker manual ────────────────────────────────────────────────────────
    st.markdown("---")
    ticker_manual = st.text_input(
        "✏️ Agregar ticker manual",
        placeholder="Ej: TSLA, GD30.BA, TLT ...",
        help="Cualquier símbolo válido de Yahoo Finance.",
    )
    if ticker_manual.strip():
        t = ticker_manual.strip().upper()
        if t not in tickers_seleccionados:
            tickers_seleccionados.append(t)

    # ── Período ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Período")
    c1, c2 = st.columns(2)
    with c1:
        fecha_ini = st.date_input("Desde",
            value=date.today() - timedelta(days=3*365),
            min_value=date(2010,1,1), max_value=date.today()-timedelta(90))
    with c2:
        fecha_fin = st.date_input("Hasta",
            value=date.today(),
            min_value=date(2010,1,1), max_value=date.today())

    # ── Parámetros ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Parámetros")

    rf = st.slider("Tasa libre de riesgo (%)",
                   0.0, 15.0, 5.0, 0.25,
                   help="Usada para el Sharpe Ratio. Referencia: tasa de política monetaria.") / 100

    monto_inicial = st.number_input(
        "💰 Monto inicial a invertir (USD)",
        min_value=100, max_value=10_000_000,
        value=10_000, step=1000,
        help="Solo para calcular el valor final estimado del portafolio.",
    )

    permitir_short = st.checkbox(
        "Permitir posiciones cortas (short)",
        value=False,
        help="Si está activado, los activos pueden tener peso negativo.",
    )
    peso_min = -1.0 if permitir_short else 0.0

    n_sim = st.select_slider(
        "Portafolios Monte Carlo",
        options=[1_000, 3_000, 5_000, 10_000],
        value=5_000,
    )

    # ── API IA ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Interpretación IA (opcional)")
    api_status = detectar_api_disponible()
    if api_status != "ninguna":
        st.markdown(f'<p style="color:#A5D6A7;font-size:.8rem;">✅ API detectada: {api_status}</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#80CBC4;font-size:.8rem;">Sin API → análisis automático.</p>',
                    unsafe_allow_html=True)
    api_key = st.text_input("API Key (Anthropic / OpenAI)", type="password",
                            placeholder="sk-ant-... o sk-...")

    st.markdown("---")
    ejecutar = st.button("▶  EVALUAR RIESGO", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<h1>📈 Optimizador de Portafolio Interactivo</h1>
<p style="color:#546E7A;font-size:1rem;margin-top:0;margin-bottom:1.2rem;">
  Aplicación de Markowitz, frontera eficiente y métricas de riesgo
  para activos argentinos e internacionales ·
  <span style="background:#E3F2FD;color:#1565C0;
    padding:2px 8px;border-radius:10px;font-size:.75rem;">
    Solo uso educativo
  </span>
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📋 Evaluación",
    "🔀 Comparador",
    "🎯 Frontera Eficiente",
    "⚡ Riesgo",
    "🌐 Contexto Financiero",
    "📚 Metodología",
])


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN DEL ANÁLISIS
# ─────────────────────────────────────────────────────────────────────────────

if ejecutar:
    if len(tickers_seleccionados) < 2:
        st.error("⚠️ Seleccioná al menos **2 activos** para continuar.")
        st.stop()

    tickers_arg_sel = [t for t in tickers_seleccionados if es_argentino(t)]
    tickers_int_sel = [t for t in tickers_seleccionados if not es_argentino(t)]

    with st.spinner("⏳ Descargando datos desde Yahoo Finance..."):
        precios, exitosos, fallidos = descargar_precios(
            tickers_seleccionados,
            fecha_ini.strftime("%Y-%m-%d"),
            fecha_fin.strftime("%Y-%m-%d"),
        )

    if fallidos:
        st.warning(f"⚠️ No se pudieron descargar: **{', '.join(fallidos)}** — excluidos del análisis.")

    if len(exitosos) < 2:
        st.error("❌ Menos de 2 activos con datos válidos. Ajustá la selección o el período.")
        st.stop()

    tickers_arg_sel = [t for t in tickers_arg_sel if t in exitosos]
    tickers_int_sel = [t for t in tickers_int_sel if t in exitosos]
    retornos = calcular_retornos_log(precios)
    metricas = tabla_metricas(precios, retornos, rf)

    with st.spinner("⚙️ Optimizando portafolios..."):
        w_ms = optimizar_max_sharpe(retornos, rf, peso_min)
        w_mv = optimizar_min_varianza(retornos, peso_min)
        w_eq = portafolio_equiponderado(retornos)

    with st.spinner("🎲 Simulación Monte Carlo..."):
        mc_df = simulacion_monte_carlo(retornos, n_sim, rf)

    with st.spinner("📐 Calculando frontera eficiente..."):
        frontera_df = calcular_frontera_eficiente(retornos, n_puntos=80, peso_min=peso_min)

    ms_met = metricas_portafolio(w_ms, retornos, precios, rf) if w_ms is not None else {}
    mv_met = metricas_portafolio(w_mv, retornos, precios, rf) if w_mv is not None else {}
    eq_met = metricas_portafolio(w_eq, retornos, precios, rf)
    tabla_cart = tabla_comparativa_carteras(retornos, precios, w_ms, w_mv, rf)
    periodo_txt = calcular_periodo_texto(fecha_ini.strftime("%Y-%m-%d"),
                                         fecha_fin.strftime("%Y-%m-%d"))

    st.session_state.update({
        "precios": precios, "retornos": retornos, "exitosos": exitosos,
        "metricas": metricas, "mc_df": mc_df, "frontera_df": frontera_df,
        "w_ms": w_ms, "w_mv": w_mv, "w_eq": w_eq,
        "ms_met": ms_met, "mv_met": mv_met, "eq_met": eq_met,
        "tabla_cart": tabla_cart,
        "tickers_arg": tickers_arg_sel, "tickers_int": tickers_int_sel,
        "periodo_txt": periodo_txt, "rf": rf,
        "monto_inicial": monto_inicial,
        "api_key": api_key,
        "listo": True,
    })
    st.success(f"✅ Análisis completado · {len(exitosos)} activos · {periodo_txt}")

listo = s("listo", False)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EVALUACIÓN
# ═════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    if not listo:
        vacío("Completá el perfil del inversor y hacé clic en ▶ EVALUAR RIESGO.")
    else:
        precios    = s("precios")
        retornos   = s("retornos")
        metricas   = s("metricas")
        ms_met     = s("ms_met", {})
        mv_met     = s("mv_met", {})
        exitosos   = s("exitosos")
        tickers_arg = s("tickers_arg", [])
        monto_ini  = s("monto_inicial", 10_000)
        periodo    = s("periodo_txt")

        st.markdown(f"### Evaluación · {periodo}")
        st.markdown(f"**Activos analizados:** {', '.join(exitosos)}")
        st.markdown("---")

        # ── KPIs portafolio de máximo Sharpe ────────────────────────────────
        st.markdown("#### ⭐ Portafolio de Máximo Sharpe")
        c1, c2, c3, c4, c5 = st.columns(5)
        ms_ret  = ms_met.get("retorno_anual", 0) or 0
        ms_vol  = ms_met.get("volatilidad", 0) or 0
        ms_shr  = ms_met.get("sharpe", 0) or 0
        ms_mdd  = ms_met.get("max_drawdown", 0) or 0
        ms_racu = retorno_acumulado(ms_met["evolucion"] / 100) if ms_met.get("evolucion") is not None else 0
        ms_monto = monto_final(monto_ini, ms_racu)

        with c1: kpi("Retorno Anual", f"{ms_ret:+.1%}", "pos" if ms_ret>=0 else "neg")
        with c2: kpi("Volatilidad Anual", f"{ms_vol:.1%}", "neu")
        with c3: kpi("Sharpe Ratio", f"{ms_shr:.2f}",
                     "pos" if ms_shr>=1 else "neu" if ms_shr>=0.5 else "neg")
        with c4: kpi("Máx. Drawdown", f"{ms_mdd:.1%}", "neg" if ms_mdd<-0.20 else "neu")
        with c5: kpi(f"Monto Final (USD ${monto_ini:,.0f})",
                     f"${ms_monto:,.0f}", "pos" if ms_monto>monto_ini else "neg")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Precios base 100 ─────────────────────────────────────────────────
        st.markdown("#### Evolución de Activos — Base 100")
        fig = grafico_precios_normalizados(precios, tickers_arg)
        st.plotly_chart(fig, use_container_width=True)
        ibox("Línea sólida = activos argentinos. Línea punteada = activos internacionales. "
             "Base 100 permite comparar activos con precios absolutos muy distintos.")

        # ── Tabla de métricas individuales ───────────────────────────────────
        st.markdown("#### Tabla de Métricas Individuales")
        cols_show = ["Precio Inicial","Precio Final","Retorno Acumulado",
                     "Retorno Anualizado","Volatilidad Anual","Sharpe Ratio",
                     "Sortino Ratio","Max Drawdown","VaR 95% (diario)"]
        cols_ok   = [c for c in cols_show if c in metricas.columns]
        pct_c     = ["Retorno Acumulado","Retorno Anualizado","Volatilidad Anual",
                     "Max Drawdown","VaR 95% (diario)"]
        fmt = {c: ("{:.1%}" if c in pct_c else
                   "{:.3f}" if "Ratio" in c else "{:.2f}") for c in cols_ok}
        st.dataframe(
            metricas[cols_ok].style.format(fmt)
                .background_gradient(subset=["Retorno Anualizado"], cmap="RdYlGn")
                .background_gradient(subset=["Sharpe Ratio"],       cmap="RdYlGn"),
            use_container_width=True,
        )
        csv = metricas[cols_ok].to_csv().encode("utf-8")
        st.download_button("⬇️ Descargar métricas CSV", csv,
                           "metricas.csv", "text/csv")

        st.markdown("---")

        # ── Resultados portafolios óptimos ───────────────────────────────────
        st.markdown("#### Portafolios Óptimos")
        col_ms, col_mv = st.columns(2)

        def _bloque_portafolio(col, titulo, met, color):
            with col:
                st.markdown(f"<h4 style='color:{color}'>{titulo}</h4>",
                            unsafe_allow_html=True)
                if not met:
                    st.warning("Optimización no convergió. Probá con más activos.")
                    return
                ret = met.get("retorno_anual", 0) or 0
                vol = met.get("volatilidad", 0) or 0
                shr = met.get("sharpe", 0) or 0
                mdd = met.get("max_drawdown", 0) or 0
                var = met.get("var_95_diario", 0) or 0
                st.markdown(f"- **Retorno anual:** {ret:.1%}")
                st.markdown(f"- **Volatilidad:** {vol:.1%}")
                st.markdown(f"- **Sharpe Ratio:** {shr:.2f}")
                st.markdown(f"- **Sortino Ratio:** {met.get('sortino', 0):.2f}")
                st.markdown(f"- **Máx. Drawdown:** {mdd:.1%}")
                st.markdown(f"- **VaR 95% diario:** {var:.2%}")
                st.markdown("**Composición:**")
                pesos = met.get("pesos", {})
                for t, pw in sorted(pesos.items(), key=lambda x: -x[1]):
                    bar = "█" * int(pw * 20) + "░" * (20 - int(pw * 20))
                    st.markdown(f"`{t}` {bar} {pw:.1%}")

        _bloque_portafolio(col_ms, "⭐ Máximo Sharpe",   ms_met, "#F9A825")
        _bloque_portafolio(col_mv, "🛡️ Mínima Varianza", mv_met, "#00BFA5")

        st.markdown("---")

        # ── Interpretación automática ─────────────────────────────────────────
        st.markdown("#### 🤖 Interpretación Automática")
        if st.button("Generar interpretación", key="btn_eval_ia"):
            with st.spinner("Analizando..."):
                ant_key = s("api_key","") if (s("api_key","") or "").startswith("sk-ant") else None
                oai_key = s("api_key","") if (s("api_key","") or "").startswith("sk-") and not ant_key else None
                texto, fuente = generar_interpretacion(
                    tickers=exitosos,
                    metricas_dict=metricas.T.to_dict(),
                    portafolio_ms=ms_met,
                    portafolio_mv=mv_met,
                    periodo=s("periodo_txt",""),
                    tasa_libre_riesgo=s("rf", 0.0),
                    api_key_anthropic=ant_key,
                    api_key_openai=oai_key,
                )
                st.session_state["ia_texto"]  = texto
                st.session_state["ia_fuente"] = fuente

        if s("ia_texto"):
            st.caption(f"Fuente: {s('ia_fuente')}")
            st.markdown(s("ia_texto"))

        wbox("Resultados basados en datos históricos. No constituyen asesoramiento financiero.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARADOR
# ═════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    if not listo:
        vacío()
    else:
        retornos   = s("retornos")
        precios    = s("precios")
        tabla_cart = s("tabla_cart")
        ms_met     = s("ms_met", {})
        mv_met     = s("mv_met", {})
        eq_met     = s("eq_met", {})
        tickers    = retornos.columns.tolist()

        st.markdown("### 🔀 Comparador de Carteras")
        ibox("Comparamos tres estrategias: Máximo Sharpe, Mínima Varianza y Equiponderada (1/N).")

        # ── Tabla comparativa ────────────────────────────────────────────────
        st.markdown("#### Tabla Comparativa")
        met_cols = ["Retorno Anual","Volatilidad","Sharpe","Max Drawdown","VaR 95%"]
        met_ok   = [c for c in met_cols if c in tabla_cart.columns]
        fmt_tc   = {c: "{:.2%}" for c in met_ok}
        fmt_tc["Sharpe"] = "{:.3f}"
        st.dataframe(
            tabla_cart[met_ok].style.format(fmt_tc)
                .background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # ── Gráficos ─────────────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Evolución Histórica (Base 100)")
            ev_ms = ms_met.get("evolucion")
            ev_mv = mv_met.get("evolucion")
            ev_eq = eq_met.get("evolucion")
            if ev_ms is not None:
                fig = grafico_evolucion_portafolios(ev_ms, ev_mv, ev_eq)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Retorno, Volatilidad y Sharpe")
            fig2 = grafico_barras_comparativo(tabla_cart)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Composición por cartera ──────────────────────────────────────────
        st.markdown("#### Composición de cada Cartera")
        col_a, col_b, col_c = st.columns(3)

        def _donut(col, titulo, met):
            with col:
                pesos = met.get("pesos", {})
                if pesos:
                    fig = grafico_composicion(pesos, titulo)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sin datos")

        _donut(col_a, "⭐ Máximo Sharpe",   ms_met)
        _donut(col_b, "🛡️ Mínima Varianza", mv_met)
        _donut(col_c, "◆ Equiponderado",    eq_met)

        # ── Pesos apilados ───────────────────────────────────────────────────
        st.markdown("#### Pesos por Activo y Cartera")
        fig3 = grafico_pesos_comparativo(tabla_cart, tickers)
        st.plotly_chart(fig3, use_container_width=True)

        # ── Explicación de dominio ───────────────────────────────────────────
        st.markdown("#### ¿Qué cartera domina y por qué?")
        ms_s = ms_met.get("sharpe", 0) or 0
        mv_s = mv_met.get("sharpe", 0) or 0
        eq_s = eq_met.get("sharpe", 0) or 0
        ms_v = ms_met.get("volatilidad", 1) or 1
        mv_v = mv_met.get("volatilidad", 1) or 1

        mejor_sharpe = max([("Máximo Sharpe", ms_s), ("Mínima Varianza", mv_s),
                            ("Equiponderado", eq_s)], key=lambda x: x[1])
        menor_vol    = "Mínima Varianza" if mv_v <= ms_v else "Máximo Sharpe"

        st.markdown(f"""
**En términos de Sharpe Ratio**, el portafolio de **{mejor_sharpe[0]}** es el más eficiente
({mejor_sharpe[1]:.2f}), lo que significa que ofrece el mayor retorno por unidad de riesgo.

**En términos de volatilidad**, el portafolio de **{menor_vol}** es el más conservador.

El portafolio **equiponderado** (1/N) rara vez es óptimo según el criterio de Markowitz:
asignar partes iguales sin considerar correlaciones desperdicia el potencial de diversificación.
La diferencia entre el equiponderado y el máximo Sharpe ilustra cuánto valor agrega
la optimización matemática frente a la intuición naïve.
        """)

        # Exportar
        csv_p = tabla_cart.to_csv().encode("utf-8")
        st.download_button("⬇️ Descargar tabla comparativa CSV",
                           csv_p, "carteras_comparativa.csv", "text/csv")

        wbox("Resultados históricos. No garantizan desempeño futuro.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — FRONTERA EFICIENTE
# ═════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    if not listo:
        vacío()
    else:
        retornos    = s("retornos")
        mc_df       = s("mc_df")
        frontera_df = s("frontera_df")
        w_ms, w_mv, w_eq = s("w_ms"), s("w_mv"), s("w_eq")
        rf          = s("rf", 0.0)

        st.markdown("### 🎯 Frontera Eficiente de Markowitz")

        fig = grafico_frontera_eficiente(
            mc_df, frontera_df, w_ms, w_mv, w_eq, retornos, rf
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Interpretación económica ─────────────────────────────────────────
        st.markdown("#### Interpretación Económica")
        n_sim_real = len(mc_df)
        mc_ms_shr  = mc_df["sharpe"].max()

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""
**¿Qué muestra este gráfico?**

Cada punto gris representa uno de los **{n_sim_real:,} portafolios simulados** 
mediante Monte Carlo, con pesos aleatorios. El color indica el Sharpe Ratio:
verde = mayor eficiencia, rojo = menor eficiencia.

La **curva azul** es la *Frontera Eficiente*: el subconjunto de portafolios que 
**maximizan el retorno para cada nivel de riesgo dado**. Todo portafolio 
por debajo de esa curva está dominado — existe otro portafolio con el mismo riesgo 
y mayor retorno.

La **línea dorada** es la *Capital Market Line* (CML): la combinación óptima 
entre el activo libre de riesgo ({rf:.1%} anual) y el portafolio de máximo Sharpe. 
Cualquier punto sobre la CML supera a cualquier portafolio sobre la frontera eficiente 
en términos de Sharpe Ratio.
            """)
        with col2:
            st.markdown(f"""
**Referencias del gráfico:**

⭐ **Estrella dorada** = Máximo Sharpe  
  → Mayor retorno por unidad de riesgo  
  → Sharpe simulación: {mc_ms_shr:.2f}

🛡️ **Estrella teal** = Mínima Varianza  
  → Menor volatilidad posible  
  → Preferible en entornos de alta incertidumbre

◆ **Diamante gris** = Equiponderado (1/N)  
  → Benchmark de comparación  
  → Generalmente dominado

🟡 **Línea dorada** = Capital Market Line  
  → Con activo libre de riesgo al {rf:.1%}
            """)

        wbox("La frontera eficiente se construye con datos históricos. "
             "En períodos futuros las correlaciones y volatilidades pueden cambiar.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — RIESGO
# ═════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    if not listo:
        vacío()
    else:
        precios  = s("precios")
        retornos = s("retornos")
        metricas = s("metricas")

        st.markdown("### ⚡ Análisis de Riesgo")

        sub1, sub2, sub3, sub4 = st.tabs([
            "🔗 Correlaciones", "📉 Drawdown",
            "📊 VaR y Distribuciones", "🏆 Rankings",
        ])

        # ── Correlaciones ────────────────────────────────────────────────────
        with sub1:
            fig = grafico_correlacion(retornos)
            st.plotly_chart(fig, use_container_width=True)
            ibox("Correlaciones cercanas a 0 o negativas indican activos que se mueven "
                 "de forma independiente: mayor beneficio de diversificación.")

            # Pares más y menos correlacionados
            import itertools
            corr = retornos.corr()
            pares = [(a, b, corr.loc[a, b])
                     for a, b in itertools.combinations(corr.columns, 2)]
            pares.sort(key=lambda x: x[2])
            st.markdown("**Par con menor correlación** (mejor diversificador): "
                        f"`{pares[0][0]}` — `{pares[0][1]}` → ρ = {pares[0][2]:.3f}")
            st.markdown("**Par con mayor correlación** (menos diversificador): "
                        f"`{pares[-1][0]}` — `{pares[-1][1]}` → ρ = {pares[-1][2]:.3f}")

        # ── Drawdown ─────────────────────────────────────────────────────────
        with sub2:
            fig = grafico_drawdown(precios)
            st.plotly_chart(fig, use_container_width=True)

            cols_dd = ["Max Drawdown","VaR 95% (diario)","CVaR 95% (diario)",
                       "Volatilidad Anual","Peor Día","Fecha Peor Día"]
            cols_ok = [c for c in cols_dd if c in metricas.columns]
            fmt_dd  = {c: "{:.2%}" for c in cols_ok if c not in ["Fecha Peor Día","Peor Día"]}
            fmt_dd["Peor Día"] = "{:.2%}"
            st.dataframe(
                metricas[cols_ok].style.format(fmt_dd)
                    .background_gradient(subset=["Max Drawdown"], cmap="RdYlGn_r"),
                use_container_width=True,
            )

        # ── VaR y distribuciones ─────────────────────────────────────────────
        with sub3:
            ticker_dist = st.selectbox("Activo a analizar",
                                       retornos.columns.tolist(), key="dist_sel")
            var_val = metricas.loc[ticker_dist, "VaR 95% (diario)"] \
                      if ticker_dist in metricas.index else None
            fig = grafico_distribucion(retornos, ticker_dist, var_val)
            st.plotly_chart(fig, use_container_width=True)

            cvar_val = metricas.loc[ticker_dist, "CVaR 95% (diario)"] \
                       if ticker_dist in metricas.index else None
            c1, c2, c3 = st.columns(3)
            with c1: kpi("VaR 95% Diario",
                         f"{var_val:.2%}" if var_val else "N/D", "neg")
            with c2: kpi("CVaR 95% Diario",
                         f"{cvar_val:.2%}" if cvar_val else "N/D", "neg")
            with c3:
                vol_t = metricas.loc[ticker_dist, "Volatilidad Anual"] \
                        if ticker_dist in metricas.index else None
                kpi("Volatilidad Anual",
                    f"{vol_t:.1%}" if vol_t else "N/D", "neu")

            ibox(f"**VaR 95%**: en el 5% peor de los días, {ticker_dist} puede perder "
                 f"más del {abs(var_val):.2%}. "
                 f"**CVaR 95%**: en esos días malos, la pérdida promedio es {abs(cvar_val):.2%}.")

            # FODA
            st.markdown("---")
            st.markdown(f"#### FODA Financiero — {ticker_dist}")
            foda = generar_foda(ticker_dist, metricas)
            fa, fb = st.columns(2)
            with fa:
                st.markdown(f"""<div class="foda foda-f">
                  <strong style="color:#1a7a4a">💪 Fortalezas</strong>
                  <ul>{"".join(f"<li>{i}</li>" for i in foda["fortalezas"])}</ul></div>""",
                  unsafe_allow_html=True)
                st.markdown(f"""<div class="foda foda-o">
                  <strong style="color:#2196F3">🚀 Oportunidades</strong>
                  <ul>{"".join(f"<li>{i}</li>" for i in foda["oportunidades"])}</ul></div>""",
                  unsafe_allow_html=True)
            with fb:
                st.markdown(f"""<div class="foda foda-d">
                  <strong style="color:#FF6B35">⚠️ Debilidades</strong>
                  <ul>{"".join(f"<li>{i}</li>" for i in foda["debilidades"])}</ul></div>""",
                  unsafe_allow_html=True)
                st.markdown(f"""<div class="foda foda-a">
                  <strong style="color:#c0392b">🛑 Amenazas</strong>
                  <ul>{"".join(f"<li>{i}</li>" for i in foda["amenazas"])}</ul></div>""",
                  unsafe_allow_html=True)
            st.caption("FODA orientativo basado en métricas históricas. No es asesoramiento financiero.")

        # ── Rankings ─────────────────────────────────────────────────────────
        with sub4:
            c1, c2, c3 = st.columns(3)
            with c1:
                fig = grafico_ranking_barras(metricas, "Retorno Anualizado",
                                             "Ranking por Retorno")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = grafico_ranking_barras(metricas, "Volatilidad Anual",
                                             "Ranking por Volatilidad (↓ = mejor)",
                                             asc=True)
                st.plotly_chart(fig, use_container_width=True)
            with c3:
                fig = grafico_ranking_barras(metricas, "Sharpe Ratio",
                                             "Ranking por Sharpe Ratio", pct=False)
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — CONTEXTO FINANCIERO
# ═════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### 🌐 Contexto Financiero")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
#### ¿Qué problema económico resuelve esta aplicación?

El problema de asignación de capital es uno de los más antiguos de las finanzas:
dado un conjunto de activos disponibles, ¿cuánto dinero destinar a cada uno para
maximizar el retorno y minimizar el riesgo?

La respuesta intuitiva de muchos inversores —asignar partes iguales a todo—
es matemáticamente sub-óptima. Esta app implementa la solución de Markowitz
(1952), que demuestra que el riesgo de un portafolio **no es la suma** de
los riesgos individuales: las correlaciones entre activos son clave.

---

#### ¿Por qué es útil para inversores minoristas?

En Argentina, el inversor minorista típico enfrenta tres desafíos simultáneos:

1. **Riesgo cambiario**: diferencia entre activos en ARS y USD
2. **Riesgo país**: volatilidad macroeconómica elevada y eventos discontinuos
3. **Acceso fragmentado**: los CEDEARs, bonos y ADRs cotizan en mercados distintos

Esta herramienta unifica en un mismo análisis activos de la BYMA, ADRs en NYSE
y acciones internacionales, permitiendo ver qué combinación históricamente
habría ofrecido el mejor equilibrio retorno-riesgo.

---

#### ¿Por qué es relevante en Argentina?

- El tipo de cambio oficial y el CCL generan diferencias de precio no triviales
- Los bonos soberanos (AL30, GD30) tienen un comportamiento muy distinto a los bonos EM globales
- La diversificación hacia activos en USD reduce la exposición al riesgo ARS
- Los ADRs argentinos (YPF, GGAL, PAM) permiten invertir en empresas locales desde cuentas en el exterior

""")

    with col2:
        st.markdown("""
#### Usuarios a quienes está dirigida

Esta aplicación está diseñada para:

🎓 **Estudiantes** de economía, finanzas y administración que quieran ver Markowitz en acción con datos reales

📊 **Inversores minoristas** que buscan una herramienta educativa para comparar activos y carteras

👨‍🏫 **Docentes** que necesitan un prototipo interactivo para explicar teoría de portafolios

🏦 **Analistas** que quieran explorar combinaciones de activos antes de un análisis más profundo

---

#### Tecnología involucrada

| Componente | Tecnología |
|-----------|-----------|
| Interfaz web | Streamlit |
| Datos | yfinance (Yahoo Finance) |
| Optimización | scipy.optimize SLSQP |
| Visualización | Plotly |
| Procesamiento | pandas + numpy |
| IA | Anthropic Claude / OpenAI GPT |
""")

    st.markdown("---")
    st.markdown("#### ⚠️ Limitaciones del Modelo")

    lims = {
        "📉 Datos históricos": (
            "El modelo se basa en retornos pasados para estimar correlaciones y volatilidades futuras. "
            "Los mercados son dinámicos: lo que funcionó en el período analizado puede no repetirse."
        ),
        "📐 Sensibilidad a los inputs": (
            "El portafolio óptimo de Markowitz es muy sensible a pequeños cambios en las estimaciones "
            "de retornos esperados. Una estimación errónea puede cambiar radicalmente la composición."
        ),
        "💱 Riesgo cambiario": (
            "Los activos listados en USD (ADRs, acciones internacionales) tienen exposición cambiaria "
            "respecto al ARS. Un inversor argentino debe considerar la evolución del tipo de cambio."
        ),
        "💧 Liquidez": (
            "No todos los activos tienen el mismo nivel de liquidez. Los tickers .BA de la BYMA "
            "pueden tener spreads amplios y menor volumen que los ADRs en NYSE."
        ),
        "💸 Costos de transacción": (
            "El modelo asume que no hay costos de compra-venta, comisiones ni impuestos. "
            "En la práctica, estos costos reducen el retorno efectivo del portafolio."
        ),
        "📈 Inflación": (
            "Los retornos calculados son nominales en USD. Para un inversor en ARS, "
            "la inflación local erosiona el poder adquisitivo real."
        ),
        "📊 Normalidad": (
            "El modelo asume retornos normalmente distribuidos. Los mercados reales presentan "
            "colas pesadas (fat tails): los eventos extremos son más frecuentes de lo que predice la teoría."
        ),
    }

    c1, c2 = st.columns(2)
    items  = list(lims.items())
    for i, (titulo, desc) in enumerate(items):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"**{titulo}**")
            st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
            st.markdown("")

    wbox("Esta aplicación es exclusivamente educativa y no constituye asesoramiento financiero. "
         "Consultá a un profesional antes de tomar decisiones de inversión.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — METODOLOGÍA
# ═════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### 📚 Metodología")

    st.markdown("""
#### Teoría Moderna de Portafolios — Harry Markowitz (1952)

Harry Markowitz publicó en 1952 el artículo *"Portfolio Selection"* en el Journal of Finance,
sentando las bases de la gestión cuantitativa de inversiones. El aporte central:
un inversor racional no evalúa activos de forma aislada, sino en función de su
**contribución al riesgo y retorno del portafolio total**. Por este trabajo,
Markowitz recibió el Premio Nobel de Economía en 1990.

---

#### Retornos Logarítmicos

$$r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right)$$

Se usan retornos logarítmicos porque son **aditivos en el tiempo**
(el retorno acumulado es la suma de los diarios), se aproximan mejor a
una distribución normal y son simétricos: una suba y una baja del mismo porcentaje
se compensan exactamente.

---

#### Retorno y Volatilidad del Portafolio

| Métrica | Fórmula |
|---------|---------|
| Retorno esperado | $E[r_p] = \\mathbf{w}^T \\boldsymbol{\\mu} \\times 252$ |
| Varianza del portafolio | $\\sigma^2_p = \\mathbf{w}^T \\boldsymbol{\\Sigma} \\mathbf{w}$ |
| Volatilidad | $\\sigma_p = \\sqrt{\\mathbf{w}^T \\boldsymbol{\\Sigma} \\mathbf{w}}$ |

Donde $\\mathbf{w}$ es el vector de pesos, $\\boldsymbol{\\mu}$ es el vector de retornos medios
diarios y $\\boldsymbol{\\Sigma}$ es la **matriz de covarianzas** anualizada.

La clave del modelo: si los activos tienen correlación baja o negativa,
$\\mathbf{w}^T \\boldsymbol{\\Sigma} \\mathbf{w}$ es **menor** que la suma ponderada
de las varianzas individuales. Eso es la diversificación.

---

#### Frontera Eficiente

La frontera eficiente es el conjunto de portafolios que:
- Para un **retorno dado**, minimizan la volatilidad, o equivalentemente:
- Para una **volatilidad dada**, maximizan el retorno.

Se calcula resolviendo el problema de optimización cuadrática:

$$\\min_{\\mathbf{w}} \\; \\mathbf{w}^T \\boldsymbol{\\Sigma} \\mathbf{w} \\quad
\\text{s.a.} \\; \\mathbf{w}^T \\boldsymbol{\\mu} = r^*, \\;
\\sum_i w_i = 1, \\; w_i \\geq 0$$

para cada nivel de retorno objetivo $r^*$ en el rango posible.

---

#### Optimización SLSQP

El algoritmo **Sequential Least Squares Programming** (SLSQP) de `scipy.optimize`
resuelve dos problemas:

**Máximo Sharpe Ratio:**
$$\\max_{\\mathbf{w}} \\; S = \\frac{E[r_p] - r_f}{\\sigma_p}
\\quad \\text{s.a.} \\; \\sum_i w_i = 1, \\; w_i \\geq 0$$

**Mínima Varianza:**
$$\\min_{\\mathbf{w}} \\; \\sigma_p
\\quad \\text{s.a.} \\; \\sum_i w_i = 1, \\; w_i \\geq 0$$

SLSQP converge en decenas de iteraciones y garantiza la solución global
para problemas convexos como el de mínima varianza.

---

#### Sharpe Ratio

$$S = \\frac{E[r_p] - r_f}{\\sigma_p}$$

Mide el retorno en exceso sobre la tasa libre de riesgo $r_f$ por unidad de volatilidad.
Un Sharpe > 1 indica que el activo compensa holgadamente el riesgo; < 0.5 sugiere
una mala relación retorno/riesgo.

#### Sortino Ratio

$$\\text{Sortino} = \\frac{E[r_p] - r_f}{\\sigma_{\\text{neg}}}$$

Variante del Sharpe que penaliza solo la volatilidad negativa (downside deviation),
siendo más adecuado para distribuciones asimétricas.

---

#### Métricas de Riesgo

| Métrica | Definición |
|---------|-----------|
| **VaR 95%** | Percentil 5% de retornos diarios: pérdida máxima esperada en el 95% de los días |
| **CVaR 95%** | Pérdida promedio en el 5% peor de los días (Expected Shortfall) |
| **Max Drawdown** | Caída máxima desde un pico histórico: $(P_t - \\max_{s \\leq t} P_s) / \\max_{s \\leq t} P_s$ |
| **Calmar Ratio** | Retorno acumulado / \\|Max Drawdown\\| |

---

#### Simulación Monte Carlo

Se generan $N$ portafolios con pesos aleatorios ($\\sum w_i = 1$, $w_i \\geq 0$)
y se calcula el retorno, volatilidad y Sharpe de cada uno. La nube resultante
permite visualizar el espacio de posibilidades y aproximar la frontera eficiente
antes de la optimización exacta.

---

#### Librerías utilizadas

| Librería | Versión mínima | Uso |
|---------|---------------|-----|
| `streamlit` | 1.35 | Interfaz web |
| `yfinance` | 0.2.40 | Descarga de datos |
| `pandas` | 2.0 | Series temporales y DataFrames |
| `numpy` | 1.26 | Álgebra matricial |
| `scipy` | 1.12 | Optimización SLSQP |
| `plotly` | 5.20 | Gráficos interactivos |
| `anthropic` / `openai` | — | Interpretación IA (opcional) |

---

*Proyecto desarrollado para el Challenge de Economía Computacional — UADE 2025.*
*Todos los resultados son exclusivamente educativos y no constituyen asesoramiento financiero.*
""")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<p style="text-align:center;color:#B0BEC5;font-size:.75rem;">
  Optimizador de Portafolio Interactivo · Economía Computacional · UADE 2025 ·
  Datos: Yahoo Finance · Solo uso educativo · No constituye asesoramiento financiero
</p>
""", unsafe_allow_html=True)
