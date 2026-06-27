"""
ai_interpreter.py
=================
Módulo de interpretación inteligente de resultados del portafolio.

Modo 1 (con API): llama a Claude (Anthropic) o GPT (OpenAI) para generar
                  un análisis en lenguaje natural enriquecido.
Modo 2 (fallback): genera interpretación automática basada en reglas
                   programadas. Siempre funciona, sin API.

La app funciona completamente sin ninguna API configurada.
"""

import os
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE APIs DISPONIBLES
# ─────────────────────────────────────────────────────────────────────────────

def detectar_api_disponible() -> str:
    """
    Detecta qué API está disponible según las variables de entorno.

    Returns:
        'anthropic' | 'openai' | 'ninguna'
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ninguna"


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def _construir_prompt(
    tickers: list[str],
    metricas_dict: dict,
    portafolio_ms: dict,
    portafolio_mv: dict,
    periodo: str,
    tasa_libre_riesgo: float,
) -> str:
    """
    Arma el prompt para enviar a la API de IA con los datos del análisis.
    """
    def fmt_pct(v):
        return f"{v * 100:.1f}%" if v is not None else "N/D"

    def fmt_num(v):
        return f"{v:.2f}" if v is not None else "N/D"

    # Resumen de activos
    lineas_activos = []
    for t in tickers:
        m = metricas_dict.get(t, {})
        lineas_activos.append(
            f"  - {t}: Retorno anual {fmt_pct(m.get('Retorno Anualizado'))}, "
            f"Volatilidad {fmt_pct(m.get('Volatilidad Anualizada'))}, "
            f"Sharpe {fmt_num(m.get('Sharpe Ratio'))}, "
            f"Max Drawdown {fmt_pct(m.get('Max Drawdown'))}"
        )

    # Portafolio Máximo Sharpe
    pesos_ms = portafolio_ms.get("pesos", {})
    pesos_ms_str = ", ".join([f"{t}: {v:.1%}" for t, v in pesos_ms.items()])

    # Portafolio Mínima Varianza
    pesos_mv = portafolio_mv.get("pesos", {})
    pesos_mv_str = ", ".join([f"{t}: {v:.1%}" for t, v in pesos_mv.items()])

    prompt = f"""Sos un analista financiero experto en teoría de portafolios de Markowitz.
Analizá los siguientes resultados de un optimizador de portafolio educativo y generá
una interpretación clara, concisa y económicamente fundamentada en español.

PERÍODO ANALIZADO: {periodo}
TASA LIBRE DE RIESGO: {fmt_pct(tasa_libre_riesgo)}

ACTIVOS ANALIZADOS:
{chr(10).join(lineas_activos)}

PORTAFOLIO DE MÁXIMO SHARPE:
  Pesos: {pesos_ms_str}
  Retorno anual: {fmt_pct(portafolio_ms.get('retorno_anual'))}
  Volatilidad: {fmt_pct(portafolio_ms.get('volatilidad'))}
  Sharpe Ratio: {fmt_num(portafolio_ms.get('sharpe'))}
  Max Drawdown: {fmt_pct(portafolio_ms.get('max_drawdown'))}

PORTAFOLIO DE MÍNIMA VARIANZA:
  Pesos: {pesos_mv_str}
  Retorno anual: {fmt_pct(portafolio_mv.get('retorno_anual'))}
  Volatilidad: {fmt_pct(portafolio_mv.get('volatilidad'))}
  Sharpe Ratio: {fmt_num(portafolio_mv.get('sharpe'))}
  Max Drawdown: {fmt_pct(portafolio_mv.get('max_drawdown'))}

Generá una interpretación estructurada con estas secciones:
1. **Resumen ejecutivo** (2-3 oraciones sobre el conjunto de activos)
2. **Activos destacados** (quién lideró, quién fue más volátil, quién tuvo mejor Sharpe)
3. **Portafolio óptimo recomendado** (entre máximo Sharpe y mínima varianza, con justificación)
4. **Diversificación** (¿el portafolio está bien diversificado? ¿hay concentración?)
5. **Advertencia educativa** (recordar que es análisis histórico, no asesoramiento financiero)

Sé directo. Usá lenguaje accesible pero técnico. Máximo 300 palabras."""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# LLAMADAS A APIS
# ─────────────────────────────────────────────────────────────────────────────

def _llamar_anthropic(prompt: str, api_key: str) -> str:
    """Llama a la API de Anthropic (Claude)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        mensaje = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return mensaje.content[0].text
    except ImportError:
        return _error_import("anthropic", "pip install anthropic")
    except Exception as e:
        return f"Error al llamar a Claude API: {str(e)}"


def _llamar_openai(prompt: str, api_key: str) -> str:
    """Llama a la API de OpenAI (GPT-4o-mini)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        respuesta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        return respuesta.choices[0].message.content
    except ImportError:
        return _error_import("openai", "pip install openai")
    except Exception as e:
        return f"Error al llamar a OpenAI API: {str(e)}"


def _error_import(libreria: str, cmd: str) -> str:
    return (
        f"La librería '{libreria}' no está instalada. "
        f"Instalá con: `{cmd}`. "
        f"Usando interpretación automática por reglas."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: INTERPRETACIÓN POR REGLAS
# ─────────────────────────────────────────────────────────────────────────────

def _interpretacion_por_reglas(
    tickers: list[str],
    metricas_dict: dict,
    portafolio_ms: dict,
    portafolio_mv: dict,
    periodo: str,
    tasa_libre_riesgo: float,
) -> str:
    """
    Genera una interpretación basada en reglas programadas.
    No requiere API. Siempre disponible.
    """
    lineas = []

    # ── Resumen ejecutivo ────────────────────────────────────────────────────
    n = len(tickers)
    rets = {t: metricas_dict[t].get("Retorno Anualizado", 0) for t in tickers if t in metricas_dict}
    vols = {t: metricas_dict[t].get("Volatilidad Anualizada", 0) for t in tickers if t in metricas_dict}
    shrp = {t: metricas_dict[t].get("Sharpe Ratio", 0) for t in tickers if t in metricas_dict}

    if not rets:
        return "No hay datos suficientes para generar una interpretación."

    mejor_ret  = max(rets, key=rets.get)
    peor_ret   = min(rets, key=rets.get)
    menor_vol  = min(vols, key=vols.get)
    mayor_vol  = max(vols, key=vols.get)
    mejor_shr  = max(shrp, key=shrp.get)

    lineas.append("## 📊 Resumen Ejecutivo\n")
    lineas.append(
        f"Se analizaron **{n} activos** durante el período {periodo}. "
        f"El activo con mayor retorno anualizado fue **{mejor_ret}** "
        f"({rets[mejor_ret]:.1%}), mientras que **{peor_ret}** registró el menor "
        f"({rets[peor_ret]:.1%}). La volatilidad varió entre {vols[menor_vol]:.1%} "
        f"({menor_vol}) y {vols[mayor_vol]:.1%} ({mayor_vol}).\n"
    )

    # ── Activos destacados ───────────────────────────────────────────────────
    lineas.append("\n## 🏆 Activos Destacados\n")
    lineas.append(f"- **Mayor retorno**: {mejor_ret} ({rets[mejor_ret]:.1%} anual)")
    lineas.append(f"- **Menor volatilidad**: {menor_vol} ({vols[menor_vol]:.1%} anual)")
    lineas.append(f"- **Mejor Sharpe Ratio**: {mejor_shr} ({shrp[mejor_shr]:.2f})")

    activos_neg = [t for t in rets if rets[t] < 0]
    if activos_neg:
        lineas.append(f"- **Retorno negativo en el período**: {', '.join(activos_neg)}")

    # ── Portafolio óptimo ────────────────────────────────────────────────────
    lineas.append("\n## 🎯 Portafolio Óptimo\n")
    ms_shr = portafolio_ms.get("sharpe", 0) or 0
    mv_shr = portafolio_mv.get("sharpe", 0) or 0
    ms_ret = portafolio_ms.get("retorno_anual", 0) or 0
    mv_vol = portafolio_mv.get("volatilidad", 0) or 0

    if portafolio_ms.get("pesos"):
        pesos_ms = portafolio_ms["pesos"]
        pesos_str = ", ".join([f"**{t}**: {v:.1%}" for t, v in
                                sorted(pesos_ms.items(), key=lambda x: -x[1])])
        lineas.append(
            f"**Portafolio de Máximo Sharpe** (Sharpe: {ms_shr:.2f}, "
            f"Retorno: {ms_ret:.1%}):\n{pesos_str}\n"
        )

    if portafolio_mv.get("pesos"):
        pesos_mv = portafolio_mv["pesos"]
        pesos_str = ", ".join([f"**{t}**: {v:.1%}" for t, v in
                                sorted(pesos_mv.items(), key=lambda x: -x[1])])
        lineas.append(
            f"**Portafolio de Mínima Varianza** (Volatilidad: {mv_vol:.1%}, "
            f"Sharpe: {mv_shr:.2f}):\n{pesos_str}\n"
        )

    if ms_shr >= mv_shr:
        lineas.append(
            "El **portafolio de Máximo Sharpe** ofrece la mejor relación "
            "retorno/riesgo y es preferible para inversores con tolerancia moderada al riesgo."
        )
    else:
        lineas.append(
            "El **portafolio de Mínima Varianza** es más conservador y preferible "
            "para inversores con alta aversión al riesgo o en contextos de alta incertidumbre."
        )

    # ── Diversificación ──────────────────────────────────────────────────────
    lineas.append("\n## 🔗 Diversificación\n")
    if portafolio_ms.get("pesos"):
        pesos_arr = list(portafolio_ms["pesos"].values())
        max_peso  = max(pesos_arr)
        if max_peso > 0.5:
            ticker_conc = max(portafolio_ms["pesos"], key=portafolio_ms["pesos"].get)
            lineas.append(
                f"⚠️ Alta concentración: **{ticker_conc}** representa el {max_peso:.1%} del portafolio. "
                "Considerar aumentar el número de activos o aplicar un peso máximo."
            )
        elif max_peso > 0.3:
            lineas.append("El portafolio muestra concentración moderada. La diversificación es aceptable.")
        else:
            lineas.append("✅ El portafolio está bien diversificado entre los activos seleccionados.")

    # ── Advertencia ──────────────────────────────────────────────────────────
    lineas.append("\n---\n")
    lineas.append(
        "⚠️ **Advertencia educativa**: Este análisis se basa en datos históricos del período "
        f"seleccionado. El desempeño pasado **no garantiza resultados futuros**. "
        "Este resultado es exclusivamente educativo y **no constituye asesoramiento financiero**. "
        "Consultá a un profesional antes de tomar decisiones de inversión."
    )

    return "\n".join(lineas)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def generar_interpretacion(
    tickers: list[str],
    metricas_dict: dict,
    portafolio_ms: dict,
    portafolio_mv: dict,
    periodo: str,
    tasa_libre_riesgo: float = 0.0,
    api_key_anthropic: Optional[str] = None,
    api_key_openai: Optional[str] = None,
) -> tuple[str, str]:
    """
    Genera la interpretación del portafolio.

    Prioridad: Anthropic → OpenAI → Reglas (fallback)

    Parameters:
        tickers             : lista de tickers analizados
        metricas_dict       : dict {ticker: {métrica: valor}}
        portafolio_ms       : dict con métricas del portafolio de máximo Sharpe
        portafolio_mv       : dict con métricas del portafolio de mínima varianza
        periodo             : string descriptivo del período
        tasa_libre_riesgo   : en decimal
        api_key_anthropic   : clave API de Anthropic (opcional)
        api_key_openai      : clave API de OpenAI (opcional)

    Returns:
        (texto_interpretacion: str, fuente: str)
        fuente ∈ {'Claude (Anthropic)', 'GPT (OpenAI)', 'Análisis automático'}
    """
    # Convertir metricas a dict plano si viene como DataFrame
    if hasattr(metricas_dict, "to_dict"):
        metricas_dict = metricas_dict.T.to_dict()

    # Intentar con Anthropic
    clave_ant = api_key_anthropic or os.environ.get("ANTHROPIC_API_KEY", "")
    if clave_ant:
        prompt = _construir_prompt(
            tickers, metricas_dict, portafolio_ms, portafolio_mv, periodo, tasa_libre_riesgo
        )
        resultado = _llamar_anthropic(prompt, clave_ant)
        if not resultado.startswith("Error") and not resultado.startswith("La librería"):
            return resultado, "Claude (Anthropic)"

    # Intentar con OpenAI
    clave_oai = api_key_openai or os.environ.get("OPENAI_API_KEY", "")
    if clave_oai:
        prompt = _construir_prompt(
            tickers, metricas_dict, portafolio_ms, portafolio_mv, periodo, tasa_libre_riesgo
        )
        resultado = _llamar_openai(prompt, clave_oai)
        if not resultado.startswith("Error") and not resultado.startswith("La librería"):
            return resultado, "GPT (OpenAI)"

    # Fallback por reglas
    resultado = _interpretacion_por_reglas(
        tickers, metricas_dict, portafolio_ms, portafolio_mv, periodo, tasa_libre_riesgo
    )
    return resultado, "Análisis automático"
