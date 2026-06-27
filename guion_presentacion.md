# Guión de Presentación Oral — 12 minutos
## Optimizador de Portafolio Interactivo · Economía Computacional

---

## ESTRUCTURA (12 min total)

| Bloque | Tiempo | Contenido |
|--------|--------|-----------|
| Introducción | 1.5 min | Problema, hook, relevancia |
| Marco teórico | 2 min | Markowitz en 90 segundos |
| Demo de la app | 5 min | Recorrido live por tabs |
| Lógica en Python | 2 min | Módulos, optimización, IA |
| Limitaciones y reflexión | 1.5 min | Honestidad académica |

---

## BLOQUE 1 — INTRODUCCIÓN (1.5 min)

> *"Supongamos que tienen $10.000 dólares para invertir. Tienen acciones de YPF, Apple y MercadoLibre. ¿Cuánto ponen en cada una?"*

La respuesta intuitiva es "partes iguales" — un tercio cada una. Pero eso no es necesariamente óptimo. Depende de cuánto rinde cada activo, cuánto oscila y — lo más importante — cómo se mueven juntos.

Eso es exactamente lo que resuelve esta aplicación: dado un conjunto de activos, encontrar la combinación de pesos que **maximiza el retorno por unidad de riesgo**, usando datos reales de mercado y matemática de optimización.

El problema es relevante porque:
- Argentina tiene un mercado de capitales activo con ADRs en NYSE
- El inversor argentino enfrenta diversificación cambiaria (ARS vs USD)
- La IA puede hacer este análisis accesible para cualquier persona

---

## BLOQUE 2 — MARCO TEÓRICO (2 min)

### ¿Qué es Markowitz?

Harry Markowitz publicó en 1952 la **Teoría Moderna de Portafolios**. La idea central:

> "No es suficiente que un activo individual tenga buen retorno. Lo que importa es su contribución al riesgo y retorno del portafolio **total**."

Dos activos con retornos similares pero correlación baja, combinados, producen un portafolio con **menor riesgo que cualquiera de los dos por separado**. Eso es diversificación real.

### El modelo en tres pasos

1. **Medir** → retorno esperado, varianza, correlaciones entre activos
2. **Simular** → generar miles de combinaciones posibles (Monte Carlo)
3. **Optimizar** → encontrar la combinación que maximiza el Sharpe Ratio

### Frontera eficiente

La "frontera eficiente" es el conjunto de portafolios que dominan a todos los demás: para cualquier nivel de riesgo dado, ningún otro portafolio ofrece más retorno.

*[Mostrar slide o gráfico de la frontera]*

---

## BLOQUE 3 — DEMO EN VIVO (5 min)

### Paso 1 — Configuración (30 seg)

*[Abrir la app en el navegador]*

- Sidebar izquierdo: selecciono activos argentinos: YPF, GGAL, PAM
- Activos internacionales: AAPL, NVDA, SPY
- Período: 3 años
- Tasa libre de riesgo: 5%
- Clic en **ANALIZAR PORTAFOLIO**

---

### Paso 2 — Tab Resumen (45 seg)

*[Tab Resumen]*

Acá ven los KPIs del portafolio óptimo:
- Retorno anual esperado: **X%**
- Volatilidad: **X%**
- Sharpe Ratio: **X.XX** (lo que gana por unidad de riesgo)

Y la tabla con métricas individuales: cada fila es un activo, cada columna una métrica.

---

### Paso 3 — Tab Comparación (45 seg)

*[Tab Comparación → Precios Base 100]*

Este gráfico normaliza todos los precios a 100 al inicio del período. Permite comparar rendimientos sin que el precio absoluto interfiera. Las líneas sólidas son activos argentinos, las punteadas son internacionales.

*[Subtab Retorno vs Riesgo]*

Acá el eje X es volatilidad, el eje Y es retorno. Los activos ideales están arriba a la izquierda. El tamaño de la burbuja es el Sharpe Ratio.

---

### Paso 4 — Tab Markowitz (2 min)

*[Tab Markowitz — Frontera Eficiente]*

Esta es la "joya" del proyecto. Cada punto gris es uno de los 5.000 portafolios simulados, coloreado por Sharpe Ratio. La curva azul es la **frontera eficiente** calculada con optimización SLSQP.

- ⭐ **Estrella dorada** = portafolio de máximo Sharpe
- 🩵 **Estrella teal** = portafolio de mínima varianza
- ◆ **Diamante gris** = portafolio equiponderado (1/N)

La línea punteada dorada es la **Capital Market Line** — la combinación de la cartera óptima con activo libre de riesgo.

*[Bajar a los tres bloques de pesos]*

Acá se muestran los pesos exactos de cada portafolio. El máximo Sharpe concentra más en activos con mejor relación retorno/riesgo; el de mínima varianza privilegia activos de baja volatilidad.

---

### Paso 5 — Tab Riesgo (45 seg)

*[Tab Riesgo → Drawdown]*

El drawdown muestra cuánto cayó cada activo desde su pico histórico. Un drawdown de -40% significa que en algún momento el activo valía 40% menos que su máximo anterior.

*[Tab Riesgo → Correlaciones]*

El heatmap muestra cómo se correlacionan los activos. Verde = correlación positiva alta (se mueven juntos = menos diversificación). Rojo = correlación negativa (se mueven en sentido opuesto = diversificación ideal).

---

### Paso 6 — Tab IA (30 seg)

*[Tab IA → Generar Interpretación]*

La aplicación puede interpretar los resultados en lenguaje natural. Si hay una API de Claude o GPT configurada, la usa. Si no, genera automáticamente una interpretación por reglas.

*[Mostrar el texto generado]*

---

## BLOQUE 4 — LÓGICA EN PYTHON (2 min)

### Arquitectura modular

```
app.py          → interfaz Streamlit (solo UI)
data_loader.py  → yfinance, validación de tickers
portfolio_model.py → Markowitz: Monte Carlo, SLSQP, frontera
plots.py        → todos los gráficos con Plotly
utils.py        → métricas, VaR, FODA, rankings
ai_interpreter.py → Claude API / GPT / fallback por reglas
```

### Punto técnico clave: la optimización

```python
# Máximo Sharpe con SLSQP (scipy.optimize)
def neg_sharpe(w):
    return -(retorno(w) - rf) / volatilidad(w)

resultado = minimize(
    neg_sharpe, w0,
    method="SLSQP",
    bounds=[(0, 1)] * n,   # sin cortos
    constraints=[{"type": "eq", "fun": lambda w: sum(w) - 1}]
)
```

En lugar de probar millones de combinaciones a mano, el algoritmo navega el espacio de pesos de forma eficiente, garantizando convergencia al óptimo global.

### Librerías usadas del curso

- `pandas` y `numpy` → procesamiento de datos
- `scipy.optimize` → optimización SLSQP
- `plotly` → visualizaciones interactivas
- `streamlit` → interfaz web

---

## BLOQUE 5 — LIMITACIONES Y REFLEXIÓN (1.5 min)

### Limitaciones honestas

1. **El modelo mira hacia atrás.** Optimiza sobre datos pasados; el futuro puede ser diferente.
2. **Supone normalidad.** Los retornos reales tienen "colas pesadas" — los crashes son más frecuentes de lo que el modelo predice.
3. **Las correlaciones cambian.** En momentos de crisis, todo cae junto — justo cuando la diversificación más se necesita.
4. **No considera costos.** Comisiones y spreads afectan los retornos reales.

### Reflexión final

> *"Markowitz no nos dice qué va a pasar. Nos dice cuál es la mejor decisión dado lo que sabemos."*

Esta aplicación muestra cómo Python puede convertir un modelo matemático de 1952 en una herramienta accesible para cualquier inversor. La combinación de datos reales, optimización numérica e IA es exactamente el tipo de tecnología que está transformando el sector financiero.

---

*⚠️ Todos los resultados son exclusivamente educativos. No constituyen asesoramiento financiero.*
