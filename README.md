# 📈 Optimizador de Portafolio Interactivo

**Materia:** Economía Computacional — UADE  
**Challenge:** Inteligencia Artificial aplicada a Economía

---

## Estructura del proyecto

```
portfolio_challenge/
├── app.py                  # Interfaz Streamlit (6 tabs)
├── data_loader.py          # Descarga yfinance + catálogo activos
├── portfolio_model.py      # Markowitz: MC, SLSQP, frontera eficiente
├── plots.py                # Gráficos Plotly
├── utils.py                # Métricas financieras y FODA
├── ai_interpreter.py       # Claude/GPT + fallback por reglas
├── requirements.txt
├── notebook_desarrollo.ipynb
├── informe_3_paginas.md
├── guion_presentacion.md
└── README.md
```

---

## Instalación local

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr la app
streamlit run app.py
```

Abre en: http://localhost:8501

---

## Subir a Streamlit Cloud

1. Subir la carpeta a un repositorio GitHub (público o privado)
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar el repo, seleccionar `app.py` como main file
4. Hacer clic en **Deploy**

**Variables de entorno opcionales** (para IA):
En Streamlit Cloud → Settings → Secrets:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY    = "sk-..."
```

---

## Cómo usar la app

1. **Sidebar** → elegí activos por grupo (BYMA, ADRs, internacionales, ETFs)
2. **Período** → seleccioná rango de fechas
3. **Parámetros** → tasa libre de riesgo, monto inicial, short on/off
4. **▶ EVALUAR RIESGO** → ejecutar el análisis
5. Explorá los resultados en los 6 tabs

---

## Activos disponibles

### 🇦🇷 Acciones BYMA (en ARS)
`YPFD.BA`, `GGAL.BA`, `PAMP.BA`, `TXAR.BA`, `TGSU2.BA`,  
`CEPU.BA`, `ALUA.BA`, `LOMA.BA`, `BMA.BA`, `SUPV.BA`

### 🇦🇷 ADRs argentinos (NYSE, en USD)
`YPF`, `GGAL`, `PAM`, `MELI`, `GLOB`, `BMA`, `CEPU`, `BIOX`

### 🌎 Internacionales
`AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, `META`, `TSLA`, `KO`, `JPM`, `XOM`

### 📦 ETFs y Bonos
`SPY`, `QQQ`, `EWZ`, `GLD`, `TLT`, `EEM`, `EMB`

---

## Errores frecuentes

### Ticker no encontrado
```
⚠️ No se pudieron descargar: YPFD.BA — excluidos del análisis.
```
- Los tickers `.BA` a veces están suspendidos o tienen problemas de liquidez
- Probá el ADR equivalente en NYSE (ej: `GGAL` en vez de `GGAL.BA`)
- Ajustá el período si el activo es reciente

### "Menos de 2 activos con datos válidos"
- Seleccioná activos de distintos grupos
- Ampliá el período histórico

### Optimización no convergió
- Aumentá a 3+ activos
- Usá período mínimo de 1 año
- Reducí el peso mínimo a 0%

### Error de importación de anthropic/openai
```bash
pip install anthropic   # para Claude
pip install openai      # para GPT
```
La app funciona completamente sin API (análisis automático por reglas).

---

## Conexión con el notebook

`notebook_desarrollo.ipynb` contiene el mismo análisis que la app pero en formato académico secuencial. La relación es:

| Notebook | App |
|----------|-----|
| Sección 3: Descarga de datos | `data_loader.py` |
| Sección 5: Métricas individuales | `utils.py` |
| Sección 7-8: Monte Carlo + SLSQP | `portfolio_model.py` |
| Sección 12: Gráficos | `plots.py` |
| Tab Evaluación / Comparador | `app.py` tabs 0-1 |

Para presentar en clase: mostrá primero el notebook para explicar la teoría paso a paso, luego la app para la demo interactiva.

---

> ⚠️ **Uso educativo.** No constituye asesoramiento financiero.
