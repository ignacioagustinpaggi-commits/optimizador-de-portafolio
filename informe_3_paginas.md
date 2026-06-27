# Optimizador de Portafolio Interactivo con Activos Argentinos e Internacionales
## Informe de Challenge — Economía Computacional

**Materia:** Economía Computacional | **Universidad:** UADE  
**Instancia:** Challenge — IA aplicada a Economía | **Año:** 2025

---

## 1. Planteamiento del Problema

La gestión eficiente de inversiones es uno de los problemas centrales de las finanzas modernas. Un inversor que combina activos argentinos y del exterior enfrenta múltiples desafíos simultáneos: evaluar el riesgo individual de cada activo, estimar la correlación entre ellos, y determinar cuánto capital asignar a cada uno para maximizar el retorno por unidad de riesgo asumido.

El problema es computacionalmente intensivo: para un portafolio de N activos, el espacio de combinaciones posibles de pesos es infinito. Sin herramientas adecuadas, el inversor no puede determinar objetivamente cuál es la composición óptima del portafolio. Esta brecha entre el problema teórico y la práctica cotidiana es el punto de partida del presente proyecto.

---

## 2. Objetivo del Ejercicio Aplicado

Desarrollar una aplicación web interactiva en Python que permita a cualquier usuario:

1. Seleccionar activos financieros argentinos (ADRs en NYSE) e internacionales
2. Descargar datos reales desde Yahoo Finance
3. Calcular métricas de riesgo y retorno para cada activo
4. Construir la frontera eficiente de Markowitz mediante simulación Monte Carlo y optimización SLSQP
5. Obtener los portafolios de máximo Sharpe Ratio y mínima varianza
6. Comparar grupos de activos (Argentina vs. exterior)
7. Obtener una interpretación en lenguaje natural mediante IA o análisis automático por reglas

---

## 3. Metodología

### 3.1 Fuente de datos

Se utilizó la librería `yfinance` para descargar precios de cierre ajustados desde Yahoo Finance. Los precios ajustados incorporan dividendos y splits, lo que permite una comparación correcta entre períodos. El rango de fechas es seleccionable por el usuario.

### 3.2 Cálculo de retornos

Se utilizaron **retornos logarítmicos diarios**: $r_t = \ln(P_t / P_{t-1})$

Los retornos logarítmicos son el estándar en finanzas cuantitativas porque son aditivos en el tiempo, se aproximan mejor a una distribución normal y evitan valores inferiores a -100%.

### 3.3 Métricas calculadas por activo

Para cada activo se calcularon:

| Métrica | Descripción |
|---------|-------------|
| Retorno anualizado | $\bar{r} \times 252$ |
| Volatilidad anualizada | $\sigma \times \sqrt{252}$ |
| Sharpe Ratio | $(E[r] - r_f) / \sigma$ |
| Sortino Ratio | $(E[r] - r_f) / \sigma_{neg}$ |
| Calmar Ratio | Retorno acumulado / \|Max Drawdown\| |
| Max Drawdown | Caída máxima desde un pico histórico |
| VaR histórico 95% | Percentil 5% de los retornos diarios |
| CVaR 95% | Media de retornos por debajo del VaR |

### 3.4 Optimización de Markowitz

Se implementaron tres enfoques complementarios:

**a) Simulación Monte Carlo:** Se generaron 10.000 portafolios con pesos aleatorios para visualizar el espacio retorno-riesgo y aproximar los portafolios óptimos.

**b) Optimización SLSQP:** Se utilizó el algoritmo Sequential Least Squares Programming de SciPy para encontrar con precisión matemática:
- **Máximo Sharpe Ratio**: $\max_w \frac{E[r_p] - r_f}{\sigma_p}$ sujeto a $\sum w_i = 1$, $w_i \geq 0$
- **Mínima Varianza**: $\min_w \sigma_p$ sujeto a $\sum w_i = 1$, $w_i \geq 0$

**c) Frontera eficiente:** Se calculó minimizando la varianza para 80 niveles distintos de retorno objetivo, trazando la curva completa de portafolios eficientes.

### 3.5 Módulo de IA

El componente de Inteligencia Artificial acepta opcionalmente claves de la API de Anthropic (Claude) u OpenAI (GPT-4o-mini). Si no se configura ninguna API, el módulo genera automáticamente una interpretación estructurada basada en reglas programadas, garantizando que la aplicación funcione en todo momento.

---

## 4. Resultados Principales

La aplicación genera los siguientes resultados de forma automática:

- **Tabla de métricas individuales** con gradiente de color por retorno y Sharpe
- **Gráfico de frontera eficiente** con nube Monte Carlo coloreada por Sharpe, Capital Market Line y portafolios óptimos marcados
- **Evolución histórica comparada** de los tres portafolios (base 100)
- **Donut charts de composición** para máximo Sharpe y mínima varianza
- **Heatmap de correlaciones** entre activos
- **Gráficos de drawdown** y distribución de retornos por activo
- **Comparación Argentina vs. Exterior** con métricas promedio por grupo
- **FODA financiero** orientativo por activo
- **Interpretación en lenguaje natural** del portafolio óptimo

Los resultados dependen del período seleccionado y los activos elegidos; no se presentan valores fijos ya que la aplicación calcula todo en tiempo real.

---

## 5. Limitaciones

1. **Normalidad**: el modelo asume retornos normales; los mercados reales presentan colas pesadas y asimetría
2. **Estacionariedad**: las correlaciones y varianzas varían en el tiempo; el modelo las trata como constantes
3. **Retrospección**: el optimizador usa datos pasados para construir el portafolio futuro
4. **Costos omitidos**: no se consideran comisiones, spreads bid-ask, ni impuestos sobre ganancias
5. **Riesgo cambiario**: los ADRs argentinos cotizan en USD; el riesgo ARS/USD no está modelado explícitamente

---

## 6. Conclusión

El proyecto demuestra la aplicación directa de la Teoría Moderna de Portafolios de Markowitz mediante herramientas de Python de uso estándar en la industria financiera. La combinación de `yfinance` para datos reales, `scipy` para optimización numérica, `plotly` para visualización interactiva y `streamlit` para la interfaz web produce una herramienta educativa completa, funcional y reproducible.

La incorporación de un módulo de IA que puede usar modelos de lenguaje (Claude, GPT) o funcionar de forma autónoma ilustra cómo la inteligencia artificial puede complementar el análisis cuantitativo tradicional, convirtiendo resultados numéricos en interpretaciones accesibles para usuarios sin formación técnica profunda.

---

*⚠️ Todos los resultados son exclusivamente educativos y no constituyen asesoramiento financiero.*
