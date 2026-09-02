# Sistema de Predicción de Fallas PLC — Agente Adaptable

Sistema de 4 agentes para predicción de fallas en componentes industriales.
Acepta datos en cualquier formato y se adapta automáticamente al volumen y
calidad disponible.

## Estructura del proyecto

```
plc_agent/
├── orchestrator.py          ← Punto de entrada principal
├── core/
│   ├── config.py            ← Configuración y modelos de datos
│   └── predictor.py         ← AdaptiveFailurePredictor (ML)
├── agents/
│   ├── agent1_analyzer.py   ← Detecta schema y carga cualquier fuente
│   ├── agent2_planner.py    ← Decide estrategia de modelo
│   ├── agent3_executor.py   ← Ejecuta entrenamiento y predicción
│   └── agent4_reporter.py   ← Genera reporte HTML
```

## Instalación

```bash
pip install pandas scikit-learn sentence-transformers joblib sqlalchemy openpyxl
```

## Uso básico

```python
from orchestrator import PLCFailureAgentSystem

system = PLCFailureAgentSystem()
report_path = system.run("datos.xlsx")
```

## Uso con datos en diferentes formatos

```python
# CSV con separador automático
system.run("logs.csv")

# JSON
system.run("registros.json")

# DataFrame de pandas
import pandas as pd
df = pd.read_excel("datos.xlsx")
system.run(df)

# Base de datos SQL
system.run(
    source=("postgresql://user:pw@localhost/plc_db",
            "SELECT * FROM fault_logs WHERE fecha >= '2023-01-01'")
)
```

## Columnas con nombres diferentes

```python
system.run(
    source="mis_datos.csv",
    column_map={
        "col_component":       "TAG_ID",        # columna de componente
        "col_timestamp_start": "FECHA_INICIO",  # timestamp de inicio
        "col_timestamp_end":   "FECHA_FIN",     # timestamp de fin
        "col_panel":           "LINEA",         # panel / línea
        "col_log_text":        "DESCRIPCION",   # texto del log
    }
)
```

## Configuración avanzada

```python
from core.config import AgentConfig
from orchestrator import PLCFailureAgentSystem

config = AgentConfig(
    umbral_critico    = 20,   # minutos para considerar falla crítica
    ventana_dias      = 7,    # horizonte de predicción
    default_n_eventos = 10,   # ventana de historial por defecto
    min_eventos_modelo_local = 30,  # mínimo para modelo local
    top_n_ranking     = 25,   # componentes en el ranking
    output_dir        = "resultados",
)

system = PLCFailureAgentSystem(config=config)
report = system.run("datos.xlsx", client_id="empresa_abc")
```

## Actualización incremental (sin reentrenar)

```python
# Primer entrenamiento
system.run("datos_historicos.xlsx", client_id="cliente_1")

# Semanas después, con datos nuevos
system.update("datos_semana_nueva.xlsx", client_id="cliente_1")
```

## CLI

```bash
python orchestrator.py datos.xlsx \
    --client empresa_xyz \
    --umbral 20 \
    --ventana 7 \
    --col-component TAG_ID \
    --col-panel LINEA \
    --output resultados/
```

## Arquitectura de decisión del Agente 2

| Calidad de datos | Estrategia |
|---|---|
| < 30 eventos | Heurístico (frecuencia + prior por categoría) |
| 30–100 eventos | Modelo global RF (generaliza a componentes nuevos) |
| 100–500 eventos | Híbrido (global + modelos locales donde hay datos) |
| > 500 eventos, >90 días | Local RF dedicado por componente |

## Qué hace cada agente

**Agente 1 — Analizador:**
- Detecta automáticamente qué columna es el timestamp, el componente, el texto del log, etc.
- Soporta Excel, CSV, JSON, Parquet, SQL
- Calcula calidad del dataset y advierte sobre problemas

**Agente 2 — Planificador:**
- Decide qué estrategia de modelo usar según calidad y volumen
- Calibra umbral_critico y n_eventos basado en la distribución real
- Determina si se necesita ajuste de hiperparámetros

**Agente 3 — Ejecutor:**
- Preprocesa, genera embeddings, construye features
- Entrena modelo global y locales según el plan
- Produce ranking de riesgo con incertidumbre explícita

**Agente 4 — Redactor:**
- Genera reporte HTML imprimible como PDF
- Incluye ranking, métricas, importancia de features, advertencias
- Recomendaciones automáticas en lenguaje natural
