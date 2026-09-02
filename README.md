# Framework Multi-Vertical de Mantenimiento Predictivo — VecTech

Framework de agentes para predicción de fallas y desgaste en entornos
industriales. Detecta automáticamente el dominio de los datos de entrada,
selecciona la estrategia de modelo adecuada, entrena o carga un modelo ya
entrenado, y genera un reporte HTML — todo sin configuración manual
obligatoria.

**¿Vas a extender o mantener este código, no solo usarlo?** Lee
[`ARCHITECTURE.md`](ARCHITECTURE.md) primero — explica el patrón de 4
agentes, la interfaz `BasePredictor`, el contrato de API pública, y por
qué se tomaron las decisiones de diseño clave. Este README se enfoca en
instalación y uso.

---

## 1. Instalación

### Desde un repositorio git (recomendado)

```bash
git clone <url-de-tu-repositorio> plc_agent
cd plc_agent

python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Instalación editable — instala el paquete Y sus dependencias, y
# habilita el comando `plc-agent` en la terminal
pip install -e ".[dev]"
```

`pip install -e .` (modo "editable") significa que los cambios que hagas
al código se reflejan de inmediato sin tener que reinstalar — ideal para
desarrollo. El extra `[dev]` agrega `pytest` para poder correr las
pruebas.

### Desde un .zip (si no usas git todavía)

```bash
cd plc_agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # o requirements-lock.txt para versiones exactas
```

### Verifica que todo quedó bien instalado

```bash
python test_smoke.py
```

Este script prueba las 3 variantes de predictor con datos sintéticos (no
necesita tus datos reales) y te dice exactamente qué falta si algo no
funciona. Debe terminar con:

```
✓ Entorno listo — puedes trabajar con datos reales.
```

**La primera vez que uses la vertical PLC** (ya sea `AdaptiveFailurePredictor`
o `LegacyPLCPredictor`) necesitas internet: el sistema descarga el modelo de
embeddings `all-MiniLM-L6-v2` (~91 MB) desde Hugging Face. Después de esa
descarga queda cacheado localmente y puedes trabajar sin conexión. La
vertical de **desgaste de herramienta no necesita internet en ningún
momento**.

---

## 2. Qué predictor usar

El framework tiene **tres predictores intercambiables**, todos accesibles
desde el mismo punto de entrada (`MultiVerticalAgentSystem`):

| Predictor | Cuándo usarlo | Entrena desde cero | Carga modelo ya entrenado |
|---|---|---|---|
| `AdaptiveFailurePredictor` | Fallas PLC — flujo normal del framework, con niveles LOCAL/GLOBAL/HEURÍSTICO e incertidumbre | `system.run(...)` | `system.predict_only(..., domain="plc_failure")` |
| `ToolWearPredictor` | Desgaste de herramienta (PHM 2010) — regresión + RUL | `system.run(...)` | `system.predict_only(..., domain="tool_wear")` |
| `LegacyPLCPredictor` | Fallas PLC — SOLO si ya tienes un `.pkl` entrenado con el repositorio `luisroberto-maker/PLC-failure-prediction-pipeline` | `predictor.fit(df)` directo (ver abajo) | `system.predict_only(..., domain="plc_failure", legacy=True)` |

Si no tienes un modelo legado ya entrenado, usa `AdaptiveFailurePredictor`
(la ruta por defecto) — es la variante más robusta.

---

## 3. Flujo A — Entrenar desde cero (fallas PLC o desgaste)

```python
from orchestrator import MultiVerticalAgentSystem

system = MultiVerticalAgentSystem()

# El sistema detecta automáticamente si son logs de eventos PLC
# o señales de sensor (desgaste de herramienta)
reporte = system.run("mis_datos.xlsx", client_id="cliente_a")
```

Esto entrena el modelo, lo guarda en `output/models/`, y genera el reporte
HTML en `output/reports/`. Si tus columnas tienen nombres distintos a los
esperados:

```python
reporte = system.run(
    source="datos.csv",
    client_id="cliente_a",
    column_map={
        "col_component":       "TAG_ID",
        "col_timestamp_start": "FECHA_INICIO",
        "col_panel":           "LINEA",
        "col_log_text":        "DESCRIPCION",
    },
)
```

---

## 4. Flujo B — Inferencia con un modelo ya entrenado (sin reentrenar)

### 4.1 Modelo entrenado con este mismo framework

```python
reporte = system.predict_only(
    source="datos_nuevos.xlsx",
    client_id="cliente_a",
    domain="plc_failure",                        # o "tool_wear"
    model_path="output/models/predictor_cliente_a.pkl",
)
```

### 4.2 Modelo de desgaste entrenado en Kaggle

Si entrenaste en Kaggle con el notebook de PHM 2010, descarga los 4 archivos
que genera (`model_rf.joblib`, `model_xgb.joblib`, `scaler.joblib`,
`metadata.json`) a una misma carpeta local:

```python
reporte = system.predict_only(
    source=df_nuevas_pasadas,
    client_id="taller_cnc",
    domain="tool_wear",
    artifacts_dir="mis_modelos/desgaste_v1/",
)
```

### 4.3 Modelo legado (repositorio `PLC-failure-prediction-pipeline`)

Si ya tienes un `.pkl` entrenado con ese repositorio (un estimador
scikit-learn "pelado", sin envoltorio):

```python
reporte = system.predict_only(
    source="datos_nuevos.xlsx",
    client_id="cliente_legacy",
    domain="plc_failure",
    legacy=True,
    model_path="models/modelo.pkl",
)
```

Esta ruta replica el comportamiento exacto del repositorio original —
mismo `MAPEO_FALLAS`, mismas features, mismo encoder — sin ninguna mejora
del framework aplicada encima.

---

## 5. Estructura del proyecto

Ambas verticales siguen exactamente el mismo patrón de 4 agentes
(Analizar → Planificar → Ejecutar → Reportar) — ver el diagrama de
arquitectura más abajo. La única excepción documentada es el modelo
legado (`legacy_plc_predictor.py`), que por diseño no tiene un Agente 2
propio: no hay ninguna decisión de estrategia que tomar en un flujo de
solo-inferencia sobre un modelo ya fijo.

```
plc_agent/
├── orchestrator.py                    ← Punto de entrada único (MultiVerticalAgentSystem)
├── test_smoke.py                      ← Prueba end-to-end con datos sintéticos
├── requirements.txt / requirements-lock.txt
├── tests/                             ← Suite de 44 pruebas automatizadas (pytest)
│
├── core/
│   ├── config.py                      ← AgentConfig y dataclasses compartidas
│   ├── base_predictor.py              ← Interfaz BasePredictor + ReportSchema
│   ├── predictor.py                   ← AdaptiveFailurePredictor (vertical PLC)
│   ├── tool_wear_predictor.py         ← ToolWearPredictor (vertical desgaste, PHM 2010)
│   └── legacy_plc_predictor.py        ← LegacyPLCPredictor (réplica exacta del repo original)
│
├── agents/
│   ├── domain_router.py               ← Detecta si los datos son PLC o desgaste
│   │
│   │   Vertical PLC                       Vertical desgaste de herramienta
│   ├── agent1_analyzer.py             │   agent1_tool_wear_loader.py       ← Agente 1: ingesta y schema
│   ├── agent2_planner.py              │   agent2_tool_wear_planner.py      ← Agente 2: decide estrategia
│   ├── agent3_executor.py             │   agent3_tool_wear_executor.py     ← Agente 3: entrena/predice/guarda
│   └── agent4_reporter.py             ← Agente 4: reporte HTML — COMPARTIDO por ambas verticales
│
└── output/                            ← Se crea automáticamente: models/, reports/
```

---

## 6. Columnas esperadas por vertical

**Fallas PLC** (Excel, CSV, JSON, Parquet o SQL):

| Columna | Obligatoria | Descripción |
|---|---|---|
| `START TIME` | Sí | Timestamp de inicio de la falla |
| `END TIME` | Recomendada | Timestamp de fin (si falta, se asume 1 min de duración) |
| `ADDRESS` | Sí | Identificador del componente |
| `FAIL COMMENT` | Recomendada | Texto del log — activa los embeddings semánticos |
| `PANEL` | Opcional | Línea o zona — si falta, se asigna un valor por defecto |

**Desgaste de herramienta** (PHM 2010): DataFrame con las 133 columnas de
features (`FEAT_COLS` en `core/tool_wear_predictor.py`) más una columna
`VB` (o `wear`/`desgaste`) con el desgaste medido en mm. Si tienes señales
crudas (fuerza, vibración, emisión acústica) en vez de features ya
extraídas, revisa `load_experiment_from_folder()` en el mismo archivo.

El nombre exacto de las columnas puede variar — el Agente 1 las detecta
automáticamente por similitud de nombre, o puedes forzarlas con
`column_map`.

---

## 7. Configuración avanzada

```python
from core.config import AgentConfig

config = AgentConfig(
    umbral_critico=20,          # minutos para considerar falla crítica
    ventana_dias=7,             # horizonte de predicción
    min_eventos_modelo_local=30,# mínimo de eventos para modelo local por componente
    top_n_ranking=25,           # componentes mostrados en el reporte
    output_dir="resultados",
)

system = MultiVerticalAgentSystem(config=config)
```

### Validación automática

`AgentConfig` está construido con [Pydantic](https://docs.pydantic.dev/) —
un valor inválido falla **al construir el objeto**, con un mensaje claro,
en vez de fallar minutos después a mitad del entrenamiento:

```python
AgentConfig(umbral_critico=-5)
# pydantic_core._pydantic_core.ValidationError: 1 validation error for AgentConfig
# umbral_critico
#   Input should be greater than 0 [type=greater_than, input_value=-5, input_type=int]
```

Un typo en el nombre de un parámetro también se detecta de inmediato
(`AgentConfig(umbral_critco=20)` con la "i" mal puesta falla en vez de
ignorarse en silencio). Si usas el CLI (`plc-agent` o
`python orchestrator.py`), un valor inválido en `--umbral` o `--ventana`
se reporta igual de claro, sin traceback:

```bash
$ plc-agent datos.xlsx --umbral -5
✗ Configuración inválida — revisa los parámetros que pasaste:
✗   • umbral_critico: Input should be greater than 0 (recibido: -5)
```

---

## 8. Logging y control de verbosidad

Todo el progreso (`[1/4] Agente 1 — Analizando datos...`, advertencias,
errores) se emite con el módulo `logging` estándar de Python, no con
`print()`. Esto significa que puedes controlar cuánto se muestra sin
tocar ni una línea de código.

### Desde la línea de comandos

```bash
plc-agent datos.xlsx --client cliente_a                     # nivel por defecto: INFO
plc-agent datos.xlsx --client cliente_a --log-level WARNING # solo advertencias y errores
plc-agent datos.xlsx --client cliente_a --log-level DEBUG   # máximo detalle
plc-agent datos.xlsx --client cliente_a --log-file corrida.log  # además, guarda todo a archivo
```

### Sin tocar código, con una variable de entorno

```bash
PLC_AGENT_LOG_LEVEL=WARNING python probar_modelo_real.py
```

### Desde Python, si usas el framework como biblioteca

```python
from core.logging_config import setup_logging

setup_logging(level="INFO")                          # consola limpia (por defecto)
setup_logging(level="DEBUG", log_file="debug.log")    # detalle completo, también a archivo
setup_logging(level="WARNING")                        # silencia el progreso, solo problemas
```

**Nota para los scripts propios** (`probar_modelo_real.py`,
`probar_modelo_kaggle.py`, `test_smoke.py`) ya llaman a `setup_logging()`
al inicio — si escribes un script nuevo que importe `orchestrator` o
`core/` directamente, agrega esa misma llamada al principio, o el
progreso no se mostrará (a diferencia de `print()`, `logging` no muestra
nada hasta que se configura explícitamente).

---

## 9. Pruebas automatizadas

El proyecto incluye una suite de 44 pruebas con `pytest` que cubre las tres
verticales/predictores y, en particular, **pruebas de regresión explícitas
para cada bug real encontrado durante el desarrollo** — para que no vuelvan
a aparecer silenciosamente en una actualización futura.

```bash
pip install -e ".[dev]"   # o pip install -r requirements.txt
python -m pytest
```

Debe terminar con algo como `44 passed`. Para correr solo un archivo o una
prueba específica:

```bash
python -m pytest tests/test_predictor_tool_wear.py -v
python -m pytest tests/test_predictor_tool_wear.py::TestComputeRUL -v
```

Las pruebas no requieren datos reales ni conexión a internet — usan datos
sintéticos y un modelo de embeddings simulado (`FakeEmbeddingModel` en
`tests/conftest.py`) para no depender de descargar el modelo real de
Hugging Face en cada corrida.

**Integración continua:** si subes este proyecto a GitHub, las pruebas
corren automáticamente en cada `push` y cada Pull Request gracias a
`.github/workflows/tests.yml` — sobre Python 3.11 y 3.12. No necesitas
acordarte de correr `pytest` manualmente antes de fusionar un cambio.

**Estructura:**

| Archivo | Qué cubre |
|---|---|
| `tests/conftest.py` | Fixtures compartidos: datos sintéticos PLC y desgaste, embeddings simulados |
| `tests/test_domain_router.py` | Detección automática de dominio |
| `tests/test_predictor_plc.py` | `AdaptiveFailurePredictor` — incluye el bug de `KeyError` y compatibilidad pandas 3.0 |
| `tests/test_predictor_tool_wear.py` | `ToolWearPredictor` — incluye el bug de RUL absurda y carga de artefactos Kaggle |
| `tests/test_predictor_legacy.py` | `LegacyPLCPredictor` — incluye el bug de `IndexError` por clase única |
| `tests/test_agent2_tool_wear_planner.py` | `ToolWearStrategyPlanner` — decisiones de estrategia (LOEO, ensamble) |
| `tests/test_orchestrator.py` | `MultiVerticalAgentSystem` — flujos `run()` y `predict_only()` completos |

Si modificas el código y una prueba empieza a fallar, es una señal real —
no la ignores sin entender por qué.

## 10. Versiones de dependencias

Hay dos archivos de requisitos, con propósitos distintos:

| Archivo | Cuándo usarlo |
|---|---|
| `requirements.txt` | Instalación normal — rangos flexibles con límites superiores de seguridad (ej. `pandas>=2.0,<4.0`) |
| `requirements-lock.txt` | Reproducir EXACTAMENTE un entorno ya probado, versión por versión — útil si algo funciona en una máquina y falla en otra |

```bash
# Instalación normal
pip install -r requirements.txt

# Reproducir el entorno exacto donde pasaron las 40 pruebas
pip install -r requirements-lock.txt
```

**Sobre la versión de Python:** el proyecto funciona en Python 3.9+, pero
Python 3.9 llegó a su fin de vida (sin parches de seguridad) el 31 de
octubre de 2025. Se recomienda Python 3.11 o 3.12 para instalaciones
nuevas.

**¿Por qué límites superiores en `requirements.txt`?** Ya nos pasó una vez
con pandas 3.0: una dependencia con un cambio mayor de versión introdujo 3
incompatibilidades distintas que tardamos varias corridas en diagnosticar
(ver `core/predictor.py` y `tests/test_predictor_plc.py`). Los límites
superiores evitan que una actualización automática de una librería rompa
el sistema sin aviso — si necesitas una versión más nueva de algo, se
actualiza el límite a propósito, después de probarlo.

## 11. Problemas comunes

| Síntoma | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: sentence_transformers` | Falta instalar dependencias | `pip install -r requirements.txt` |
| Error mencionando `huggingface.co` | Sin internet en la primera ejecución de la vertical PLC | Verifica conexión; solo hace falta una vez |
| `xgboost no disponible` (mensaje, no error) | xgboost no instalado — es opcional | El sistema usa solo Random Forest automáticamente, funciona igual |
| `KeyError` con columnas de features en la vertical de desgaste | El DataFrame no pasó por extracción de features | Usa `column_map={"domain": "tool_wear"}` o revisa que tenga las 133 columnas de `FEAT_COLS` |
| Predicciones distintas entre corridas con `LegacyPLCPredictor` | Comportamiento esperado — hereda el bug conocido del `LabelEncoder` y el PCA que se reajustan en cada llamada (documentado en `core/legacy_plc_predictor.py`) | Si te afecta en producción, considera migrar a `AdaptiveFailurePredictor` |
| No se ve NINGÚN progreso en consola al usar el framework como biblioteca | Falta llamar a `setup_logging()` — `logging` no muestra nada hasta que se configura, a diferencia de `print()` | Agrega `from core.logging_config import setup_logging; setup_logging()` al inicio de tu script |

## 12. Publicar este proyecto en un repositorio git

```bash
git init
git add .
git commit -m "Versión inicial: framework multi-vertical v1.8"

# Crea el repositorio remoto en GitHub/GitLab primero, luego:
git remote add origin <url-de-tu-repositorio>
git branch -M main
git push -u origin main
```

`.gitignore` ya está configurado para excluir el entorno virtual, los
modelos entrenados, los reportes generados y cualquier archivo de datos
(`.xlsx`, `.csv`) — así nunca subes por accidente datos de un cliente o
archivos pesados innecesarios al repositorio.

**Nota sobre `VERSION.txt` y `CHECKSUMS.txt`:** estos archivos se crearon
como solución temporal mientras el proyecto se compartía como `.zip`, para
poder verificar que dos copias del código eran idénticas sin git. Una vez
que el proyecto vive en un repositorio, **git ya resuelve ese problema
mejor** — `git log`, `git diff` y los números de commit identifican
exactamente qué versión tienes, sin necesidad de checksums manuales.
Puedes dejar de generar `CHECKSUMS.txt` en adelante; `VERSION.txt` puedes
conservarlo como un changelog legible si te resulta útil, o migrarlo a un
`CHANGELOG.md` más estándar.

**Sobre la licencia:** el proyecto es propietario de VecTech — ver el
archivo `LICENSE`. Nadie fuera del equipo puede usar, copiar o modificar
el código sin permiso explícito. `pyproject.toml` referencia ese archivo
directamente (`license = { file = "LICENSE" }`), así que queda embebido
en los metadatos del paquete cuando alguien lo instala.

Un detalle a revisar antes de compartir el repositorio más ampliamente:
`core/legacy_plc_predictor.py` replica la lógica de tu repositorio previo
(`luisroberto-maker/PLC-failure-prediction-pipeline`, el que usa tu
compañero) — si ese otro repositorio tiene una licencia pública distinta
declarada en GitHub, vale la pena homologarlas o dejar documentada la
diferencia, para que no haya ambigüedad sobre bajo qué términos circula
esa parte específica del código.
