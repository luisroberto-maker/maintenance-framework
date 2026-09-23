# Contrato de API — respuestas estructuradas

Este documento describe **exactamente** lo que devuelven `run_safe()` y
`predict_only_safe()` (`MultiVerticalAgentSystem`, ver `orchestrator.py`)
— los dos métodos recomendados para cualquier interfaz externa (como la
interfaz web).

**Todos los ejemplos de este documento son salida real, capturada
ejecutando el sistema** — no están escritos a mano. Si el formato
cambia en una versión futura, este documento debe regenerarse de la
misma forma (ver el pie de página).

## El shape base

Toda respuesta tiene siempre `status`. El resto de los campos varía
según el status:

```
"status": "SUCCESS" | "WARNING" | "ERROR"
```

- **SUCCESS** / **WARNING** → incluyen `report_path`, `warnings` (lista,
  vacía en `SUCCESS`), y `domain`.
- **ERROR** → incluyen `code` (ver tabla de `ErrorCode` abajo) y
  `details` (varía según el error).

## Códigos de error (`ErrorCode`)

Definidos en `core/errors.py` — estables entre versiones. Una interfaz
externa puede tomar decisiones distintas por código sin tener que leer
el texto de `message`.

| Código | Cuándo ocurre |
|---|---|
| `INSUFFICIENT_DATA` | No hay suficientes datos para entrenar, ni siquiera en modo heurístico |
| `UNKNOWN_SCHEMA` | No se pudieron identificar las columnas necesarias |
| `MODEL_NOT_FOUND` | `predict_only()` no encontró el `.pkl` esperado |
| `VALIDATION_ERROR` | Un parámetro de configuración (`AgentConfig`) es inválido |
| `UNKNOWN_DOMAIN` | No se pudo determinar si los datos son PLC o desgaste, y no se especificó explícitamente |
| `INTERNAL_ERROR` | Cualquier excepción no anticipada (nunca debería verse en operación normal) |

**Nota importante:** "pocos datos" no siempre produce `ERROR`. En la
vertical PLC, con pocos eventos el sistema puede degradar a modo
heurístico y devolver `WARNING` en vez de fallar — ver ejemplos 4 y 4b
abajo, que muestran ambos comportamientos con datos reales.

---

## 1. Entrenamiento exitoso — PLC

`system.run_safe(df, client_id="cliente_demo")` con 300 eventos válidos.

```json
{
  "status": "SUCCESS",
  "code": null,
  "message": "Completado correctamente.",
  "report_path": "output/reports/reporte_cliente_demo_20260923_164748.html",
  "warnings": [],
  "domain": "plc_failure"
}
```

**Cómo manejarlo en la UI:** mostrar el reporte (o los datos de
`get_ranking()`), sin ninguna advertencia que destacar.

---

## 2. Predicción — PLC (modelo ya entrenado)

`system.predict_only_safe(df_nuevo, client_id="cliente_demo", domain="plc_failure")`.

```json
{
  "status": "SUCCESS",
  "code": null,
  "message": "Completado correctamente.",
  "report_path": "output/reports/reporte_cliente_demo_20260923_164749.html",
  "warnings": [],
  "domain": "plc_failure"
}
```

**Cómo manejarlo en la UI:** igual que el caso 1 — mismo shape para
entrenar o solo predecir.

---

## 3. Predicción — desgaste de herramienta

`system.predict_only_safe(df_nuevo, client_id="cliente_tw", domain="tool_wear")`,
con solo 20 pasadas nuevas (por debajo del mínimo recomendado).

```json
{
  "status": "WARNING",
  "code": null,
  "message": "Completado con advertencias.",
  "report_path": "output/reports/reporte_cliente_tw_20260923_164750.html",
  "warnings": [
    "Rango de desgaste (VB): 0.0242–0.1727 mm. 0 pasadas superan el umbral de falla (0.2 mm).",
    "Solo 20 pasadas disponibles — el modelo de regresión puede tener alta varianza. Se recomiendan al menos 100-200."
  ],
  "domain": "tool_wear"
}
```

**Cómo manejarlo en la UI:** mostrar el resultado, pero destacar
visualmente las advertencias (por ejemplo, un banner amarillo) — la
predicción es válida pero con menos certeza de la ideal.

---

## 4. Datos insuficientes — caso WARNING (PLC, degrada a heurístico)

`system.run_safe(df, client_id="cliente_pocos_datos")` con solo 15 eventos.

```json
{
  "status": "WARNING",
  "code": null,
  "message": "Completado con advertencias.",
  "report_path": "output/reports/reporte_cliente_pocos_datos_20260923_164750.html",
  "warnings": [
    "Solo 15 eventos totales. El modelo global puede tener baja confianza.",
    "El rango temporal es de solo 2 días. Se recomienda al menos 30 días de histórico."
  ],
  "domain": "plc_failure"
}
```

**Cómo manejarlo en la UI:** el sistema SÍ completó (modo heurístico),
pero con confianza baja — comunicarlo claramente, no tratarlo como una
falla dura.

## 4b. Datos insuficientes — caso ERROR real (desgaste, sin mínimo viable)

`system.run_safe(df, client_id="cliente_muy_pocas_pasadas")` con solo 5
pasadas (mínimo requerido: 10 — ni el modo heurístico aplica aquí).

```json
{
  "status": "ERROR",
  "code": "INSUFFICIENT_DATA",
  "message": "No hay suficientes datos para entrenar un modelo.",
  "details": {
    "warnings": [
      "Rango de desgaste (VB): 0.0021–0.0069 mm. 0 pasadas superan el umbral de falla (0.2 mm).",
      "Solo 5 pasadas disponibles — el modelo de regresión puede tener alta varianza. Se recomiendan al menos 100-200.",
      "Solo 5 pasadas — por debajo del mínimo (10) para intentar entrenar."
    ]
  }
}
```

**Cómo manejarlo en la UI:** mostrar un mensaje de "necesitas más datos
históricos para este componente/herramienta" — no un error genérico.
`details.warnings` trae el detalle completo para mostrar u omitir según
el nivel de detalle que quieras darle al usuario final.

---

## 5. Modelo inexistente

`system.predict_only_safe(df, client_id="cliente_sin_modelo", domain="plc_failure")`
sin haber entrenado nunca un modelo para ese `client_id`.

```json
{
  "status": "ERROR",
  "code": "MODEL_NOT_FOUND",
  "message": "No se encontró un modelo entrenado en output/models/predictor_cliente_sin_modelo.pkl. Entrena primero con system.run(...).",
  "details": {
    "model_path": "output/models/predictor_cliente_sin_modelo.pkl",
    "client_id": "cliente_sin_modelo"
  }
}
```

**Cómo manejarlo en la UI:** este es el caso ideal para mostrar un botón
de "Entrenar modelo" en vez de un error — `details.client_id` te da
directamente el identificador para ofrecer la acción correctiva.

---

## 6. Formato inválido / columnas no reconocibles

`system.run_safe(df, client_id="cliente_formato_invalido")` con un
DataFrame sin ninguna columna reconocible (`columna_x`, `columna_y`).

```json
{
  "status": "ERROR",
  "code": "UNKNOWN_SCHEMA",
  "message": "No se pudo interpretar el schema de los datos.",
  "details": {
    "warnings": [
      "No se detectó columna de texto de log. Los embeddings semánticos no estarán disponibles.",
      "No se detectó columna de panel/línea. Se asignará 'PANEL_DESCONOCIDO' a todos los componentes.",
      "No se pudo identificar columna de componente o timestamp de inicio. Usa column_map para especificarlas manualmente."
    ]
  }
}
```

**Cómo manejarlo en la UI:** sugerir al usuario revisar el formato del
archivo, o exponer la opción de mapear columnas manualmente
(`column_map`) si tu interfaz lo permite.

---

## 7. Dominio desconocido

`system.predict_only_safe(df, client_id="cliente_dominio_ambiguo")` sin
`domain` explícito, con datos que no calzan claramente en ninguna
vertical.

```json
{
  "status": "ERROR",
  "code": "UNKNOWN_DOMAIN",
  "message": "No se pudo determinar el dominio automáticamente. Especifica domain='plc_failure' o domain='tool_wear' explícitamente.",
  "details": {}
}
```

**Cómo manejarlo en la UI:** pedirle al usuario que elija manualmente
el tipo de análisis (fallas PLC vs desgaste de herramienta), y reintenta
pasando `domain` explícitamente. Nota: esto solo ocurre en
`predict_only_safe()` — `run_safe()` con dominio ambiguo intenta PLC por
defecto en vez de fallar (ver `ARCHITECTURE.md`, sección 5).

---

## 8. Warning de baja confianza (dataset marginal, sí entrena)

`system.run_safe(df, client_id="cliente_marginal")` con 35 eventos —
apenas por debajo del umbral de calidad "moderada".

```json
{
  "status": "WARNING",
  "code": null,
  "message": "Completado con advertencias.",
  "report_path": "output/reports/reporte_cliente_marginal_20260923_164751.html",
  "warnings": [
    "Solo 35 eventos totales. El modelo global puede tener baja confianza.",
    "El rango temporal es de solo 5 días. Se recomienda al menos 30 días de histórico."
  ],
  "domain": "plc_failure"
}
```

**Cómo manejarlo en la UI:** igual que el caso 4 — completó, pero vale
la pena comunicar que la confianza es limitada. Este es el mismo
mecanismo que vimos con componentes de pocos eventos en un reporte real
— la recomendación es siempre cruzar esto con `N_EVENTOS` en
`get_ranking()` antes de presentar un resultado como definitivo.

---

## Cómo se generaron estos ejemplos

Todos se capturaron ejecutando `run_safe()`/`predict_only_safe()` contra
datos sintéticos (para no exponer datos reales de cliente en este
documento). Si cambias el formato de las respuestas en
`core/errors.py` o `orchestrator.py`, regenera este documento corriendo
los mismos escenarios y actualizando los JSON — no los edites a mano
para que sigan reflejando la salida real del sistema.
