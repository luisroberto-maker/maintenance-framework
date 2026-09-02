"""
agents/agent1_analyzer.py — Agente 1: Analizador de datos.

Responsabilidad: recibir CUALQUIER fuente de datos (Excel, CSV, JSON, SQL,
parquet, TXT) e inferir automáticamente el schema, calidad y viabilidad
para entrenar el predictor de fallas.
"""
from __future__ import annotations

import io
import re
import logging
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd

from core.config import AgentConfig, InferredSchema

logger = logging.getLogger(__name__)


# ─── Heurísticas de mapeo de columnas ────────────────────────────────────────

_TIMESTAMP_START_HINTS = [
    "start", "inicio", "begin", "fecha_inicio", "start_time",
    "starttime", "timestamp", "fecha", "time", "dt_start",
]
_TIMESTAMP_END_HINTS = [
    "end", "fin", "finish", "fecha_fin", "end_time",
    "endtime", "stop", "dt_end",
]
_COMPONENT_HINTS = [
    "address", "componente", "component", "equipo", "device",
    "tag", "asset", "nombre_componente", "id_componente",
]
_LOG_TEXT_HINTS = [
    "fail comment", "log", "mensaje", "message", "descripcion",
    "description", "alarm", "alarma", "texto", "text", "comment",
    "fault", "error_msg", "detalle",
]
_PANEL_HINTS = [
    "panel", "linea", "line", "zona", "zone", "area",
    "machine", "maquina", "celula", "cell",
]
_DURATION_HINTS = [
    "duracion", "duration", "tiempo", "elapsed", "minutes",
    "minutos", "segundos", "seconds", "horas", "hours",
]


def _best_match(col_names: list[str], hints: list[str]) -> Optional[str]:
    """Elige la columna cuyo nombre más se parece a alguno de los hints."""
    col_lower = {c: c.lower().replace(" ", "_") for c in col_names}
    for hint in hints:
        for original, normalized in col_lower.items():
            if hint in normalized:
                return original
    return None


# ─── Loaders ─────────────────────────────────────────────────────────────────

def _load_excel(path: Path, schema: InferredSchema) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    if len(xl.sheet_names) > 1:
        schema.warnings.append(
            f"Múltiples hojas detectadas: {xl.sheet_names}. "
            f"Se usará '{xl.sheet_names[0]}'. Configura sheet_name si deseas otra."
        )
    sheet = schema.sheet_name or xl.sheet_names[0]
    schema.sheet_name = sheet
    return pd.read_excel(path, sheet_name=sheet)


def _load_csv(path: Path, schema: InferredSchema) -> pd.DataFrame:
    # Detectar separador automáticamente
    sample = path.read_bytes()[:4096].decode("utf-8", errors="replace")
    sniffer = __import__("csv").Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = ","
    schema.separator_detected = sep
    return pd.read_csv(path, sep=sep, encoding=schema.encoding_detected,
                       on_bad_lines="skip")


def _load_json(path: Path, schema: InferredSchema) -> pd.DataFrame:
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    # Soporta lista de registros o dict con clave 'data'/'records'
    if isinstance(data, list):
        return pd.DataFrame(data)
    for key in ("data", "records", "rows", "items"):
        if key in data:
            return pd.DataFrame(data[key])
    # Intenta normalizar anidado
    return pd.json_normalize(data)


def _load_parquet(path: Path, schema: InferredSchema) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_sql(connection_string: str, query: str,
              schema: InferredSchema) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ─── Agente principal ─────────────────────────────────────────────────────────

class DataAnalyzerAgent:
    """
    Agente 1: Analiza la fuente de datos y produce un InferredSchema.

    Uso:
        agent = DataAnalyzerAgent(config)
        df, schema = agent.analyze("datos.xlsx")
        df, schema = agent.analyze("datos.csv")
        df, schema = agent.analyze(mi_dataframe)   # también acepta DataFrames directamente
        df, schema = agent.analyze(
            source=("postgresql://user:pw@host/db", "SELECT * FROM plc_logs")
        )
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config

    def analyze(
        self,
        source: Union[str, Path, pd.DataFrame, tuple],
        column_map: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, InferredSchema]:
        """
        Punto de entrada principal.

        Parameters
        ----------
        source : archivo, DataFrame o (connection_string, query)
        column_map : dict opcional para forzar el mapeo de columnas,
                     ej. {"col_component": "TAG_ID", "col_timestamp_start": "FECHA"}

        Returns
        -------
        (df_normalizado, schema)
        """
        schema = InferredSchema()
        df = self._load(source, schema)

        if df is None or len(df) == 0:
            schema.is_valid = False
            schema.warnings.append("No se pudieron cargar datos.")
            return pd.DataFrame(), schema

        logger.info(f"Datos cargados: {len(df)} filas × {len(df.columns)} columnas")

        # 1. Inferir / aplicar mapeo de columnas
        self._infer_columns(df, schema, column_map or {})

        # 2. Normalizar el DataFrame al schema canónico
        df = self._normalize(df, schema)

        # 3. Evaluar calidad
        self._assess_quality(df, schema)

        # 4. Reporte de columnas no reconocidas
        self._warn_unrecognized_columns(df, schema)

        if schema.col_component is None or schema.col_timestamp_start is None:
            schema.is_valid = False
            schema.warnings.append(
                "No se pudo identificar columna de componente o timestamp de inicio. "
                "Usa column_map para especificarlas manualmente."
            )

        return df, schema

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load(self, source, schema: InferredSchema) -> Optional[pd.DataFrame]:
        try:
            if isinstance(source, pd.DataFrame):
                return source.copy()

            if isinstance(source, tuple):
                conn_str, query = source
                return _load_sql(conn_str, query, schema)

            path = Path(source)
            suffix = path.suffix.lower()

            if suffix in (".xlsx", ".xls", ".xlsm"):
                return _load_excel(path, schema)
            elif suffix == ".csv":
                return _load_csv(path, schema)
            elif suffix == ".json":
                return _load_json(path, schema)
            elif suffix in (".parquet", ".pq"):
                return _load_parquet(path, schema)
            elif suffix == ".txt":
                return _load_csv(path, schema)
            else:
                schema.warnings.append(
                    f"Extensión '{suffix}' no reconocida. Intentando leer como CSV."
                )
                return _load_csv(path, schema)

        except Exception as e:
            schema.warnings.append(f"Error al cargar datos: {e}")
            schema.is_valid = False
            return None

    # ── Inferencia de columnas ─────────────────────────────────────────────────

    def _infer_columns(
        self, df: pd.DataFrame, schema: InferredSchema, forced: dict
    ):
        cols = df.columns.tolist()

        def resolve(attr, hints):
            if attr in forced:
                return forced[attr]
            return _best_match(cols, hints)

        schema.col_timestamp_start = resolve("col_timestamp_start", _TIMESTAMP_START_HINTS)
        schema.col_timestamp_end   = resolve("col_timestamp_end",   _TIMESTAMP_END_HINTS)
        schema.col_component       = resolve("col_component",       _COMPONENT_HINTS)
        schema.col_log_text        = resolve("col_log_text",        _LOG_TEXT_HINTS)
        schema.col_panel           = resolve("col_panel",           _PANEL_HINTS)
        schema.col_duration        = resolve("col_duration",        _DURATION_HINTS)

        # Log del mapeo detectado
        logger.info(
            f"Schema inferido → "
            f"timestamp_start='{schema.col_timestamp_start}', "
            f"timestamp_end='{schema.col_timestamp_end}', "
            f"component='{schema.col_component}', "
            f"log_text='{schema.col_log_text}', "
            f"panel='{schema.col_panel}'"
        )

        # Advertencias si falta algo importante
        if schema.col_log_text is None:
            schema.warnings.append(
                "No se detectó columna de texto de log. "
                "Los embeddings semánticos no estarán disponibles."
            )
            schema.has_text_logs = False
        else:
            schema.has_text_logs = True

        if schema.col_panel is None:
            schema.warnings.append(
                "No se detectó columna de panel/línea. "
                "Se asignará 'PANEL_DESCONOCIDO' a todos los componentes."
            )

    # ── Normalización ─────────────────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame, schema: InferredSchema) -> pd.DataFrame:
        """Renombra las columnas al schema canónico que espera el pipeline."""
        rename_map = {}
        if schema.col_timestamp_start:
            rename_map[schema.col_timestamp_start] = "START TIME"
        if schema.col_timestamp_end:
            rename_map[schema.col_timestamp_end] = "END TIME"
        if schema.col_component:
            rename_map[schema.col_component] = "ADDRESS"
        if schema.col_log_text:
            rename_map[schema.col_log_text] = "FAIL COMMENT"
        if schema.col_panel:
            rename_map[schema.col_panel] = "PANEL"

        df = df.rename(columns=rename_map)

        # Columnas obligatorias ausentes → rellenar con defaults
        if "PANEL" not in df.columns:
            df["PANEL"] = "PANEL_DESCONOCIDO"
        if "FAIL COMMENT" not in df.columns:
            df["FAIL COMMENT"] = "UNKNOWN_LOG"

        # Parsear timestamps con detección automática de formato
        for col in ["START TIME", "END TIME"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Calcular duración si no existe
        if "DURACION_MIN" not in df.columns:
            if "START TIME" in df.columns and "END TIME" in df.columns:
                df["DURACION_MIN"] = (
                    (df["END TIME"] - df["START TIME"]).dt.total_seconds() / 60
                )
                schema.has_duration = True
            elif schema.col_duration and schema.col_duration in df.columns:
                df["DURACION_MIN"] = pd.to_numeric(
                    df[schema.col_duration], errors="coerce"
                ).fillna(0)
                schema.has_duration = True
            else:
                df["DURACION_MIN"] = 1.0  # default si no hay duración
                schema.warnings.append(
                    "No se encontró columna de duración ni timestamps de fin. "
                    "Se asume duración de 1 minuto para todos los eventos."
                )

        # Eliminar filas con timestamp nulo en START TIME
        n_before = len(df)
        df = df.dropna(subset=["START TIME"]).copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            schema.warnings.append(
                f"{n_dropped} filas eliminadas por timestamp inválido en START TIME."
            )

        return df.sort_values("START TIME").reset_index(drop=True)

    # ── Calidad ───────────────────────────────────────────────────────────────

    def _assess_quality(self, df: pd.DataFrame, schema: InferredSchema):
        schema.n_rows = len(df)
        schema.n_components = df["ADDRESS"].nunique() if "ADDRESS" in df.columns else 0

        if "START TIME" in df.columns and len(df) > 1:
            schema.date_range_days = (
                df["START TIME"].max() - df["START TIME"].min()
            ).days

        if len(df) > 0:
            schema.missing_pct = df.isnull().mean().mean() * 100

        # Advertencias de calidad
        if schema.n_rows < 50:
            schema.warnings.append(
                f"Solo {schema.n_rows} eventos totales. "
                "El modelo global puede tener baja confianza."
            )
        if schema.date_range_days < 30:
            schema.warnings.append(
                f"El rango temporal es de solo {schema.date_range_days} días. "
                "Se recomienda al menos 30 días de histórico."
            )
        if schema.missing_pct > 20:
            schema.warnings.append(
                f"El {schema.missing_pct:.1f}% de los datos son nulos. "
                "Esto puede afectar la calidad del modelo."
            )

    def _warn_unrecognized_columns(self, df: pd.DataFrame, schema: InferredSchema):
        canonical = {"START TIME", "END TIME", "ADDRESS", "FAIL COMMENT",
                     "PANEL", "DURACION_MIN"}
        extra = set(df.columns) - canonical
        if extra:
            logger.info(f"Columnas adicionales disponibles (no usadas): {extra}")

    # ── Reporte de schema ─────────────────────────────────────────────────────

    def print_schema_report(self, schema: InferredSchema):
        lineas = [
            "",
            "═" * 55,
            "  AGENTE 1 — REPORTE DE SCHEMA DETECTADO",
            "═" * 55,
            f"  Filas totales:        {schema.n_rows:,}",
            f"  Componentes únicos:   {schema.n_components:,}",
            f"  Rango temporal:       {schema.date_range_days:.0f} días",
            f"  Datos faltantes:      {schema.missing_pct:.1f}%",
            f"  Tiene logs de texto:  {'✓' if schema.has_text_logs else '✗'}",
            f"  Tiene duración:       {'✓' if schema.has_duration else '✗'}",
            "",
            "  Mapeo de columnas detectado:",
            f"    START TIME  ← '{schema.col_timestamp_start}'",
            f"    END TIME    ← '{schema.col_timestamp_end}'",
            f"    ADDRESS     ← '{schema.col_component}'",
            f"    FAIL COMMENT← '{schema.col_log_text}'",
            f"    PANEL       ← '{schema.col_panel}'",
        ]
        if schema.warnings:
            lineas.append("")
            lineas.append("  ⚠ Advertencias:")
            lineas.extend(f"    • {w}" for w in schema.warnings)

        status = "✓ VÁLIDO" if schema.is_valid else "✗ INVÁLIDO"
        lineas.append(f"\n  Estado: {status}")
        lineas.append("═" * 55 + "\n")
        logger.info("\n".join(lineas))
