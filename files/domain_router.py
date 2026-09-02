"""
agents/domain_router.py — Detección de dominio (vertical).

Antes de invocar cualquier agente, el sistema necesita saber qué tipo de
problema tiene enfrente: eventos discretos de fallas PLC, o señales
continuas de sensor para desgaste de herramienta. Este módulo decide eso
inspeccionando la forma de los datos, no su contenido semántico — así
generaliza a cualquier cliente nuevo dentro de cada vertical sin necesitar
configuración manual.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class Domain(Enum):
    PLC_FAILURE = "plc_failure"
    TOOL_WEAR = "tool_wear"
    UNKNOWN = "unknown"


# Señales de que estamos ante features de señal de sensor ya extraídas
_TOOL_WEAR_FEATURE_HINTS = ["_rms", "_kurtosis", "_crest", "_zpf1x", "_p0_500", "_entropy"]
_TOOL_WEAR_TARGET_HINTS = ["vb", "wear", "desgaste"]
_TOOL_WEAR_RAW_SIGNAL_COLS = {"fx", "fy", "fz", "vx", "vy", "vz", "ae"}

_PLC_EVENT_HINTS = ["start", "inicio", "fail", "comment", "log", "address", "componente",
                    "equipo", "alarm", "panel"]


def detect_domain(
    source: Union[str, Path, pd.DataFrame, tuple, dict],
    column_map: Optional[dict] = None,
) -> Domain:
    """
    Decide a qué vertical del framework pertenecen los datos de entrada.

    Reglas, en orden de prioridad:
      1. Si el usuario fuerza el dominio vía column_map — se respeta.
      2. Si source es un dict de experimentos {exp: {'signals':dir,'wear':csv}}
         (estructura cruda tipo PHM 2010) — TOOL_WEAR.
      3. Si source es una carpeta que contiene subcarpetas con archivos *_wear.csv
         — TOOL_WEAR.
      4. Si es DataFrame/archivo tabular:
         a. Si las columnas incluyen los 7 canales de señal cruda (fx,fy,fz,...)
            — TOOL_WEAR (requiere extracción de features).
         b. Si las columnas incluyen features ya extraídas (sufijos _rms, _kurtosis...)
            y una columna target tipo VB/wear — TOOL_WEAR (features precomputadas).
         c. Si las columnas contienen hints de eventos (start, fail comment, address...)
            — PLC_FAILURE.
      5. Si nada calza claramente — UNKNOWN (el orquestador decide el fallback).
    """
    if column_map and "domain" in column_map:
        try:
            return Domain(column_map["domain"])
        except ValueError:
            pass

    # ── Caso 2: dict de experimentos crudo ──────────────────────────────────────
    if isinstance(source, dict):
        sample = next(iter(source.values()), {})
        if isinstance(sample, dict) and "signals" in sample and "wear" in sample:
            return Domain.TOOL_WEAR

    # ── Caso 3: carpeta con estructura de experimentos ──────────────────────────
    if isinstance(source, (str, Path)) and os.path.isdir(str(source)):
        for _, _, files in os.walk(str(source)):
            if any(f.endswith("_wear.csv") for f in files):
                return Domain.TOOL_WEAR
        # Carpeta sin wear files reconocibles — no es claramente ninguna vertical
        return Domain.UNKNOWN

    # ── Caso 4: DataFrame o archivo tabular ──────────────────────────────────────
    df_cols = None
    if isinstance(source, pd.DataFrame):
        df_cols = [str(c).strip().lower() for c in source.columns]
    elif isinstance(source, (str, Path)) and os.path.isfile(str(source)):
        df_cols = _peek_columns(str(source))

    if df_cols:
        col_set = set(df_cols)

        # 4a. Señal cruda (7 canales, sin header semántico de evento)
        if _TOOL_WEAR_RAW_SIGNAL_COLS.issubset(col_set):
            return Domain.TOOL_WEAR

        # 4b. Features ya extraídas (sufijos característicos) + columna target
        has_feature_suffixes = any(
            any(hint in c for hint in _TOOL_WEAR_FEATURE_HINTS) for c in df_cols
        )
        has_wear_target = any(
            any(hint == c or hint in c for hint in _TOOL_WEAR_TARGET_HINTS) for c in df_cols
        )
        if has_feature_suffixes and has_wear_target:
            return Domain.TOOL_WEAR

        # 4c. Hints de eventos PLC
        has_event_hints = any(
            any(hint in c for hint in _PLC_EVENT_HINTS) for c in df_cols
        )
        if has_event_hints:
            return Domain.PLC_FAILURE

    return Domain.UNKNOWN


def _peek_columns(filepath: str) -> Optional[list[str]]:
    """Lee solo el header de un archivo tabular sin cargarlo completo."""
    suffix = Path(filepath).suffix.lower()
    try:
        if suffix in (".xlsx", ".xls", ".xlsm"):
            return [str(c).strip().lower() for c in pd.read_excel(filepath, nrows=0).columns]
        elif suffix == ".csv":
            return [str(c).strip().lower() for c in pd.read_csv(filepath, nrows=0).columns]
        elif suffix == ".parquet":
            return [str(c).strip().lower() for c in pd.read_parquet(filepath).columns]
    except Exception:
        return None
    return None
