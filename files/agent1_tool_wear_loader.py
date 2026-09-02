"""
agents/agent1_tool_wear_loader.py — Ingesta de datos para la vertical de
desgaste de herramienta. Cumple el mismo rol que DataAnalyzerAgent pero
para señales de sensor en vez de logs de eventos.

Acepta tres formas de entrada, de más cruda a más procesada:
  1. dict de experimentos {exp: {'signals': carpeta, 'wear': csv}} — extrae
     los 133 features por pasada desde las señales crudas (más lento).
  2. DataFrame/CSV con columnas fx,fy,fz,vx,vy,vz,ae de señal cruda por fila
     — no soportado directamente aquí (requeriría reconstrucción por
     pasada); se recomienda la forma 1 o 3 para este caso.
  3. DataFrame/CSV con features ya extraídas (columnas FEAT_COLS) + columna
     'VB' — el camino más común una vez el cliente ya tiene un pipeline de
     extracción propio, o para reentrenar con nuevas pasadas.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from core.config import InferredSchema
from core.tool_wear_predictor import (
    FEAT_COLS, load_experiment_from_folder, VB_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ToolWearDataLoaderAgent:
    """Agente 1 (variante) para la vertical de desgaste de herramienta."""

    def analyze(
        self,
        source: Union[dict, str, Path, pd.DataFrame],
        column_map: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, InferredSchema]:
        schema = InferredSchema()

        if isinstance(source, dict):
            df = self._load_from_experiments_dict(source, schema)
        elif isinstance(source, pd.DataFrame):
            df = self._load_from_dataframe(source, schema)
        elif isinstance(source, (str, Path)):
            df = self._load_from_path(source, schema)
        else:
            schema.is_valid = False
            schema.warnings.append(f"Tipo de fuente no soportado: {type(source)}")
            return pd.DataFrame(), schema

        if df is None or len(df) == 0:
            schema.is_valid = False
            schema.warnings.append("No se pudieron cargar datos de señal.")
            return pd.DataFrame(), schema

        self._assess_quality(df, schema)
        return df, schema

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_from_experiments_dict(self, source: dict, schema: InferredSchema) -> pd.DataFrame:
        """Ingesta cruda: extrae features desde carpetas de señal + wear files."""
        dfs = []
        for exp, paths in source.items():
            if "signals" not in paths or "wear" not in paths:
                schema.warnings.append(f"Experimento '{exp}' incompleto — se omite.")
                continue
            print(f"  Extrayendo features de '{exp}' (esto puede tardar varios minutos)...")
            df_exp = load_experiment_from_folder(exp, paths["signals"], paths["wear"])
            if len(df_exp) == 0:
                schema.warnings.append(f"Experimento '{exp}' no produjo pasadas válidas.")
                continue
            dfs.append(df_exp)

        if not dfs:
            return pd.DataFrame()

        schema.sheet_name = None
        schema.has_text_logs = False
        return pd.concat(dfs, ignore_index=True)

    def _load_from_dataframe(self, df: pd.DataFrame, schema: InferredSchema) -> pd.DataFrame:
        return self._validate_feature_frame(df, schema)

    def _load_from_path(self, path: Union[str, Path], schema: InferredSchema) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                schema.warnings.append(f"Extensión '{suffix}' no soportada para esta vertical.")
                return pd.DataFrame()
        except Exception as e:
            schema.warnings.append(f"Error al cargar {path}: {e}")
            schema.is_valid = False
            return pd.DataFrame()
        return self._validate_feature_frame(df, schema)

    def _validate_feature_frame(self, df: pd.DataFrame, schema: InferredSchema) -> pd.DataFrame:
        """Valida que el DataFrame traiga las features esperadas o al menos
        una fracción suficiente de ellas, y una columna target reconocible."""
        present = [c for c in FEAT_COLS if c in df.columns]
        missing = [c for c in FEAT_COLS if c not in df.columns]

        if len(present) < len(FEAT_COLS):
            schema.warnings.append(
                f"Faltan {len(missing)}/{len(FEAT_COLS)} columnas de features esperadas. "
                f"Verifica que los datos pasaron por la extracción de 133 features "
                f"(19 estadísticos × 7 canales de señal)."
            )
        if len(present) < len(FEAT_COLS) * 0.5:
            schema.is_valid = False
            schema.warnings.append(
                "Menos de la mitad de las features esperadas están presentes — "
                "no es posible entrenar de forma confiable."
            )
            return df

        target_col = None
        for cand in ["VB", "vb", "wear", "desgaste"]:
            if cand in df.columns:
                target_col = cand
                break
        if target_col is None:
            schema.is_valid = False
            schema.warnings.append(
                "No se encontró columna target de desgaste (se esperaba 'VB', 'wear' "
                "o 'desgaste'). Sin ella no es posible entrenar el modelo de regresión."
            )
            return df
        if target_col != "VB":
            df = df.rename(columns={target_col: "VB"})

        schema.has_text_logs = False
        return df

    # ── Calidad ───────────────────────────────────────────────────────────────

    def _assess_quality(self, df: pd.DataFrame, schema: InferredSchema):
        schema.n_rows = len(df)
        schema.n_components = df["experiment"].nunique() if "experiment" in df.columns else 1

        if "VB" in df.columns:
            vb = df["VB"].dropna()
            if len(vb) > 0:
                n_fallas = int((vb >= VB_THRESHOLD).sum())
                schema.warnings.append(
                    f"Rango de desgaste (VB): {vb.min():.4f}–{vb.max():.4f} mm. "
                    f"{n_fallas} pasadas superan el umbral de falla ({VB_THRESHOLD} mm)."
                )

        if "experiment" in df.columns:
            n_exp = df["experiment"].nunique()
            if n_exp < 2:
                schema.warnings.append(
                    "Solo hay un experimento/herramienta en los datos — la evaluación "
                    "usará split aleatorio en vez de Leave-One-Experiment-Out (LOEO), "
                    "lo que sobreestima el rendimiento real en herramientas nuevas."
                )

        if schema.n_rows < 60:
            schema.warnings.append(
                f"Solo {schema.n_rows} pasadas disponibles — el modelo de regresión "
                "puede tener alta varianza. Se recomiendan al menos 100-200."
            )

    def print_schema_report(self, schema: InferredSchema):
        print("\n" + "═" * 55)
        print("  AGENTE 1 (desgaste de herramienta) — SCHEMA DETECTADO")
        print("═" * 55)
        print(f"  Pasadas totales:      {schema.n_rows:,}")
        print(f"  Experimentos/herram.: {schema.n_components}")
        if schema.warnings:
            print()
            for w in schema.warnings:
                print(f"  ⚠ {w}")
        status = "✓ VÁLIDO" if schema.is_valid else "✗ INVÁLIDO"
        print(f"\n  Estado: {status}")
        print("═" * 55 + "\n")
