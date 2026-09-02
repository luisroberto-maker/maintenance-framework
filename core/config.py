"""
core/config.py — Configuración central y modelos de datos del agente.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class DataQuality(Enum):
    RICH      = "rich"       # >100 eventos, >1 año, columnas completas
    MODERATE  = "moderate"   # 30-100 eventos
    SPARSE    = "sparse"     # 5-30 eventos
    MINIMAL   = "minimal"    # <5 eventos — solo heurístico


class ModelStrategy(Enum):
    LOCAL_RF   = "local_rf"    # RF dedicado por componente
    GLOBAL_RF  = "global_rf"   # RF global + transfer
    HEURISTIC  = "heuristic"   # Reglas basadas en priors
    HYBRID     = "hybrid"      # Global + ajuste local


class ConfidenceLevel(Enum):
    HIGH   = "ALTO"
    MEDIUM = "MEDIO"
    LOW    = "BAJO"


# ─── Schema inferido ──────────────────────────────────────────────────────────

@dataclass
class InferredSchema:
    """Resultado del Agente 1: descripción del schema detectado."""
    col_timestamp_start: Optional[str]  = None
    col_timestamp_end:   Optional[str]  = None
    col_component:       Optional[str]  = None
    col_log_text:        Optional[str]  = None
    col_panel:           Optional[str]  = None
    col_duration:        Optional[str]  = None   # si ya existe en los datos

    # Calidad detectada
    n_rows:              int   = 0
    n_components:        int   = 0
    date_range_days:     float = 0.0
    missing_pct:         float = 0.0
    has_duration:        bool  = False
    has_text_logs:       bool  = False
    encoding_detected:   str   = "utf-8"
    separator_detected:  str   = ","
    sheet_name:          Optional[str] = None

    # Warnings para el reporte
    warnings: list[str] = field(default_factory=list)
    is_valid: bool       = True


# ─── Plan de estrategia ───────────────────────────────────────────────────────

@dataclass
class StrategyPlan:
    """Resultado del Agente 2: decisión sobre qué pipeline ejecutar."""
    data_quality:    DataQuality
    model_strategy:  ModelStrategy
    can_train:       bool
    n_components_with_enough_data: int = 0
    recommended_umbral_critico:    int = 30
    recommended_ventana_dias:      int = 7
    recommended_n_eventos:         int = 10
    needs_hyperparameter_tuning:   bool = False

    rationale: list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)


# ─── Resultado de ejecución ───────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Resultado del Agente 3: predicciones y métricas."""
    ranking: object           = None   # pd.DataFrame
    metrics: dict             = field(default_factory=dict)
    feature_importances: dict = field(default_factory=dict)
    panel_metrics: object     = None   # pd.DataFrame
    model_path: Optional[str] = None
    execution_time_sec: float = 0.0
    n_components_predicted: int = 0
    strategy_used: ModelStrategy = ModelStrategy.GLOBAL_RF


# ─── Configuración global ─────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    """
    Parámetros ajustables para todo el sistema.

    Antes era un @dataclass simple: aceptaba cualquier valor, incluido
    umbral_critico=-5 o n_pca_components=0, y el error solo aparecía
    minutos después, a mitad del entrenamiento, con un mensaje que no
    decía nada sobre la causa real. Con Pydantic, un valor inválido
    falla AL CONSTRUIR el objeto, con un mensaje que dice exactamente
    qué campo está mal y por qué.

    Las restricciones de cada campo no son arbitrarias — reflejan cómo
    se usa ese valor en el código: por ejemplo, split_ratio se usa como
    int(len(df) * split_ratio) para partir train/test, así que 0 o 1
    dejarían un lado completamente vacío — por eso el límite es
    estrictamente entre 0 y 1, no "cualquier float".
    """

    model_config = ConfigDict(extra="forbid")  # typo en un nombre de campo falla al construir, no se ignora en silencio

    # ── Modelo ───────────────────────────────────────────────────────────────
    umbral_critico: int = Field(
        default=30, gt=0,
        description="Minutos de duración a partir de los cuales una falla PLC se considera crítica.",
    )
    ventana_dias: int = Field(
        default=7, gt=0,
        description="Horizonte de predicción en días (usado como pd.Timedelta(days=...)).",
    )
    default_n_eventos: int = Field(
        default=10, gt=0,
        description="Tamaño de ventana rolling por defecto — debe ser positivo (rolling(n) lo exige).",
    )
    split_ratio: float = Field(
        default=0.80, gt=0.0, lt=1.0,
        description="Proporción train/test. 0 o 1 dejarían un lado del split vacío.",
    )
    random_state: int = Field(default=42, ge=0)
    n_pca_components: int = Field(
        default=15, gt=0,
        description="Componentes de PCA para los embeddings — PCA exige n_components > 0.",
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2", min_length=1)

    # ── Umbrales de decisión ────────────────────────────────────────────────
    min_eventos_modelo_local: int = Field(default=30, gt=0)
    min_eventos_entrenamiento: int = Field(default=50, gt=0)
    umbral_incertidumbre: float = Field(default=0.25, gt=0.0)

    # ── Rutas ────────────────────────────────────────────────────────────────
    output_dir: str = Field(default="output", min_length=1)
    model_dir: str = Field(default="models", min_length=1)
    report_dir: str = Field(default="reports", min_length=1)

    # ── Reporte ──────────────────────────────────────────────────────────────
    top_n_ranking: int = Field(default=20, gt=0)
    idioma_reporte: Literal["es", "en"] = "es"

    @field_validator("output_dir", "model_dir", "report_dir")
    @classmethod
    def _sin_solo_espacios(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("no puede ser una cadena vacía o solo espacios en blanco")
        return v

    @field_validator("embedding_model")
    @classmethod
    def _sin_solo_espacios_embedding(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("no puede ser una cadena vacía o solo espacios en blanco")
        return v
