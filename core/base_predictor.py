"""
core/base_predictor.py — Interfaz común que deben implementar todos los
predictores del framework, sin importar el dominio (fallas de eventos PLC,
desgaste de herramienta por señales de sensor, o verticales futuras).

Esto es lo que permite que el Agente 3 (ejecutor) y el Agente 4 (reporte)
funcionen igual sin importar qué modelo hay detrás — solo dependen de esta
interfaz, nunca de una implementación concreta.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ReportSchema:
    """
    Describe cómo el Agente 4 debe renderizar el resultado de un predictor,
    sin que el Agente 4 necesite saber nada específico del dominio.
    """
    domain_label: str                 # ej. "Predicción de fallas PLC" / "Desgaste de herramienta"
    id_col: str                       # columna identificadora de la entidad (ADDRESS / experiment+cut)
    id_label: str                     # etiqueta legible ("Componente" / "Herramienta / pasada")
    group_col: str                    # columna de agrupación secundaria (PANEL / experimento)
    group_label: str                  # etiqueta legible ("Panel" / "Experimento")
    primary_metric_col: str           # columna numérica principal (PROB_FALLA / VB_ESTIMADO)
    primary_metric_label: str         # etiqueta legible ("Probabilidad de falla" / "Desgaste estimado (mm)")
    status_col: str                   # columna categórica de estado (NIVEL_RIESGO / ESTADO)
    status_levels: list[tuple[str, str]] = field(default_factory=list)  # [(nivel, color_hex), ...]
    confidence_col: str = "NIVEL_CONFIANZA"
    extra_cols: list[tuple[str, str]] = field(default_factory=list)     # [(col, etiqueta), ...] adicionales
    higher_is_worse: bool = True      # True: valores altos = peor (prob falla). False: ej. RUL alto = mejor.


class BasePredictor(ABC):
    """
    Todo predictor del framework —sin importar el dominio— expone estos
    cuatro métodos. El Agente 3 solo conoce esta interfaz; el Agente 4 solo
    conoce el ReportSchema que devuelve report_schema().
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> dict:
        """Entrena el modelo. Devuelve un diccionario de métricas de evaluación."""
        raise NotImplementedError

    @abstractmethod
    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera predicciones para todas las entidades presentes en df."""
        raise NotImplementedError

    @abstractmethod
    def report_schema(self) -> ReportSchema:
        """Describe cómo debe renderizarse el resultado en el reporte."""
        raise NotImplementedError

    def feature_importance(self) -> dict:
        """Opcional — no todos los predictores necesitan exponerla."""
        return {}

    def save(self, path: str):
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BasePredictor":
        import joblib
        return joblib.load(path)
