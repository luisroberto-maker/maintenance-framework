"""
agents/agent3_tool_wear_executor.py — Agente 3 para la vertical de
desgaste de herramienta.

Antes de este módulo, la lógica de entrenar/predecir/cargar el
ToolWearPredictor vivía directamente dentro del orquestador
(MultiVerticalAgentSystem._run_tool_wear /  _predict_only_tool_wear),
mezclando responsabilidades de orquestación con las de ejecución del
pipeline. Este módulo la extrae, igual que agents/agent3_executor.py lo
hace para la vertical PLC — mismo rol, misma posición en el pipeline.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from core.config import AgentConfig
from core.tool_wear_predictor import ToolWearPredictor
from agents.agent2_tool_wear_planner import ToolWearPlan

logger = logging.getLogger(__name__)


class ToolWearExecutionResult:
    """Resultado de una ejecución — deliberadamente simple (no reutiliza
    el ExecutionResult de la vertical PLC porque ese dataclass está
    modelado alrededor de conceptos específicos de PLC como ModelStrategy
    y panel_metrics, que no aplican aquí)."""

    def __init__(self, result_df, metrics, feature_importances, model_path,
                execution_time_sec):
        self.result_df = result_df
        self.metrics = metrics
        self.feature_importances = feature_importances
        self.model_path = model_path
        self.execution_time_sec = execution_time_sec


class ToolWearExecutorAgent:
    """
    Agente 3 (desgaste de herramienta): entrena o carga el predictor,
    genera predicciones, y guarda el modelo para uso futuro sin reentrenar.
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config
        self.predictor: Optional[ToolWearPredictor] = None

    # ── Entrenamiento desde cero ────────────────────────────────────────────

    def execute(self, df: pd.DataFrame, plan: ToolWearPlan,
                client_id: str = "default") -> ToolWearExecutionResult:
        t0 = time.time()

        if not plan.can_train:
            raise ValueError(
                "El Agente 2 determinó que no se puede entrenar con estos "
                "datos. Revisa plan.rationale para el motivo."
            )

        self.predictor = ToolWearPredictor(vb_threshold=plan.vb_threshold)
        metrics = self.predictor.fit(df)
        result_df = self.predictor.predict_all(df)

        model_path = self._guardar_modelo(client_id)

        return ToolWearExecutionResult(
            result_df=result_df,
            metrics=metrics,
            feature_importances=self.predictor.feature_importance(),
            model_path=model_path,
            execution_time_sec=round(time.time() - t0, 1),
        )

    def _guardar_modelo(self, client_id: str) -> str:
        model_dir = Path(self.cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / f"tool_wear_predictor_{client_id}.pkl")
        self.predictor.save(model_path)
        return model_path

    # ── Inferencia con modelo ya entrenado (sin reentrenar) ─────────────────

    def load_and_predict(
        self,
        df: pd.DataFrame,
        client_id: str = "default",
        model_path: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
    ) -> ToolWearExecutionResult:
        t0 = time.time()

        if artifacts_dir:
            logger.info(f"Cargando modelo desde artefactos de Kaggle ({artifacts_dir})...")
            self.predictor = ToolWearPredictor.from_kaggle_artifacts(artifacts_dir)
            origen = artifacts_dir
        else:
            model_path = model_path or str(
                Path(self.cfg.model_dir) / f"tool_wear_predictor_{client_id}.pkl"
            )
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"No se encontró un modelo entrenado en {model_path}. "
                    f"Entrena primero con system.run(...), o pasa artifacts_dir "
                    f"si tu modelo viene de Kaggle."
                )
            logger.info(f"Cargando modelo entrenado desde {model_path}...")
            self.predictor = ToolWearPredictor.load(model_path)
            origen = model_path

        result_df = self.predictor.predict_all(df)
        metrics = self.predictor.metrics_ or {
            "nota": "modelo cargado sin reentrenar — métricas del entrenamiento original"
        }

        return ToolWearExecutionResult(
            result_df=result_df,
            metrics=metrics,
            feature_importances=self.predictor.feature_importance(),
            model_path=origen,
            execution_time_sec=round(time.time() - t0, 1),
        )

    def report_schema(self):
        """Delega al predictor activo — lo necesita el Agente 4."""
        return self.predictor.report_schema()
