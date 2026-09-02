"""
agents/agent3_executor.py — Agente 3: Ejecutor del pipeline.

Responsabilidad: orquestar preprocesamiento, entrenamiento y predicción
usando el AdaptiveFailurePredictor con la estrategia decidida por el Agente 2.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

from core.config import AgentConfig, StrategyPlan, ExecutionResult, ModelStrategy
from core.predictor import AdaptiveFailurePredictor

logger = logging.getLogger(__name__)


class PipelineExecutorAgent:
    """
    Agente 3: Ejecuta el pipeline completo de entrenamiento y predicción.

    Recibe:  DataFrame normalizado + StrategyPlan del Agente 2
    Produce: ExecutionResult con ranking, métricas e importancia de features
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config
        self.predictor: AdaptiveFailurePredictor | None = None

    def execute(
        self,
        df: pd.DataFrame,
        plan: StrategyPlan,
        has_text_logs: bool = True,
        client_id: str = "default",
    ) -> ExecutionResult:
        """
        Ejecuta entrenamiento + predicción según el plan.

        Parameters
        ----------
        df          : DataFrame normalizado (salida del Agente 1)
        plan        : StrategyPlan (salida del Agente 2)
        has_text_logs : si hay columna de texto para embeddings
        client_id   : identificador del cliente (para aislar modelos)

        Returns
        -------
        ExecutionResult
        """
        t0 = time.time()
        result = ExecutionResult(strategy_used=plan.model_strategy)

        # ── Ajustar config según recomendaciones del planificador ────────────
        cfg = AgentConfig(
            umbral_critico    = plan.recommended_umbral_critico,
            ventana_dias      = self.cfg.ventana_dias,
            default_n_eventos = plan.recommended_n_eventos,
            split_ratio       = self.cfg.split_ratio,
            random_state      = self.cfg.random_state,
            n_pca_components  = self.cfg.n_pca_components,
            embedding_model   = self.cfg.embedding_model,
            min_eventos_modelo_local = self.cfg.min_eventos_modelo_local,
            umbral_incertidumbre     = self.cfg.umbral_incertidumbre,
            output_dir  = self.cfg.output_dir,
            model_dir   = self.cfg.model_dir,
            report_dir  = self.cfg.report_dir,
            top_n_ranking = self.cfg.top_n_ranking,
        )

        # ── Caso: sin datos suficientes — solo heurístico ────────────────────
        if not plan.can_train or plan.model_strategy == ModelStrategy.HEURISTIC:
            logger.warning("Estrategia heurística: no se entrenará un modelo ML.")
            result = self._ejecutar_heuristico(df, plan, result)
            result.execution_time_sec = round(time.time() - t0, 1)
            return result

        # ── Crear predictor ──────────────────────────────────────────────────
        self.predictor = AdaptiveFailurePredictor(
            cfg=cfg,
            strategy=plan.model_strategy,
            has_text_logs=has_text_logs,
        )

        # ── Entrenamiento ────────────────────────────────────────────────────
        logger.info(f"[Agente 3] Entrenando predictor ({plan.model_strategy.value})...")
        metricas = self.predictor.fit(
            df,
            tune_hyperparams=plan.needs_hyperparameter_tuning,
        )
        result.metrics = metricas

        # ── Predicción: ranking de riesgo ────────────────────────────────────
        logger.info("[Agente 3] Generando ranking de riesgo...")
        ranking = self.predictor.predict_all(df)
        result.ranking = ranking
        result.n_components_predicted = len(ranking)

        # ── Importancia de features ──────────────────────────────────────────
        result.feature_importances = self.predictor.feature_importance()

        # ── Métricas por panel ───────────────────────────────────────────────
        result.panel_metrics = self._metricas_por_panel(df, ranking)

        # ── Guardar modelo ───────────────────────────────────────────────────
        model_path = self._guardar_modelo(client_id)
        result.model_path = model_path

        result.execution_time_sec = round(time.time() - t0, 1)
        logger.info(f"[Agente 3] Completado en {result.execution_time_sec}s")
        return result

    def _ejecutar_heuristico(
        self, df: pd.DataFrame, plan: StrategyPlan, result: ExecutionResult
    ) -> ExecutionResult:
        """Genera un ranking basado en frecuencia de eventos cuando no hay ML."""
        result.strategy_used = ModelStrategy.HEURISTIC
        result.metrics = {"modo": "heurístico", "razón": plan.rationale}

        if "ADDRESS" not in df.columns:
            result.ranking = pd.DataFrame()
            return result

        agg = df.groupby("ADDRESS").agg(
            N_EVENTOS=("ADDRESS", "count"),
            PANEL=("PANEL", "first"),
        ).reset_index()

        # Probabilidad heurística = frecuencia normalizada
        agg["PROB_FALLA"] = (
            agg["N_EVENTOS"] / agg["N_EVENTOS"].max()
        ).round(4)
        agg["INCERTIDUMBRE"]   = 0.5
        agg["NIVEL_RIESGO"]    = agg["PROB_FALLA"].apply(
            lambda p: "🔴 ALTO" if p >= 0.6 else ("🟡 MEDIO" if p >= 0.3 else "🟢 BAJO")
        )
        agg["NIVEL_CONFIANZA"] = "BAJO"
        agg["MODO"]            = "HEURÍSTICO"

        result.ranking = agg.sort_values("PROB_FALLA", ascending=False).reset_index(drop=True)
        return result

    def _metricas_por_panel(
        self, df: pd.DataFrame, ranking: pd.DataFrame
    ) -> pd.DataFrame:
        if "PANEL" not in df.columns or ranking is None or len(ranking) == 0:
            return pd.DataFrame()
        panel_stats = (
            ranking.groupby("PANEL")
                   .agg(
                       n_componentes=("ADDRESS", "count"),
                       prob_media=("PROB_FALLA", "mean"),
                       n_alto_riesgo=(
                           "NIVEL_RIESGO",
                           lambda x: (x == "🔴 ALTO").sum(),
                       ),
                   )
                   .reset_index()
                   .sort_values("prob_media", ascending=False)
        )
        return panel_stats

    def _guardar_modelo(self, client_id: str) -> str:
        if self.predictor is None:
            return ""
        path = Path(self.cfg.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        model_file = str(path / f"predictor_{client_id}.pkl")
        self.predictor.save(model_file)
        return model_file

    def update(self, df_nuevos: pd.DataFrame, client_id: str = "default") -> int:
        """Actualización incremental con nuevos datos."""
        if self.predictor is None:
            model_path = Path(self.cfg.model_dir) / f"predictor_{client_id}.pkl"
            if model_path.exists():
                self.predictor = AdaptiveFailurePredictor.load(str(model_path))
            else:
                logger.error("No hay predictor cargado y no se encontró archivo guardado.")
                return 0
        return self.predictor.update(df_nuevos)

    def print_execution_report(self, result: ExecutionResult):
        lineas = [
            "",
            "═" * 55,
            "  AGENTE 3 — RESUMEN DE EJECUCIÓN",
            "═" * 55,
            f"  Estrategia usada:      {result.strategy_used.value.upper()}",
            f"  Componentes predichos: {result.n_components_predicted}",
            f"  Tiempo ejecución:      {result.execution_time_sec}s",
        ]
        if result.metrics:
            auc = result.metrics.get("auc_roc", "N/A")
            apr = result.metrics.get("auc_pr",  "N/A")
            lineas.append(f"  AUC-ROC:               {auc}")
            lineas.append(f"  AUC-PR:                {apr}")
        if result.model_path:
            lineas.append(f"  Modelo guardado en:    {result.model_path}")
        lineas.append("═" * 55 + "\n")
        logger.info("\n".join(lineas))
