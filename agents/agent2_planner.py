"""
agents/agent2_planner.py — Agente 2: Planificador de estrategia.

Responsabilidad: dado el InferredSchema del Agente 1, decidir qué
pipeline y estrategia de modelo usar, calibrar hiperparámetros base
y emitir advertencias sobre limitaciones esperadas.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from core.config import (
    AgentConfig, InferredSchema, StrategyPlan,
    DataQuality, ModelStrategy,
)

logger = logging.getLogger(__name__)


class StrategyPlannerAgent:
    """
    Agente 2: Evalúa los datos y decide la estrategia de modelado.

    Reglas de decisión
    ------------------
    Calidad del dataset (global):
      RICH     → >500 eventos totales, >90 días, >5 componentes con datos suficientes
      MODERATE → 100-500 eventos totales
      SPARSE   → 30-100 eventos totales
      MINIMAL  → <30 eventos totales

    Estrategia de modelo:
      LOCAL_RF  → calidad RICH + componentes con >=30 eventos individuales
      HYBRID    → calidad MODERATE + algunos componentes con >=30 eventos
      GLOBAL_RF → calidad SPARSE o componentes con <30 eventos
      HEURISTIC → calidad MINIMAL o imposible entrenar
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config

    def plan(
        self, df: pd.DataFrame, schema: InferredSchema
    ) -> StrategyPlan:
        """
        Analiza el DataFrame normalizado y produce un StrategyPlan.
        """
        plan = StrategyPlan(
            data_quality=DataQuality.MINIMAL,
            model_strategy=ModelStrategy.HEURISTIC,
            can_train=False,
        )

        if not schema.is_valid or len(df) == 0:
            plan.rationale.append("Schema inválido o datos vacíos → solo heurístico.")
            return plan

        # ── 1. Evaluar calidad global ─────────────────────────────────────────
        n_rows   = len(df)
        n_days   = schema.date_range_days
        n_comps  = schema.n_components

        if n_rows >= 500 and n_days >= 90:
            plan.data_quality = DataQuality.RICH
        elif n_rows >= 100 and n_days >= 30:
            plan.data_quality = DataQuality.MODERATE
        elif n_rows >= 30:
            plan.data_quality = DataQuality.SPARSE
        else:
            plan.data_quality = DataQuality.MINIMAL

        plan.rationale.append(
            f"Calidad global: {plan.data_quality.value} "
            f"({n_rows} eventos, {n_days} días, {n_comps} componentes)."
        )

        # ── 2. Contar componentes con datos suficientes ───────────────────────
        eventos_por_componente = df.groupby("ADDRESS").size()
        n_suficientes = int(
            (eventos_por_componente >= self.cfg.min_eventos_modelo_local).sum()
        )
        plan.n_components_with_enough_data = n_suficientes

        plan.rationale.append(
            f"{n_suficientes} componentes con >="
            f"{self.cfg.min_eventos_modelo_local} eventos (candidatos a modelo local)."
        )

        # ── 3. Verificar si hay TARGET calculable ─────────────────────────────
        # Necesitamos al menos ventana_dias de datos para que el target tenga sentido
        can_compute_target = n_days >= self.cfg.ventana_dias * 2
        if not can_compute_target:
            plan.warnings.append(
                f"El rango temporal ({n_days:.0f} días) es menor que el doble de la "
                f"ventana de predicción ({self.cfg.ventana_dias * 2} días). "
                "El target puede ser muy ruidoso."
            )

        # ── 4. Verificar si hay suficientes positivos para entrenar ───────────
        if "ES_CRITICA" in df.columns:
            tasa_critica = df["ES_CRITICA"].mean()
            if tasa_critica < 0.01:
                plan.warnings.append(
                    f"Solo el {tasa_critica*100:.2f}% de los eventos son críticos. "
                    "El modelo puede ser difícil de entrenar con class_weight='balanced'."
                )
            elif tasa_critica > 0.7:
                plan.warnings.append(
                    f"El {tasa_critica*100:.1f}% de los eventos son críticos. "
                    "Revisar el umbral_critico — puede estar demasiado bajo."
                )

        # ── 5. Decidir estrategia de modelo ───────────────────────────────────
        if plan.data_quality == DataQuality.MINIMAL or n_rows < 30:
            plan.model_strategy = ModelStrategy.HEURISTIC
            plan.can_train      = False
            plan.rationale.append("Datos insuficientes → solo heurístico.")

        elif plan.data_quality == DataQuality.SPARSE or n_suficientes == 0:
            plan.model_strategy = ModelStrategy.GLOBAL_RF
            plan.can_train      = True
            plan.rationale.append(
                "Pocos datos por componente → modelo global RF para generalización."
            )

        elif plan.data_quality == DataQuality.MODERATE:
            plan.model_strategy = ModelStrategy.HYBRID
            plan.can_train      = True
            plan.rationale.append(
                f"Datos moderados → modelo híbrido "
                f"(global + {n_suficientes} modelos locales)."
            )

        else:  # RICH
            plan.model_strategy = ModelStrategy.LOCAL_RF
            plan.can_train      = True
            plan.rationale.append(
                f"Datos ricos → modelos locales dedicados para "
                f"{n_suficientes} componentes."
            )

        # ── 6. Calibrar hiperparámetros base ──────────────────────────────────
        plan.recommended_umbral_critico = self._calibrar_umbral(df)
        plan.recommended_n_eventos      = self._calibrar_n_eventos(df)
        plan.needs_hyperparameter_tuning = (
            plan.data_quality in (DataQuality.RICH, DataQuality.MODERATE)
            and n_rows >= 200
        )

        if not schema.has_text_logs:
            plan.warnings.append(
                "Sin columna de texto → embeddings semánticos desactivados. "
                "El modelo usará solo features temporales."
            )

        return plan

    def _calibrar_umbral(self, df: pd.DataFrame) -> int:
        """Sugiere umbral_critico basado en la distribución de duraciones."""
        if "DURACION_MIN" not in df.columns:
            return self.cfg.umbral_critico
        duraciones = df["DURACION_MIN"].dropna()
        if len(duraciones) < 10:
            return self.cfg.umbral_critico
        # Percentil 70 como umbral sugerido
        p70 = int(np.percentile(duraciones, 70))
        return max(5, min(p70, 120))  # entre 5 y 120 min

    def _calibrar_n_eventos(self, df: pd.DataFrame) -> int:
        """Sugiere ventana de eventos basada en la mediana de eventos entre críticos."""
        if "ES_CRITICA" not in df.columns or "ADDRESS" not in df.columns:
            return self.cfg.default_n_eventos

        sugerencias = []
        for _, group in df.groupby("ADDRESS"):
            group = group.sort_values("START TIME")
            idx_criticos = group[group["ES_CRITICA"] == 1].index.tolist()
            if len(idx_criticos) < 2:
                continue
            diffs = [
                idx_criticos[i] - idx_criticos[i - 1]
                for i in range(1, len(idx_criticos))
            ]
            sugerencias.append(np.median(diffs))

        if not sugerencias:
            return self.cfg.default_n_eventos
        return max(3, int(np.median(sugerencias)))

    def print_plan_report(self, plan: StrategyPlan):
        lineas = [
            "",
            "═" * 55,
            "  AGENTE 2 — PLAN DE ESTRATEGIA",
            "═" * 55,
            f"  Calidad de datos:       {plan.data_quality.value.upper()}",
            f"  Estrategia elegida:     {plan.model_strategy.value.upper()}",
            f"  Puede entrenar:         {'✓' if plan.can_train else '✗'}",
            f"  Componentes con datos:  {plan.n_components_with_enough_data}",
            f"  Umbral crítico suger.:  {plan.recommended_umbral_critico} min",
            f"  N eventos sugerido:     {plan.recommended_n_eventos}",
            f"  Tuning HP sugerido:     {'✓' if plan.needs_hyperparameter_tuning else '✗'}",
        ]
        if plan.rationale:
            lineas.append("")
            lineas.append("  Razonamiento:")
            lineas.extend(f"    → {r}" for r in plan.rationale)
        if plan.warnings:
            lineas.append("")
            lineas.append("  ⚠ Advertencias:")
            lineas.extend(f"    • {w}" for w in plan.warnings)
        lineas.append("═" * 55 + "\n")
        logger.info("\n".join(lineas))
