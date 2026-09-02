"""
agents/agent2_tool_wear_planner.py — Agente 2 para la vertical de desgaste
de herramienta.

Antes de este módulo, la vertical de desgaste no tenía un Agente 2: el
orquestador decidía la estrategia de forma implícita, mezclada con la
ejecución. Este módulo separa esa decisión, igual que
agents/agent2_planner.py lo hace para la vertical PLC — mismo rol, misma
posición en el pipeline, lógica de decisión propia del dominio.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.config import InferredSchema

logger = logging.getLogger(__name__)

try:
    import xgboost  # noqa: F401
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


@dataclass
class ToolWearPlan:
    """Equivalente a StrategyPlan (vertical PLC), pero con las decisiones
    propias de la vertical de desgaste de herramienta."""
    can_train: bool
    usa_loeo: bool                  # True: split Leave-One-Experiment-Out
    usara_ensemble: bool            # True: RF + XGBoost, False: solo RF
    vb_threshold: float = 0.2
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ToolWearStrategyPlanner:
    """
    Agente 2 (desgaste de herramienta): decide cómo se debe entrenar y
    evaluar el modelo, dado lo que el Agente 1 ya cargó y validó.

    Decisiones que toma:
      - ¿Hay suficientes pasadas para intentar entrenar?
      - ¿Se puede usar split LOEO (Leave-One-Experiment-Out), o toca caer
        a split aleatorio 80/20 por tener un solo experimento?
      - ¿Se usará el ensamble RF+XGBoost, o solo Random Forest porque
        xgboost no está disponible en este entorno?
    """

    MIN_PASADAS_PARA_ENTRENAR = 10

    def __init__(self, vb_threshold: float = 0.2):
        self.vb_threshold = vb_threshold

    def plan(self, df: pd.DataFrame, schema: InferredSchema) -> ToolWearPlan:
        plan = ToolWearPlan(
            can_train=False, usa_loeo=False, usara_ensemble=False,
            vb_threshold=self.vb_threshold,
        )

        if not schema.is_valid or len(df) == 0:
            plan.rationale.append("Schema inválido o datos vacíos — no se puede entrenar.")
            return plan

        if "VB" not in df.columns:
            plan.rationale.append(
                "Sin columna de desgaste (VB) — estos datos solo sirven para "
                "inferencia con un modelo ya entrenado, no para entrenar uno nuevo."
            )
            return plan

        n_pasadas = len(df)
        if n_pasadas < self.MIN_PASADAS_PARA_ENTRENAR:
            plan.rationale.append(
                f"Solo {n_pasadas} pasadas — por debajo del mínimo "
                f"({self.MIN_PASADAS_PARA_ENTRENAR}) para intentar entrenar."
            )
            return plan

        plan.can_train = True
        plan.rationale.append(f"{n_pasadas} pasadas disponibles — suficiente para entrenar.")

        # ── Decisión: LOEO vs split aleatorio ───────────────────────────────
        n_experimentos = df["experiment"].nunique() if "experiment" in df.columns else 1
        if n_experimentos >= 2:
            plan.usa_loeo = True
            plan.rationale.append(
                f"{n_experimentos} experimentos detectados — se usará split "
                f"Leave-One-Experiment-Out (LOEO) para una evaluación honesta "
                f"sobre una herramienta nunca vista."
            )
        else:
            plan.usa_loeo = False
            plan.warnings.append(
                "Solo hay un experimento/herramienta en los datos — la "
                "evaluación usará split aleatorio en vez de Leave-One-"
                "Experiment-Out (LOEO), lo que sobreestima el rendimiento "
                "real en herramientas nuevas."
            )

        # ── Decisión: ensamble RF+XGBoost vs solo RF ────────────────────────
        if _HAS_XGB:
            plan.usara_ensemble = True
            plan.rationale.append("xgboost disponible — se usará ensamble RF + XGBoost.")
        else:
            plan.usara_ensemble = False
            plan.warnings.append(
                "xgboost no disponible en este entorno — se usará solo "
                "Random Forest. Las predicciones seguirán siendo válidas, "
                "pero no reproducirán exactamente un resultado entrenado "
                "con ambos modelos."
            )

        return plan

    def print_plan_report(self, plan: ToolWearPlan):
        lineas = [
            "",
            "═" * 55,
            "  AGENTE 2 (desgaste) — PLAN DE ESTRATEGIA",
            "═" * 55,
            f"  Puede entrenar:       {'✓' if plan.can_train else '✗'}",
            f"  Split LOEO:           {'✓' if plan.usa_loeo else '✗ (aleatorio 80/20)'}",
            f"  Ensamble RF+XGBoost:  {'✓' if plan.usara_ensemble else '✗ (solo RF)'}",
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
