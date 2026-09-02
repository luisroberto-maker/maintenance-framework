"""
tests/test_agent2_tool_wear_planner.py — Agente 2 de la vertical de
desgaste de herramienta (ToolWearStrategyPlanner), agregado como parte de
la unificación del patrón de 4 agentes entre verticales.
"""
import pandas as pd

from agents.agent2_tool_wear_planner import ToolWearStrategyPlanner
from agents.agent1_tool_wear_loader import ToolWearDataLoaderAgent


def test_plan_con_multiples_experimentos_usa_loeo(df_tool_wear_multi_experiment):
    loader = ToolWearDataLoaderAgent()
    df, schema = loader.analyze(df_tool_wear_multi_experiment)

    planner = ToolWearStrategyPlanner()
    plan = planner.plan(df, schema)

    assert plan.can_train is True
    assert plan.usa_loeo is True
    assert not any("LOEO" in w for w in plan.warnings)  # con múltiples experimentos no debe advertir


def test_plan_con_un_solo_experimento_advierte_split_aleatorio(df_tool_wear_single_experiment):
    loader = ToolWearDataLoaderAgent()
    df, schema = loader.analyze(df_tool_wear_single_experiment)

    planner = ToolWearStrategyPlanner()
    plan = planner.plan(df, schema)

    assert plan.can_train is True
    assert plan.usa_loeo is False
    assert any("LOEO" in w or "aleatorio" in w for w in plan.warnings)


def test_plan_con_muy_pocas_pasadas_no_permite_entrenar(feat_cols):
    df_minimo = pd.DataFrame([
        {**{c: 0.1 for c in feat_cols}, "experiment": "x", "cut": i, "VB": 0.05}
        for i in range(1, 5)  # menos que MIN_PASADAS_PARA_ENTRENAR
    ])
    loader = ToolWearDataLoaderAgent()
    df, schema = loader.analyze(df_minimo)

    planner = ToolWearStrategyPlanner()
    plan = planner.plan(df, schema)

    assert plan.can_train is False


def test_advertencia_de_loeo_no_se_duplica_entre_agente1_y_agente2(df_tool_wear_single_experiment):
    """
    Prueba de regresión de diseño: antes de unificar el patrón, la
    advertencia de "un solo experimento, sin LOEO" se generaba dentro del
    Agente 1 (agent1_tool_wear_loader.py). Ahora es responsabilidad del
    Agente 2 — el Agente 1 solo debe reportar observaciones de calidad de
    datos crudos, no decisiones de estrategia de evaluación.
    """
    loader = ToolWearDataLoaderAgent()
    df, schema = loader.analyze(df_tool_wear_single_experiment)

    assert not any("LOEO" in w for w in schema.warnings), (
        "La advertencia de LOEO no debe originarse en el Agente 1 — "
        "es responsabilidad del Agente 2 (ToolWearStrategyPlanner)."
    )
