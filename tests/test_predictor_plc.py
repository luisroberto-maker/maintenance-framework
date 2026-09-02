"""
tests/test_predictor_plc.py — AdaptiveFailurePredictor (vertical PLC,
framework principal).

Incluye pruebas de regresión explícitas para los bugs de compatibilidad
con pandas 3.0 encontrados durante el uso real del sistema.
"""
import numpy as np
import pandas as pd
import pytest

from core.config import AgentConfig, ModelStrategy
from core.predictor import AdaptiveFailurePredictor, SafeEncoder


# ─── Entrenamiento y predicción end-to-end ─────────────────────────────────────

def test_fit_predict_end_to_end(df_plc_events, mock_plc_embeddings):
    """
    Prueba de regresión para el bug de KeyError:
    "['tiempo_desde_ultima_falla', ...] not in index"

    Este bug ocurría porque fit() descartaba el DataFrame con las columnas
    de features ya construidas (el 4to valor de retorno de
    _build_features) y luego intentaba entrenar los modelos locales sobre
    el DataFrame crudo, que no tenía esas columnas.
    """
    cfg = AgentConfig(umbral_critico=30, ventana_dias=7, min_eventos_modelo_local=30)
    predictor = AdaptiveFailurePredictor(cfg=cfg, strategy=ModelStrategy.HYBRID)

    metricas = predictor.fit(df_plc_events)

    assert "auc_roc" in metricas
    assert 0.0 <= metricas["auc_roc"] <= 1.0
    assert predictor.modelo_global is not None

    ranking = predictor.predict_all(df_plc_events)
    assert len(ranking) > 0
    assert set(["ADDRESS", "PROB_FALLA", "NIVEL_RIESGO", "MODO"]).issubset(ranking.columns)
    assert ranking["PROB_FALLA"].between(0, 1).all()


def test_columna_address_se_conserva_tras_build_features(df_plc_events, mock_plc_embeddings):
    """
    Prueba de regresión para el bug de pandas 3.0:
    groupby(...).apply() ya no conserva la columna de agrupación (ADDRESS)
    en el resultado, ni siquiera dentro de la función aplicada — rompía
    tanto _build_features() como _calcular_target().
    """
    from core.predictor import SafeEncoder, TODAS_CATEGORIAS

    cfg = AgentConfig()
    predictor = AdaptiveFailurePredictor(cfg=cfg)

    df_prep = predictor._preparar(df_plc_events)
    assert "ADDRESS" in df_prep.columns
    assert "TARGET" in df_prep.columns

    # _build_features() normalmente se llama desde fit(), que ya inicializó
    # los encoders — aquí se replica ese paso mínimo para probar
    # _build_features() de forma aislada.
    categorias = sorted(df_prep["CATEGORIA"].unique().tolist())
    for c in TODAS_CATEGORIAS:
        if c not in categorias:
            categorias.append(c)
    predictor.encoder_categoria = SafeEncoder("CATEGORIA", categorias)
    predictor.encoder_panel = SafeEncoder("PANEL", sorted(df_prep["PANEL"].unique().tolist()))

    df_emb = predictor._generar_embeddings(df_prep, fit_pca=True)
    X, y, meta, df_model = predictor._build_features(df_emb)
    assert "ADDRESS" in df_model.columns
    assert not df_model["ADDRESS"].isna().any()


def test_estrategia_heuristica_con_pocos_datos(df_plc_events_sparse, agent_config, mock_plc_embeddings):
    """Con muy pocos eventos, el Agente 2 debe elegir HEURISTIC y el
    sistema no debe intentar entrenar un modelo ML."""
    from agents.agent1_analyzer import DataAnalyzerAgent
    from agents.agent2_planner import StrategyPlannerAgent

    agent1 = DataAnalyzerAgent(agent_config)
    df, schema = agent1.analyze(df_plc_events_sparse)
    assert schema.is_valid

    agent2 = StrategyPlannerAgent(agent_config)
    plan = agent2.plan(df, schema)
    assert plan.model_strategy == ModelStrategy.HEURISTIC
    assert plan.can_train is False


# ─── SafeEncoder ───────────────────────────────────────────────────────────────

def test_safe_encoder_categoria_conocida():
    enc = SafeEncoder("CATEGORIA", ["A", "B", "C"])
    resultado = enc.transform(pd.Series(["A", "B", "C"]))
    assert list(resultado) == [0.0, 1.0, 2.0]


def test_safe_encoder_categoria_desconocida_no_lanza_excepcion():
    """Un componente/categoría nunca visto en entrenamiento debe recibir
    código -1, no lanzar una excepción — esto es lo que permite que el
    sistema funcione con componentes nuevos en producción."""
    enc = SafeEncoder("CATEGORIA", ["A", "B", "C"])
    resultado = enc.transform(pd.Series(["A", "CATEGORIA_NUNCA_VISTA"]))
    assert resultado[0] == 0.0
    assert resultado[1] == -1.0


def test_safe_encoder_con_columna_de_texto_pandas3(df_plc_events):
    """
    Prueba de regresión para el bug de Arrow-backed strings en pandas 3.0:
    series.values.reshape(-1, 1) fallaba con
    "AttributeError / NotImplementedError" en columnas de texto respaldadas
    por ArrowStringArray. La solución usa np.asarray() en vez de .values.
    """
    enc = SafeEncoder("PANEL", ["LINEA_1", "LINEA_2"])
    # Simula una columna de texto tal como la produce pandas 3.0 por defecto
    serie_texto = df_plc_events["PANEL"].astype("string")
    resultado = enc.transform(serie_texto)
    assert len(resultado) == len(df_plc_events)


# ─── Componentes nuevos ──────────────────────────────────────────────────────

def test_prediccion_con_componente_nuevo_no_falla(df_plc_events, mock_plc_embeddings):
    """Un componente nunca visto en entrenamiento debe recibir una
    predicción (vía heurístico o modelo global), no una excepción."""
    cfg = AgentConfig(umbral_critico=30, ventana_dias=7)
    predictor = AdaptiveFailurePredictor(cfg=cfg)
    predictor.fit(df_plc_events)

    df_nuevo = df_plc_events.tail(5).copy()
    df_nuevo["ADDRESS"] = "COMPONENTE_JAMAS_VISTO"

    ranking = predictor.predict_all(pd.concat([df_plc_events, df_nuevo]))
    fila_nueva = ranking[ranking["ADDRESS"] == "COMPONENTE_JAMAS_VISTO"]
    assert len(fila_nueva) == 1
    assert 0.0 <= fila_nueva.iloc[0]["PROB_FALLA"] <= 1.0
