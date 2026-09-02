"""
tests/test_orchestrator.py — Pruebas de integración de
MultiVerticalAgentSystem: los flujos completos run() y predict_only()
para las tres variantes de predictor.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from core.config import AgentConfig
from orchestrator import MultiVerticalAgentSystem


@pytest.fixture
def system(agent_config):
    return MultiVerticalAgentSystem(config=agent_config)


# ─── run(): entrenar desde cero ────────────────────────────────────────────────

def test_run_vertical_plc(system, df_plc_events, mock_plc_embeddings):
    reporte = system.run(df_plc_events, client_id="test_plc", verbose=False)
    assert reporte.endswith(".html")
    assert os.path.exists(reporte)

    ranking = system.get_ranking()
    assert ranking is not None and len(ranking) > 0


def test_run_vertical_tool_wear(system, df_tool_wear_multi_experiment):
    reporte = system.run(df_tool_wear_multi_experiment, client_id="test_wear", verbose=False)
    assert os.path.exists(reporte)

    ranking = system.get_ranking()
    assert ranking is not None and len(ranking) > 0
    assert "VB_ESTIMADO" in ranking.columns


def test_run_guarda_modelo_para_reutilizar_despues(system, df_plc_events, agent_config, mock_plc_embeddings):
    """El modelo entrenado debe quedar guardado en disco, listo para
    predict_only() sin reentrenar."""
    system.run(df_plc_events, client_id="test_persist", verbose=False)
    ruta_esperada = os.path.join(agent_config.model_dir, "predictor_test_persist.pkl")
    assert os.path.exists(ruta_esperada)


# ─── predict_only(): inferencia sin reentrenar ───────────────────────────────

def test_predict_only_framework_plc(system, df_plc_events, agent_config, mock_plc_embeddings):
    system.run(df_plc_events, client_id="cliente_a", verbose=False)

    # Nueva instancia del sistema simula "otra sesión" — el modelo debe
    # cargarse de disco, no reentrenar.
    system2 = MultiVerticalAgentSystem(config=agent_config)
    reporte = system2.predict_only(
        source=df_plc_events.sample(50, random_state=1),
        client_id="cliente_a",
        domain="plc_failure",
    )
    assert reporte is not None
    ranking = system2.get_ranking()
    assert ranking is not None and len(ranking) > 0


def test_predict_only_tool_wear_con_artefactos_kaggle(system, df_tool_wear_multi_experiment,
                                                       agent_config, tmp_path):
    """Simula el flujo de Kaggle: entrenar, exportar 4 archivos sueltos,
    cargar con artifacts_dir sin reentrenar."""
    import joblib
    from core.tool_wear_predictor import ToolWearPredictor

    predictor = ToolWearPredictor()
    predictor.fit(df_tool_wear_multi_experiment)

    artifacts_dir = tmp_path / "kaggle_artifacts"
    artifacts_dir.mkdir()
    joblib.dump(predictor.rf_model, artifacts_dir / "model_rf.joblib")
    joblib.dump(predictor.scaler, artifacts_dir / "scaler.joblib")
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump({"feature_cols": predictor.feature_cols,
                  "vb_threshold": predictor.vb_threshold}, f)

    reporte = system.predict_only(
        source=df_tool_wear_multi_experiment.head(20),
        client_id="kaggle_test",
        domain="tool_wear",
        artifacts_dir=str(artifacts_dir),
        verbose=False,
    )
    ranking = system.get_ranking()
    assert ranking is not None and len(ranking) == 20


def test_predict_only_legacy_plc(system, df_plc_events, agent_config, mock_plc_embeddings):
    import joblib
    from core.legacy_plc_predictor import LegacyPLCPredictor

    predictor = LegacyPLCPredictor(
        umbral_critico=agent_config.umbral_critico,
        ventana_dias=agent_config.ventana_dias,
        n_eventos=agent_config.default_n_eventos,
        n_componentes_pca=agent_config.n_pca_components,
    )
    predictor.fit(df_plc_events)

    os.makedirs(agent_config.model_dir, exist_ok=True)
    modelo_path = os.path.join(agent_config.model_dir, "modelo_legacy_test.pkl")
    joblib.dump(predictor.modelo, modelo_path)

    reporte = system.predict_only(
        source=df_plc_events.sample(50, random_state=2),
        client_id="legacy_test",
        domain="plc_failure",
        legacy=True,
        model_path=modelo_path,
        verbose=False,
    )
    assert reporte is not None
    ranking = system.get_ranking()
    assert ranking is not None and len(ranking) > 0
    assert "PROBABILIDAD_FALLA" in ranking.columns


def test_predict_only_sin_modelo_previo_falla_claramente(system, df_plc_events):
    """Si no hay ningún modelo entrenado en la ruta esperada, debe fallar
    con FileNotFoundError, no con un error de atributo críptico."""
    with pytest.raises(FileNotFoundError):
        system.predict_only(
            source=df_plc_events,
            client_id="cliente_sin_modelo_previo",
            domain="plc_failure",
            verbose=False,
        )


# ─── Enrutamiento automático de dominio en el orquestador ────────────────────

def test_run_detecta_dominio_automaticamente_plc(system, df_plc_events, mock_plc_embeddings):
    system.run(df_plc_events, client_id="auto_plc", verbose=False)
    from agents.domain_router import Domain
    assert system.last_domain == Domain.PLC_FAILURE


def test_run_detecta_dominio_automaticamente_desgaste(system, df_tool_wear_multi_experiment):
    system.run(df_tool_wear_multi_experiment, client_id="auto_wear", verbose=False)
    from agents.domain_router import Domain
    assert system.last_domain == Domain.TOOL_WEAR
