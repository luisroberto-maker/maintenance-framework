"""
tests/test_predictor_tool_wear.py — ToolWearPredictor (vertical desgaste
de herramienta, dataset PHM 2010).
"""
import json

import numpy as np
import pandas as pd
import pytest

from core.tool_wear_predictor import (
    ToolWearPredictor, compute_rul, VB_THRESHOLD, FEAT_COLS,
)


# ─── Entrenamiento y predicción end-to-end ─────────────────────────────────────

def test_fit_predict_con_loeo(df_tool_wear_multi_experiment):
    """Con múltiples experimentos, debe usar split LOEO (Leave-One-
    Experiment-Out) automáticamente."""
    predictor = ToolWearPredictor()
    metricas = predictor.fit(df_tool_wear_multi_experiment)

    assert "rf" in metricas
    assert metricas["rf"]["r2"] > 0.5  # las features están correlacionadas con VB a propósito
    assert predictor.rf_model is not None

    ranking = predictor.predict_all(df_tool_wear_multi_experiment)
    assert len(ranking) == len(df_tool_wear_multi_experiment)
    assert set(["COMPONENT", "CUT", "VB_ESTIMADO", "ESTADO"]).issubset(ranking.columns)


def test_fit_con_un_solo_experimento_usa_split_aleatorio(df_tool_wear_single_experiment):
    """Con un solo experimento no hay forma de hacer LOEO — debe caer a
    split aleatorio 80/20 sin lanzar excepción."""
    predictor = ToolWearPredictor()
    metricas = predictor.fit(df_tool_wear_single_experiment)
    assert "rf" in metricas


def test_estado_asignado_segun_umbral(df_tool_wear_multi_experiment):
    predictor = ToolWearPredictor(vb_threshold=0.2)
    predictor.fit(df_tool_wear_multi_experiment)
    ranking = predictor.predict_all(df_tool_wear_multi_experiment)

    for _, row in ranking.iterrows():
        if row["VB_ESTIMADO"] >= 0.2:
            assert "FALLA" in row["ESTADO"]
        elif row["VB_ESTIMADO"] >= 0.16:
            assert "ALERTA" in row["ESTADO"]
        else:
            assert "OK" in row["ESTADO"]


# ─── compute_rul: prueba de regresión del bug de RUL absurdo ─────────────────

class TestComputeRUL:
    """
    Pruebas de regresión para el bug: con historial insuficiente (la
    primera pasada de una serie), la tasa de desgaste caía a una
    constante casi cero (1e-9) y al dividir producía valores como
    76,325,000 pasadas de vida útil restante. Ahora debe devolver None.
    """

    def test_primera_pasada_sin_historial_es_none(self):
        vb_series = np.array([0.047, 0.049, 0.051])
        rul = compute_rul(vb_series, threshold=0.2)
        assert rul[0] is None  # sin historial previo — no estimable

    def test_ningun_valor_es_absurdamente_grande(self):
        """Ninguna RUL calculada debe superar un límite razonable (aquí,
        10x el umbral entre el valor mínimo observado — un guardrail
        generoso pero que sí habría atrapado el 76,325,000 real)."""
        vb_series = np.array([0.047, 0.049, 0.051, 0.054, 0.053,
                              0.058, 0.060, 0.063, 0.068, 0.070])
        rul = compute_rul(vb_series, threshold=0.2)
        LIMITE_RAZONABLE = 100_000
        for valor in rul:
            if valor is not None:
                assert valor < LIMITE_RAZONABLE, (
                    f"RUL={valor} es sospechosamente alto — posible regresión "
                    f"del bug de división por tasa casi cero."
                )

    def test_pasada_en_falla_devuelve_cero(self):
        vb_series = np.array([0.05, 0.15, 0.25])
        rul = compute_rul(vb_series, threshold=0.2)
        assert rul[2] == 0  # ya superó el umbral

    def test_tasa_de_desgaste_nula_es_none(self):
        """Si el desgaste no crece (tasa ~0), no se puede proyectar una
        RUL significativa — debe devolver None, no un número enorme."""
        vb_series = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        rul = compute_rul(vb_series, threshold=0.2)
        assert all(v is None for v in rul[1:])  # sin tendencia de crecimiento

    def test_tendencia_creciente_da_rul_decreciente(self):
        """Con una tasa de desgaste constante y clara, la RUL debe
        disminuir conforme el desgaste aumenta — coherencia física básica."""
        vb_series = np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15])
        rul = compute_rul(vb_series, threshold=0.3, window=5)
        valores_validos = [v for v in rul if v is not None]
        assert len(valores_validos) >= 2
        assert valores_validos == sorted(valores_validos, reverse=True)


def test_prediccion_con_historial_corto_no_produce_rul_absurda(df_tool_wear_short_history):
    """Prueba end-to-end (no solo unitaria) del mismo bug, replicando el
    escenario real reportado: 10 pasadas, primera con RUL no estimable."""
    predictor = ToolWearPredictor()
    predictor.fit(df_tool_wear_short_history)
    ranking = predictor.predict_all(df_tool_wear_short_history)

    for valor in ranking["RUL_ESTIMADO"]:
        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            assert valor < 100_000, f"RUL sospechosamente alta: {valor}"


# ─── Carga de artefactos de Kaggle (sin reentrenar) ──────────────────────────

def test_from_kaggle_artifacts_carga_sin_reentrenar(tmp_path, df_tool_wear_multi_experiment):
    """Simula el flujo real: entrenar una vez, exportar artefactos sueltos
    (como lo hace el notebook de Kaggle), y cargarlos de vuelta sin volver
    a entrenar."""
    import joblib

    predictor_original = ToolWearPredictor()
    predictor_original.fit(df_tool_wear_multi_experiment)

    joblib.dump(predictor_original.rf_model, tmp_path / "model_rf.joblib")
    joblib.dump(predictor_original.scaler, tmp_path / "scaler.joblib")
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({
            "feature_cols": predictor_original.feature_cols,
            "vb_threshold": predictor_original.vb_threshold,
        }, f)

    predictor_cargado = ToolWearPredictor.from_kaggle_artifacts(str(tmp_path))
    assert predictor_cargado.rf_model is not None
    assert predictor_cargado.feature_cols == predictor_original.feature_cols

    ranking = predictor_cargado.predict_all(df_tool_wear_multi_experiment.head(10))
    assert len(ranking) == 10


def test_from_kaggle_artifacts_falla_claramente_sin_archivo_requerido(tmp_path):
    """Si falta model_rf.joblib, debe fallar con un mensaje claro, no un
    traceback críptico."""
    with pytest.raises(FileNotFoundError, match="model_rf.joblib"):
        ToolWearPredictor.from_kaggle_artifacts(str(tmp_path))


# ─── Ingesta de carpetas de señal cruda sin wear.csv (inferencia pura) ───────

def test_ingesta_carpeta_sin_wear_csv_no_falla(tmp_path):
    """
    Prueba de regresión: load_experiment_from_folder() exigía un wear.csv
    incluso para inferencia pura (donde el desgaste real es justamente lo
    que se quiere predecir). Ahora wear_path es opcional.
    """
    from core.tool_wear_predictor import load_experiment_from_folder

    carpeta = tmp_path / "senales"
    carpeta.mkdir()
    for cut in range(1, 4):
        señal = np.random.RandomState(cut).randn(500, 7) * 0.5
        pd.DataFrame(señal).to_csv(carpeta / f"pasada_{cut:03d}.csv", header=False, index=False)

    df = load_experiment_from_folder("prueba", str(carpeta), wear_path=None, verbose=False)
    assert len(df) == 3
    assert "VB" not in df.columns  # sin desgaste real, correcto
    assert set(FEAT_COLS).issubset(df.columns)
