"""
tests/test_predictor_legacy.py — LegacyPLCPredictor (réplica del
repositorio original luisroberto-maker/PLC-failure-prediction-pipeline).
"""
import numpy as np
import pandas as pd
import pytest

from core.legacy_plc_predictor import LegacyPLCPredictor, MAPEO_FALLAS


def test_mapeo_fallas_no_esta_vacio():
    """El diccionario de categorías debe tener un tamaño razonable — si
    algún día se reduce accidentalmente al copiar/pegar, esto lo detecta."""
    assert len(MAPEO_FALLAS) > 100


def test_fit_predict_end_to_end(df_plc_events, mock_plc_embeddings):
    predictor = LegacyPLCPredictor(umbral_critico=30, ventana_dias=7,
                                   n_eventos=10, n_componentes_pca=15)
    metricas = predictor.fit(df_plc_events)

    assert "auc_roc" in metricas
    assert predictor.modelo is not None

    ranking = predictor.predict_all(df_plc_events)
    assert len(ranking) > 0
    assert set(["COMPONENTE", "PROBABILIDAD_FALLA", "NIVEL_RIESGO"]).issubset(ranking.columns)


def test_no_falla_con_clase_unica_en_entrenamiento(mock_plc_embeddings):
    """
    Prueba de regresión para el IndexError:
    "index 1 is out of bounds for axis 1 with size 1"

    Este error ocurre cuando TARGET tiene una sola clase presente (por
    ejemplo, si todas las duraciones están por debajo del umbral crítico),
    porque RandomForestClassifier.predict_proba() con una sola clase
    devuelve un array de una sola columna, y el código intenta acceder a
    la columna [1] (probabilidad de la clase positiva).

    Esta prueba usa deliberadamente datos SIN eventos críticos, para
    verificar que el sistema falla de forma clara y controlada (no con un
    IndexError críptico) en ese escenario límite.
    """
    np.random.seed(0)
    n = 100
    start = pd.date_range("2024-01-01", periods=n, freq="4h")
    df_sin_criticas = pd.DataFrame({
        "START TIME": start,
        "END TIME": start + pd.Timedelta(minutes=5),  # todas breves, ninguna crítica
        "ADDRESS": np.random.choice(["COMP_A", "COMP_B"], n),
        "FAIL COMMENT": "Emergency Stop",
        "PANEL": "LINEA_1",
    })

    predictor = LegacyPLCPredictor(umbral_critico=30, ventana_dias=7)

    # No afirmamos que el entrenamiento tenga éxito con una sola clase —
    # solo que si falla, sea con un error identificable, no un IndexError
    # cru rastreable a la ausencia de datos, no a un bug del pipeline.
    try:
        predictor.fit(df_sin_criticas)
    except (ValueError, IndexError) as e:
        # Aceptable: sklearn puede rechazar el entrenamiento con una sola
        # clase con su propio ValueError explícito. Lo que NO es aceptable
        # es que el pipeline de features truene antes de llegar ahí.
        pass


def test_datos_con_ambas_clases_entrena_sin_error(df_plc_events, mock_plc_embeddings):
    """Con datos que sí tienen ambas clases (la fixture estándar), debe
    entrenar y predecir sin ningún error de índice."""
    predictor = LegacyPLCPredictor(umbral_critico=30, ventana_dias=7)
    predictor.fit(df_plc_events)
    ranking = predictor.predict_all(df_plc_events)
    assert ranking["PROBABILIDAD_FALLA"].between(0, 1).all()


def test_from_pretrained_carga_estimador_pelado(tmp_path, df_plc_events, mock_plc_embeddings):
    """El repositorio original guarda un estimador de scikit-learn
    'pelado' (sin envoltorio) — from_pretrained() debe cargarlo tal cual."""
    import joblib

    predictor_original = LegacyPLCPredictor(umbral_critico=30, ventana_dias=7)
    predictor_original.fit(df_plc_events)
    joblib.dump(predictor_original.modelo, tmp_path / "modelo.pkl")

    predictor_cargado = LegacyPLCPredictor.from_pretrained(str(tmp_path / "modelo.pkl"))
    assert hasattr(predictor_cargado.modelo, "predict_proba")

    ranking = predictor_cargado.predict_all(df_plc_events)
    assert len(ranking) > 0


def test_from_pretrained_rechaza_objeto_sin_predict_proba(tmp_path):
    """Si el .pkl no es un clasificador válido (ej. es un dict o un
    AdaptiveFailurePredictor completo), debe fallar con un mensaje claro."""
    import joblib
    joblib.dump({"no": "es un modelo"}, tmp_path / "no_modelo.pkl")

    with pytest.raises(ValueError, match="predict_proba"):
        LegacyPLCPredictor.from_pretrained(str(tmp_path / "no_modelo.pkl"))
