"""
tests/conftest.py — Fixtures compartidos para todo el paquete de pruebas.

Estos fixtures generan datos sintéticos con las mismas características que
causaron bugs reales durante el desarrollo (duraciones variables para que
ambas clases estén presentes, embeddings simulados para no depender de red,
múltiples experimentos para habilitar LOEO, etc.). Cada fixture documenta
qué bug ayuda a prevenir.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# Añadir la raíz del proyecto al path para que "core" y "agents" se importen
# igual sin importar desde dónde se invoque pytest.
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Modelo de embeddings simulado (sin red) ──────────────────────────────────

class FakeEmbeddingModel:
    """
    Reemplazo determinista de SentenceTransformer para pruebas.

    Usar el modelo real de Hugging Face en pruebas automatizadas es lento
    (descarga ~91 MB la primera vez) y frágil (falla sin internet, como en
    CI). Este stub genera vectores aleatorios pero DETERMINISTAS por texto
    (mismo texto → mismo vector, usando un hash), lo cual es suficiente para
    probar la lógica del pipeline sin depender de la calidad semántica real.
    """
    def encode(self, textos, show_progress_bar=False):
        vectores = []
        for t in textos:
            seed = abs(hash(t)) % (2**32)
            rng = np.random.RandomState(seed)
            vectores.append(rng.randn(384))
        return np.array(vectores)


@pytest.fixture
def fake_embedding_model():
    return FakeEmbeddingModel()


@pytest.fixture(autouse=False)
def mock_plc_embeddings(monkeypatch):
    """Reemplaza el modelo de embeddings en AdaptiveFailurePredictor y
    LegacyPLCPredictor por el stub determinista, en todas las pruebas que
    lo soliciten explícitamente como parámetro."""
    from core.predictor import AdaptiveFailurePredictor
    from core.legacy_plc_predictor import LegacyPLCPredictor

    fake = FakeEmbeddingModel()
    monkeypatch.setattr(AdaptiveFailurePredictor, "_get_embedding_model", lambda self: fake)
    monkeypatch.setattr(LegacyPLCPredictor, "_get_embedding_model", lambda self: fake)
    return fake


# ─── Datos sintéticos: eventos PLC ────────────────────────────────────────────

@pytest.fixture
def df_plc_events():
    """
    Eventos PLC sintéticos con duración VARIABLE.

    Bug que previene: si todos los eventos tienen la misma duración por
    debajo del umbral_critico, TARGET es siempre 0, ningún modelo puede
    entrenar con una sola clase, y predict_proba()[:, 1] lanza IndexError
    ("index 1 is out of bounds for axis 1 with size 1"). Ver el historial
    de esta conversación: este bug apareció dos veces por datos sintéticos
    mal construidos, antes de encontrar el bug real de compatibilidad con
    pandas 3.0 que estaba detrás.
    """
    np.random.seed(42)
    n = 400
    start = pd.date_range("2024-01-01", periods=n, freq="3h")
    duracion_min = np.where(
        np.random.rand(n) < 0.15,
        np.random.uniform(30, 90, n),   # ~15% críticas
        np.random.uniform(1, 25, n),    # el resto, breves
    )
    end = start + pd.to_timedelta(duracion_min, unit="m")
    return pd.DataFrame({
        "START TIME": start,
        "END TIME": end,
        # DURACION_MIN normalmente la calcula el Agente 1 (Agent1_analyzer)
        # antes de llegar al predictor — se incluye aquí para que los
        # fixtures reflejen fielmente lo que el predictor recibe en
        # producción, no solo lo que un usuario cargaría en crudo.
        "DURACION_MIN": duracion_min,
        "ADDRESS": np.random.choice(["COMP_A", "COMP_B", "COMP_C", "COMP_D"], n),
        "FAIL COMMENT": np.random.choice(
            ["Emergency Stop", "PLC Connect Error(-1)", "C1 Heavy Error", "R/B-1 FAULT"], n
        ),
        "PANEL": np.random.choice(["LINEA_1", "LINEA_2"], n),
    })


@pytest.fixture
def df_plc_events_sparse():
    """Un dataset PLC pequeño (por debajo del umbral de entrenamiento) para
    probar la ruta HEURISTIC del Agente 2."""
    np.random.seed(1)
    n = 15
    start = pd.date_range("2024-01-01", periods=n, freq="6h")
    return pd.DataFrame({
        "START TIME": start,
        "END TIME": start + pd.Timedelta(minutes=10),
        "ADDRESS": np.random.choice(["COMP_X", "COMP_Y"], n),
        "FAIL COMMENT": np.random.choice(["Emergency Stop", "C1 Heavy Error"], n),
        "PANEL": "LINEA_1",
    })


# ─── Datos sintéticos: desgaste de herramienta (PHM 2010) ────────────────────

@pytest.fixture
def feat_cols():
    from core.tool_wear_predictor import FEAT_COLS
    return FEAT_COLS


def _simulate_tool_wear_experiment(feat_cols, exp_name, n_cuts=150, wear_rate=0.0012, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    vb = 0.0
    for cut in range(1, n_cuts + 1):
        vb += wear_rate * (1 + 0.3 * np.sin(cut / 20)) + rng.normal(0, 0.0005)
        vb = max(0, vb)
        # Features correlacionadas con VB para que el modelo aprenda algo real,
        # no ruido puro — importante para que las pruebas de R² tengan sentido.
        feat = {c: vb * 10 + rng.normal(0, 0.5) for c in feat_cols}
        feat.update({"experiment": exp_name, "cut": cut, "VB": vb})
        rows.append(feat)
    return pd.DataFrame(rows)


@pytest.fixture
def df_tool_wear_multi_experiment(feat_cols):
    """
    Tres experimentos (como c1/c4/c6 del dataset PHM 2010 real), habilitando
    el split LOEO (Leave-One-Experiment-Out).
    """
    return pd.concat([
        _simulate_tool_wear_experiment(feat_cols, "c1", 150, 0.0013, seed=1),
        _simulate_tool_wear_experiment(feat_cols, "c4", 150, 0.0011, seed=2),
        _simulate_tool_wear_experiment(feat_cols, "c6", 150, 0.0012, seed=3),
    ], ignore_index=True)


@pytest.fixture
def df_tool_wear_single_experiment(feat_cols):
    """Un solo experimento — debe caer al split aleatorio 80/20 con
    advertencia, en vez de LOEO."""
    return _simulate_tool_wear_experiment(feat_cols, "c_unico", 100, 0.0012, seed=4)


@pytest.fixture
def df_tool_wear_short_history(feat_cols):
    """
    Muy pocas pasadas (10) — el caso real que produjo el bug de RUL
    (76,325,000 pasadas de vida útil restante) por dividir entre una tasa
    de desgaste casi cero cuando no hay suficiente historial.
    """
    return _simulate_tool_wear_experiment(feat_cols, "herramienta_nueva", 10, 0.0012, seed=5)


# ─── Configuración con directorios temporales ─────────────────────────────────

@pytest.fixture
def agent_config(tmp_path):
    from core.config import AgentConfig
    return AgentConfig(
        output_dir=str(tmp_path / "output"),
        model_dir=str(tmp_path / "output" / "models"),
        report_dir=str(tmp_path / "output" / "reports"),
    )
