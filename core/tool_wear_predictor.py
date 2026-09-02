"""
core/tool_wear_predictor.py — ToolWearPredictor.

Segunda vertical del framework: predicción de desgaste de herramienta (VB)
y vida útil restante (RUL) a partir de señales de sensor (fuerza, vibración,
emisión acústica), basado en el dataset PHM Society 2010.

La extracción de features (133 por pasada: 19 por canal x 7 canales) y el
esquema de entrenamiento LOEO (Leave-One-Experiment-Out) replican
exactamente el pipeline validado en el notebook original. Este módulo lo
envuelve en la interfaz BasePredictor para que el orquestador pueda tratarlo
igual que al predictor de fallas PLC.
"""
from __future__ import annotations

import glob
import logging
import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from core.base_predictor import BasePredictor, ReportSchema

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception as e:
    # No solo ImportError: en macOS, xgboost puede fallar con OSError al
    # cargar libxgboost.dylib si falta la librería libomp (OpenMP) del
    # sistema — un problema de entorno, no del framework. En cualquier
    # caso, xgboost es opcional: el sistema sigue funcionando solo con
    # Random Forest.
    _HAS_XGB = False
    xgb = None
    logger.warning(
        f"xgboost no disponible ({type(e).__name__}: {e}) — "
        f"ToolWearPredictor usará solo Random Forest. "
        f"Si estás en macOS y el error menciona 'libomp', instala la "
        f"librería con: brew install libomp"
    )


# ─── Constantes del dominio (idénticas al notebook validado) ─────────────────

FS           = 50000            # Hz — frecuencia de muestreo de las señales
RPM          = 10400
N_FLUTES     = 3
ZPF          = RPM * N_FLUTES / 60   # ~520 Hz — frecuencia de paso de filo
VB_THRESHOLD = 0.2               # mm — umbral de falla ISO 8688
MAX_ROWS     = 50000              # 1 s de señal a 50kHz por pasada
SIGNAL_COLS  = ["fx", "fy", "fz", "vx", "vy", "vz", "ae"]

_TF = ["mean", "std", "rms", "peak", "p2p", "crest", "kurtosis", "skewness", "shape", "impulse"]
_FF = ["p0_500", "p500_2k", "p2k_5k", "p5k_10k", "p10k", "zpf1x", "zpf2x", "zpf3x", "entropy"]
FEAT_COLS = [f"{col}_{n}" for col in SIGNAL_COLS for n in _TF + _FF]   # 133 features


# ─── Extracción de features (idéntica al notebook) ────────────────────────────

def time_features(sig: np.ndarray, name: str) -> dict:
    rms = np.sqrt(np.mean(sig ** 2))
    ma = np.mean(np.abs(sig)) + 1e-9
    return {
        f"{name}_mean": float(np.mean(sig)),
        f"{name}_std": float(np.std(sig)),
        f"{name}_rms": float(rms),
        f"{name}_peak": float(np.max(np.abs(sig))),
        f"{name}_p2p": float(np.max(sig) - np.min(sig)),
        f"{name}_crest": float(np.max(np.abs(sig)) / (rms + 1e-9)),
        f"{name}_kurtosis": float(stats.kurtosis(sig)),
        f"{name}_skewness": float(stats.skew(sig)),
        f"{name}_shape": float(rms / ma),
        f"{name}_impulse": float(np.max(np.abs(sig)) / ma),
    }


def freq_features(sig: np.ndarray, name: str) -> dict:
    N = len(sig)
    yf = np.abs(fft(sig))[: N // 2]
    xf = fftfreq(N, 1 / FS)[: N // 2]
    tp = np.sum(yf ** 2) + 1e-9

    def bp(a, b):
        return float(np.sum(yf[(xf >= a) & (xf < b)] ** 2) / tp)

    def ha(h, bw=50):
        fc = h * ZPF
        m = (xf >= fc - bw) & (xf <= fc + bw)
        return float(np.max(yf[m])) if m.sum() > 0 else 0.0

    return {
        f"{name}_p0_500": bp(0, 500),
        f"{name}_p500_2k": bp(500, 2000),
        f"{name}_p2k_5k": bp(2000, 5000),
        f"{name}_p5k_10k": bp(5000, 10000),
        f"{name}_p10k": bp(10000, FS / 2),
        f"{name}_zpf1x": ha(1),
        f"{name}_zpf2x": ha(2),
        f"{name}_zpf3x": ha(3),
        f"{name}_entropy": float(-np.sum((yf ** 2 / tp) * np.log(yf ** 2 / tp + 1e-9))),
    }


def extract_features(filepath: str) -> dict:
    """Lee un CSV de señal cruda (fx,fy,fz,vx,vy,vz,ae, sin header) en chunks
    y extrae 133 features sin cargar la señal completa en memoria."""
    chunks, total = [], 0
    for chunk in pd.read_csv(filepath, header=None, chunksize=10000):
        chunks.append(chunk.values.astype(np.float32))
        total += len(chunk)
        if total >= MAX_ROWS:
            break
    sig = np.vstack(chunks)[:MAX_ROWS]
    feat = {}
    for ch, col in enumerate(SIGNAL_COLS):
        s = sig[:, ch].astype(np.float64)
        feat.update(time_features(s, col))
        feat.update(freq_features(s, col))
    return feat


def load_wear(wear_path: str) -> tuple[dict, float]:
    """Carga un wear file y devuelve {cut -> VB_max en mm}, detectando la escala."""
    df = pd.read_csv(wear_path, header=None, names=["cut", "flute_1", "flute_2", "flute_3"])
    if str(df["cut"].iloc[0]).strip().lower() == "cut":
        df = df.iloc[1:].reset_index(drop=True)
    df = df.astype({"cut": int, "flute_1": float, "flute_2": float, "flute_3": float})
    vb_raw = df[["flute_1", "flute_2", "flute_3"]].max(axis=1)
    scale = 1000.0 if vb_raw.max() > 5 else 1.0
    df["VB"] = vb_raw / scale
    return df.set_index("cut")["VB"].to_dict(), scale


def load_experiment_from_folder(exp: str, signals_dir: str, wear_path: Optional[str] = None,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Ingesta de datos crudos: carpeta con un CSV de señal por pasada,
    opcionalmente acompañada de un wear file. Reconstruye el DataFrame de
    features tal como en el notebook original.

    wear_path es OPCIONAL: para entrenamiento necesitas el desgaste medido
    (VB) de cada pasada, pero para inferencia pura sobre datos nuevos —
    donde el desgaste real todavía no se conoce, que es justamente lo que
    se quiere predecir— se puede omitir. Sin wear_path, se extraen las
    features de TODAS las pasadas encontradas en la carpeta, sin filtrar
    por las que tengan desgeste medido.
    """
    wear_dict = {}
    if wear_path is not None:
        wear_dict, _ = load_wear(wear_path)

    files = sorted(glob.glob(os.path.join(signals_dir, "*.csv")))
    if not files:
        logger.warning(f"No se encontraron archivos .csv en {signals_dir}")
        return pd.DataFrame()

    rows = []
    t0 = time.time()
    for i, fpath in enumerate(files):
        base = os.path.splitext(os.path.basename(fpath))[0]
        try:
            cut = int(base.split("_")[-1])
        except ValueError:
            logger.warning(f"No se pudo extraer el número de pasada de '{base}' — se omite.")
            continue

        if wear_path is not None and cut not in wear_dict:
            # Hay wear file pero esta pasada específica no está en él — se omite
            # (comportamiento de entrenamiento: solo se usan pasadas etiquetadas).
            continue

        feat = extract_features(fpath)
        feat["experiment"] = exp
        feat["cut"] = cut
        if cut in wear_dict:
            feat["VB"] = wear_dict[cut]
            feat["label"] = int(wear_dict[cut] > VB_THRESHOLD)
        rows.append(feat)

        if verbose and (i + 1) % 50 == 0:
            logger.info(f"  [{exp}] {i+1}/{len(files)} pasadas procesadas")

    if not rows:
        logger.warning(
            f"No se extrajo ninguna pasada válida de {signals_dir}. "
            f"Verifica el formato de nombre de archivo (se espera terminar en "
            f"'_<número_de_pasada>.csv', ej. 'c1_001.csv')."
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("cut").reset_index(drop=True)
    if verbose:
        logger.info(f"  [{exp}] {len(df)} pasadas extraídas de {len(files)} archivos.")
    return df


def compute_rul(vb_series: np.ndarray, threshold: float = VB_THRESHOLD,
                window: int = 5) -> list:
    """
    Vida útil restante estimada (en número de pasadas) por pendiente local.

    Devuelve None cuando la RUL no es estimable de forma confiable:
      - No hay suficiente historial (menos de 2 pasadas previas), o
      - La tasa de desgaste estimada es insignificante o no creciente
        (podría dar una RUL enorme y sin sentido si se dividiera por un
        número casi cero).
    """
    rul = []
    for i in range(len(vb_series)):
        if vb_series[i] >= threshold:
            rul.append(0)
            continue

        w = vb_series[max(0, i - window): i + 1]
        if len(w) < 2:
            # Sin historial suficiente para estimar una tasa — no se puede
            # calcular una RUL confiable todavía.
            rul.append(None)
            continue

        rate = (w[-1] - w[0]) / (len(w) - 1)
        if rate <= 1e-6:
            # Tasa de desgaste insignificante, nula o decreciente — una RUL
            # calculada aquí sería un número enorme sin significado real.
            rul.append(None)
        else:
            rul.append(max(0, int((threshold - vb_series[i]) / rate)))
    return rul


# ─── Predictor ─────────────────────────────────────────────────────────────────

class ToolWearPredictor(BasePredictor):
    """
    Predictor de desgaste de herramienta (VB) y vida útil restante (RUL)
    a partir de features de señal ya extraídas.

    Espera un DataFrame con las columnas de FEAT_COLS (133 features) más:
      - 'VB'          : desgaste medido en mm (target de entrenamiento)
      - 'experiment'  : identificador del experimento/herramienta (opcional
                        pero recomendado — habilita split LOEO)
      - 'cut'         : número de pasada (opcional — habilita cálculo de RUL)

    Si el dataset trae más de un experimento, el split de evaluación
    reproduce el esquema LOEO del notebook original: el último experimento
    (por orden alfabético/numérico) se reserva como test nunca visto.
    """

    def __init__(self, vb_threshold: float = VB_THRESHOLD, random_state: int = 42):
        self.vb_threshold = vb_threshold
        self.random_state = random_state
        self.scaler: Optional[RobustScaler] = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.xgb_model = None
        self.feature_cols: list[str] = FEAT_COLS
        self.best_model_name: str = "rf"
        self.metrics_: dict = {}

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, test_experiment: Optional[str] = None) -> dict:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Faltan {len(missing)} columnas de features esperadas "
                f"(ej. {missing[:3]}...). ¿El DataFrame pasó por extracción de features?"
            )
        if "VB" not in df.columns:
            raise ValueError("El DataFrame debe incluir la columna target 'VB' (desgaste en mm).")

        df = df.dropna(subset=["VB"]).reset_index(drop=True)

        # Split LOEO si hay múltiples experimentos; si no, split aleatorio con advertencia.
        if "experiment" in df.columns and df["experiment"].nunique() > 1:
            exps = sorted(df["experiment"].unique())
            test_exp = test_experiment or exps[-1]
            df_train = df[df["experiment"] != test_exp].reset_index(drop=True)
            df_test = df[df["experiment"] == test_exp].reset_index(drop=True)
            logger.info(f"Split LOEO: train={[e for e in exps if e != test_exp]}, test={test_exp}")
        else:
            logger.warning(
                "Solo un experimento detectado — usando split aleatorio 80/20. "
                "Para evaluación honesta se recomienda LOEO con ≥2 experimentos."
            )
            df = df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
            split_idx = int(len(df) * 0.8)
            df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

        X_train, y_train = df_train[self.feature_cols].values, df_train["VB"].values
        X_test, y_test = df_test[self.feature_cols].values, df_test["VB"].values

        self.scaler = RobustScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        logger.info("  Entrenando Random Forest...")
        t0 = time.time()
        self.rf_model = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            max_features="sqrt", random_state=self.random_state, n_jobs=-1,
        )
        self.rf_model.fit(X_train_s, y_train)
        y_pred_rf = self.rf_model.predict(X_test_s)
        rmse_rf = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
        mae_rf = float(mean_absolute_error(y_test, y_pred_rf))
        r2_rf = float(r2_score(y_test, y_pred_rf))
        logger.info(f"  RF   → RMSE={rmse_rf:.4f} mm | MAE={mae_rf:.4f} mm | R²={r2_rf:.4f} "
                    f"({time.time()-t0:.0f}s)")

        self.metrics_ = {"rf": {"rmse": rmse_rf, "mae": mae_rf, "r2": r2_rf}}
        self.best_model_name = "rf"

        if _HAS_XGB and len(X_train_s) >= 30:
            logger.info("  Entrenando XGBoost...")
            t0 = time.time()
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.5, min_child_weight=3,
                random_state=self.random_state, verbosity=0,
            )
            self.xgb_model.fit(X_train_s, y_train)
            y_pred_xgb = self.xgb_model.predict(X_test_s)
            rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
            mae_xgb = float(mean_absolute_error(y_test, y_pred_xgb))
            r2_xgb = float(r2_score(y_test, y_pred_xgb))
            logger.info(f"  XGB  → RMSE={rmse_xgb:.4f} mm | MAE={mae_xgb:.4f} mm | R²={r2_xgb:.4f} "
                       f"({time.time()-t0:.0f}s)")
            self.metrics_["xgb"] = {"rmse": rmse_xgb, "mae": mae_xgb, "r2": r2_xgb}
            if rmse_xgb < rmse_rf:
                self.best_model_name = "xgb"

        self.metrics_["best_model"] = self.best_model_name
        self.metrics_["n_train"] = len(df_train)
        self.metrics_["n_test"] = len(df_test)
        self.metrics_["vb_range_train"] = [float(y_train.min()), float(y_train.max())]
        self.metrics_["vb_range_test"] = [float(y_test.min()), float(y_test.max())]

        if y_train.max() < y_test.max():
            logger.warning(
                f"Train no cubre el rango completo del test "
                f"({y_train.max():.4f} < {y_test.max():.4f} mm) — el modelo extrapola."
            )
        return self.metrics_

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None or self.rf_model is None:
            raise RuntimeError("El predictor no está entrenado. Llama a fit() primero.")

        X = df[self.feature_cols].fillna(0).values
        X_s = self.scaler.transform(X)

        pred_rf = self.rf_model.predict(X_s)
        if self.xgb_model is not None:
            pred_xgb = self.xgb_model.predict(X_s)
            pred = (pred_rf + pred_xgb) / 2.0
            disagreement = np.abs(pred_rf - pred_xgb)
            modo = "ENSEMBLE_RF_XGB"
        else:
            pred = pred_rf
            disagreement = np.zeros_like(pred_rf)
            modo = "RF_ONLY"

        # Incertidumbre: dispersión entre árboles del RF (mismo principio que en PLC)
        tree_preds = np.stack([t.predict(X_s) for t in self.rf_model.estimators_])
        incertidumbre_rf = tree_preds.std(axis=0)
        incertidumbre = incertidumbre_rf + disagreement * 0.5

        out = pd.DataFrame({
            "COMPONENT": df["experiment"] if "experiment" in df.columns else "herramienta_unica",
            "CUT": df["cut"] if "cut" in df.columns else np.arange(len(df)),
            "VB_ESTIMADO": np.round(pred, 5),
            "INCERTIDUMBRE": np.round(incertidumbre, 5),
            "MODO": modo,
        })
        if "VB" in df.columns:
            out["VB_REAL"] = df["VB"].values

        out["ESTADO"] = out["VB_ESTIMADO"].apply(self._estado)
        out["NIVEL_CONFIANZA"] = incertidumbre.round(4)
        out["NIVEL_CONFIANZA"] = out["NIVEL_CONFIANZA"].apply(
            lambda i: "BAJO" if i > 0.05 else ("MEDIO" if i > 0.02 else "ALTO")
        )

        # RUL por componente, ordenado por pasada
        rul_all = []
        for comp, g in out.groupby("COMPONENT", sort=False):
            g_sorted = g.sort_values("CUT")
            rul_vals = compute_rul(g_sorted["VB_ESTIMADO"].values, threshold=self.vb_threshold)
            rul_all.append(pd.Series(rul_vals, index=g_sorted.index))
        out["RUL_ESTIMADO"] = pd.concat(rul_all).reindex(out.index)
        out["RUL_ESTIMADO"] = out["RUL_ESTIMADO"].apply(
            lambda v: "N/D (historial insuficiente)" if v is None or pd.isna(v) else int(v)
        )

        return out.sort_values("VB_ESTIMADO", ascending=False).reset_index(drop=True)

    def _estado(self, vb: float) -> str:
        if vb >= self.vb_threshold:
            return "🔴 FALLA"
        elif vb >= self.vb_threshold * 0.8:
            return "🟡 ALERTA"
        return "🟢 OK"

    # ── Interfaz compartida ──────────────────────────────────────────────────

    @classmethod
    def from_kaggle_artifacts(cls, artifacts_dir: str) -> "ToolWearPredictor":
        """
        Reconstruye un ToolWearPredictor YA ENTRENADO a partir de los
        artefactos exportados desde el notebook de Kaggle:
          - model_rf.joblib      (RandomForestRegressor)
          - model_xgb.joblib     (XGBRegressor, opcional)
          - scaler.joblib        (RobustScaler)
          - metadata.json        (feature_cols, vb_threshold, métricas, etc.)

        Esto NO reentrena nada — carga los pesos tal como salieron de Kaggle,
        listos para llamar predict_all() directamente.
        """
        import json
        import joblib as _joblib

        artifacts_dir = str(artifacts_dir)
        meta_path = os.path.join(artifacts_dir, "metadata.json")
        rf_path = os.path.join(artifacts_dir, "model_rf.joblib")
        xgb_path = os.path.join(artifacts_dir, "model_xgb.joblib")
        scaler_path = os.path.join(artifacts_dir, "scaler.joblib")

        if not os.path.exists(rf_path):
            raise FileNotFoundError(
                f"No se encontró model_rf.joblib en {artifacts_dir}. "
                f"Verifica que descargaste todos los archivos generados por Kaggle."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"No se encontró scaler.joblib en {artifacts_dir}.")

        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            logger.warning(
                "metadata.json no encontrado — usando valores por defecto del módulo "
                "(feature_cols y vb_threshold estándar). Si tu notebook de Kaggle usó "
                "columnas distintas, las predicciones fallarán."
            )

        predictor = cls(
            vb_threshold=metadata.get("vb_threshold", VB_THRESHOLD),
            random_state=metadata.get("random_state", 42),
        )
        predictor.feature_cols = metadata.get("feature_cols", FEAT_COLS)
        predictor.scaler = _joblib.load(scaler_path)
        predictor.rf_model = _joblib.load(rf_path)

        if os.path.exists(xgb_path):
            if not _HAS_XGB:
                logger.warning(
                    "Se encontró model_xgb.joblib pero xgboost no está instalado "
                    "localmente — se ignorará y solo se usará Random Forest. "
                    "Instala xgboost para reproducir exactamente el resultado de Kaggle."
                )
            else:
                predictor.xgb_model = _joblib.load(xgb_path)

        predictor.metrics_ = metadata.get("metrics", {})
        predictor.best_model_name = metadata.get("best_model", "xgb" if predictor.xgb_model else "rf")

        logger.info(
            f"ToolWearPredictor cargado desde artefactos de Kaggle "
            f"({'RF+XGB' if predictor.xgb_model else 'solo RF'}, "
            f"{len(predictor.feature_cols)} features)."
        )
        return predictor

    def report_schema(self) -> ReportSchema:
        return ReportSchema(
            domain_label="Predicción de desgaste de herramienta — señales de sensor",
            id_col="COMPONENT", id_label="Herramienta / experimento",
            group_col="CUT", group_label="N.º de pasada",
            primary_metric_col="VB_ESTIMADO", primary_metric_label="Desgaste estimado VB (mm)",
            status_col="ESTADO",
            status_levels=[("🔴 FALLA", "#dc2626"), ("🟡 ALERTA", "#d97706"), ("🟢 OK", "#059669")],
            confidence_col="NIVEL_CONFIANZA",
            extra_cols=[("RUL_ESTIMADO", "Vida útil restante (pasadas)"),
                       ("MODO", "Modelo usado"), ("INCERTIDUMBRE", "Incertidumbre")],
            higher_is_worse=True,
        )

    def feature_importance(self) -> dict:
        if self.rf_model is None:
            return {}
        return dict(
            sorted(zip(self.feature_cols, self.rf_model.feature_importances_),
                  key=lambda x: x[1], reverse=True)
        )
