"""
core/predictor.py — AdaptiveFailurePredictor.

Motor de ML adaptable: soporta modelo global, modelos locales por
componente, y fallback heurístico. Construido sobre el código original
del notebook con las mejoras discutidas.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import RandomizedSearchCV

from core.config import AgentConfig, ModelStrategy
from core.base_predictor import BasePredictor, ReportSchema

logger = logging.getLogger(__name__)


# ─── Mapeo de categorías (del notebook original, completo) ───────────────────

MAPEO_FALLAS: dict[str, str] = {
    'PLC Connect Error(-1) P2:L5:34-P1:L1:11': 'COMUNICACIÓN',
    'PLC Connect Error(-3) P2:L5:34-P1:L1:11': 'COMUNICACIÓN',
    'COMMUNICATION ERROR(MAIN)': 'COMUNICACIÓN',
    'BCR Read Fault': 'COMUNICACIÓN',
    'S/N Communication Fault': 'COMUNICACIÓN',
    'P1-L3 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'P1-L1 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'P1-L2 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'P1-L5 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'P1-L7 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'P3-L3 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'BCR I/F Unit Fault': 'COMUNICACIÓN',
    'BCR Reading Rate Low Alarm': 'COMUNICACIÓN',
    'BCR Reading Cycle Time Over': 'COMUNICACIÓN',
    'FL-remote communication fault batch': 'COMUNICACIÓN',
    'PCS TOTAL COMMUNICATION FAULT': 'COMUNICACIÓN',
    'Human Detect LC (Entrance)': 'SEGURIDAD',
    'Human Detect LC (Exit)': 'SEGURIDAD',
    'Safety Door Switch (Entrance)': 'SEGURIDAD',
    'SafetyHandle ENT-R(Detail)': 'SEGURIDAD',
    'SafetyHandle ENT-L(Detail)': 'SEGURIDAD',
    'OP2 safety door SW batch': 'SEGURIDAD',
    'OP1 safety door SW batch': 'SEGURIDAD',
    'OP3 safety door SW batch': 'SEGURIDAD',
    'OP1 emergency stop batch': 'SEGURIDAD',
    'OP2 emergency stop batch': 'SEGURIDAD',
    'OP3 emergency stop batch': 'SEGURIDAD',
    'OP1 light curtain batch': 'SEGURIDAD',
    'OP2 light curtain batch': 'SEGURIDAD',
    'OP3 light curtain batch': 'SEGURIDAD',
    'Emergency Stop (Entrance)': 'SEGURIDAD',
    'Emergency Stop': 'SEGURIDAD',
    '1ST Teach Pendant Emergency Stop': 'SEGURIDAD',
    'RB5-8 E-Stop (Detail)': 'SEGURIDAD',
    'RB1-4 E-Stop (Detail)': 'SEGURIDAD',
    'E-Stop Entrance Right (Detail)': 'SEGURIDAD',
    'E-Stop Entrance Left (Detail)': 'SEGURIDAD',
    'Pinching LS': 'SEGURIDAD',
    'Pinching LS Left (Detail)': 'SEGURIDAD',
    'Pinching LS Right (Detail)': 'SEGURIDAD',
    'LC E-Stop': 'SEGURIDAD',
    'FIRE SIGNAL': 'SEGURIDAD',
    'FIRE': 'SEGURIDAD',
    'EARTHQUAKE': 'SEGURIDAD',
    'Safty PLC(PROCESS)': 'SEGURIDAD',
    'SAFETY PLC FAULT PL': 'SEGURIDAD',
    'Safety PLC synthesis fault': 'SEGURIDAD',
    'Safety PLC synthesis communication fault': 'SEGURIDAD',
    'Safety PLC synthesis lead back fault': 'SEGURIDAD',
    'BOOTH MASTER PANEL - EMERGENCY STOP': 'SEGURIDAD',
    'CLEAR ASH C/P - EMERGENCY STOP': 'SEGURIDAD',
    'BASE ASH C/P - EMERGENCY STOP': 'SEGURIDAD',
    'PRIMER ASH C/P - EMERGENCY STOP': 'SEGURIDAD',
    'PRIMER EXHAUST & PREHEAT C/P - EMERGENCY STOP': 'SEGURIDAD',
    'BASE EXHAUST & PREHEAT C/P - EMERGENCY STOP': 'SEGURIDAD',
    'CLEAR EXHAUST C/P - EMERGENCY STOP': 'SEGURIDAD',
    'R/B-3 FAULT': 'HARDWARE_ROBOT',
    'R/B-1 FAULT': 'HARDWARE_ROBOT',
    'R/B-2 FAULT': 'HARDWARE_ROBOT',
    'R/B-4 FAULT': 'HARDWARE_ROBOT',
    'R/B-5 FAULT': 'HARDWARE_ROBOT',
    'R/B-6 FAULT': 'HARDWARE_ROBOT',
    'C1 Heavy Error': 'HARDWARE_ROBOT',
    'C2 Heavy Error': 'HARDWARE_ROBOT',
    'C3 Heavy Error': 'HARDWARE_ROBOT',
    'C4 Heavy Error': 'HARDWARE_ROBOT',
    'C5 Heavy Error': 'HARDWARE_ROBOT',
    'C6 Heavy Error': 'HARDWARE_ROBOT',
    'Base Robot PROTECTIVE STOP': 'HARDWARE_ROBOT',
    'Primer Robot PROTECTIVE STOP': 'HARDWARE_ROBOT',
    'Clear Robot PROTECTIVE STOP': 'HARDWARE_ROBOT',
    'AntiChip Robot PROTECTIVE STOP': 'HARDWARE_ROBOT',
    'R/B-1 ALARM': 'HARDWARE_ROBOT',
    'R/B-2 ALARM': 'HARDWARE_ROBOT',
    'R/B-3 ALARM': 'HARDWARE_ROBOT',
    'R/B-4 ALARM': 'HARDWARE_ROBOT',
    'R/B-5 ALARM': 'HARDWARE_ROBOT',
    'R/B-6 ALARM': 'HARDWARE_ROBOT',
}

TODAS_CATEGORIAS = [
    'COMUNICACIÓN', 'SEGURIDAD', 'HARDWARE_ROBOT', 'CONTROL_LOGICA',
    'PROCESO_PINTURA', 'SISTEMA_ELECTRICO', 'SUMINISTROS', 'OTRO'
]


# ─── Encoder robusto ─────────────────────────────────────────────────────────

class SafeEncoder:
    """OrdinalEncoder que asigna -1 a categorías no vistas en entrenamiento."""

    def __init__(self, columna: str, categorias_conocidas: list):
        self.columna = columna
        self.categorias_conocidas = list(categorias_conocidas)
        self.enc = OrdinalEncoder(
            categories=[self.categorias_conocidas],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=float,
        )
        self.enc.fit(np.array(self.categorias_conocidas).reshape(-1, 1))

    def transform(self, series: pd.Series) -> np.ndarray:
        # np.asarray() en vez de .values: pandas >= 3.0 puede respaldar
        # columnas de texto con ArrowStringArray, que no soporta .reshape()
        # directamente. np.asarray() siempre produce un ndarray de numpy
        # normal, compatible con reshape en cualquier versión.
        valores = np.asarray(series, dtype=object).reshape(-1, 1)
        return self.enc.transform(valores).ravel()

    def es_nuevo(self, valor: str) -> bool:
        return valor not in self.categorias_conocidas


# ─── Predictor adaptable ─────────────────────────────────────────────────────

class AdaptiveFailurePredictor(BasePredictor):
    """
    Predictor de fallas con arquitectura de 3 niveles:
      NIVEL 1 — Modelo global RF
      NIVEL 2 — Modelos locales por componente (si hay datos suficientes)
      NIVEL 3 — Heurístico (prior por categoría/panel)
    """

    def __init__(self, cfg: AgentConfig, strategy: ModelStrategy = ModelStrategy.HYBRID,
                 has_text_logs: bool = True):
        self.cfg = cfg
        self.strategy = strategy
        self.has_text_logs = has_text_logs

        self.modelo_global: Optional[RandomForestClassifier] = None
        self.modelos_locales: dict[str, RandomForestClassifier] = {}
        self.encoder_categoria: Optional[SafeEncoder] = None
        self.encoder_panel: Optional[SafeEncoder] = None
        self.pca: Optional[PCA] = None
        self._embedding_model = None
        self.feature_cols: list[str] = []
        self.prior_categoria: dict = {}
        self.prior_panel: dict = {}
        self.component_n_eventos_map: dict = {}

    # ── Preprocesamiento ──────────────────────────────────────────────────────

    def _preprocess_text(self, text: str) -> str:
        text = str(text).upper()
        subs = {
            r"PLC CONNECT ERROR\(-\d+\)": "PLC communication failure",
            r"HUMAN DETECT LC": "Safety violation human detected",
            r"R/B-(\d+)": r"Painting robot unit \1",
            r"C(\d+) HEAVY ERROR": r"Critical controller error unit",
            r"FAULT": "Operational fault",
            r"SAFETYHANDLE": "Safety handle sensor",
        }
        for pat, rep in subs.items():
            text = re.sub(pat, rep, text)
        text = re.sub(r"P\d:L\d:\d+[-]*[P\d:L\d:\d+]*", "(network_location)", text)
        text = re.sub(r"[()-]", " ", text)
        return " ".join(text.split()).capitalize()

    def _calcular_target(self, df: pd.DataFrame) -> pd.DataFrame:
        ventana = pd.Timedelta(days=self.cfg.ventana_dias)

        def _group_target(grupo):
            targets = []
            for i in range(len(grupo)):
                t = grupo["START TIME"].iloc[i]
                mask = (grupo["START TIME"] > t) & (grupo["START TIME"] <= t + ventana)
                targets.append(int(grupo[mask]["ES_CRITICA"].sum() > 0))
            grupo = grupo.copy()
            grupo["TARGET"] = targets
            return grupo

        df_sorted = df.sort_values(["ADDRESS", "START TIME"]).reset_index(drop=True)
        address_backup = df_sorted["ADDRESS"].copy()
        df_result = df_sorted.groupby("ADDRESS", group_keys=False).apply(_group_target)
        if "ADDRESS" not in df_result.columns:
            # df_result.index es una permutación del índice 0..n-1 de df_sorted
            # (mismo orden que produjo el groupby) — reindex() alinea por
            # etiqueta, no por posición, así que esto restaura el valor
            # correcto de ADDRESS para cada fila.
            df_result["ADDRESS"] = address_backup.reindex(df_result.index)
        return df_result.reset_index(drop=True)

    def _preparar(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["DURACION_MIN"] = pd.to_numeric(df["DURACION_MIN"], errors="coerce").fillna(0)
        df = df[df["DURACION_MIN"] > 0].copy()
        df = df.sort_values("START TIME").reset_index(drop=True)

        df["FAIL COMMENT_PROC"] = df["FAIL COMMENT"].apply(self._preprocess_text)
        df["CATEGORIA"]  = df["FAIL COMMENT"].map(MAPEO_FALLAS).fillna("OTRO")
        df["ES_CRITICA"]  = (df["DURACION_MIN"] >= self.cfg.umbral_critico).astype(int)
        df["hora_del_dia"] = df["START TIME"].dt.hour
        df["dia_semana"]   = df["START TIME"].dt.dayofweek
        df["es_fin_semana"]= (df["dia_semana"] >= 5).astype(int)
        df["mes"]          = df["START TIME"].dt.month
        df["turno_cod"]    = df["hora_del_dia"].apply(
            lambda h: 0 if 6 <= h < 14 else (1 if 14 <= h < 22 else 2)
        )
        return self._calcular_target(df)

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.cfg.embedding_model)
        return self._embedding_model

    def _generar_embeddings(self, df: pd.DataFrame, fit_pca: bool) -> pd.DataFrame:
        if not self.has_text_logs:
            # Llenar con ceros si no hay textos
            cols = [f"emb_{i}" for i in range(self.cfg.n_pca_components)]
            for c in cols:
                df[c] = 0.0
            return df

        model = self._get_embedding_model()
        textos_unicos = df["FAIL COMMENT_PROC"].unique()
        vectores = model.encode(textos_unicos, show_progress_bar=False)
        mapa = dict(zip(textos_unicos, vectores))
        embeddings = np.stack(df["FAIL COMMENT_PROC"].map(mapa).values)

        if fit_pca:
            n = min(self.cfg.n_pca_components, embeddings.shape[0] - 1)
            self.pca = PCA(n_components=n, random_state=self.cfg.random_state)
            reduced = self.pca.fit_transform(embeddings)
        else:
            if self.pca is None:
                reduced = embeddings[:, : self.cfg.n_pca_components]
            else:
                reduced = self.pca.transform(embeddings)

        n_comp = reduced.shape[1]
        cols = [f"emb_{i}" for i in range(n_comp)]
        # Rellenar hasta n_pca_components si PCA tiene menos componentes
        for i in range(n_comp, self.cfg.n_pca_components):
            cols_extra = f"emb_{i}"
            df[cols_extra] = 0.0

        return pd.concat(
            [df, pd.DataFrame(reduced, columns=cols, index=df.index)], axis=1
        )

    # ── Features ──────────────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> tuple:
        df = df.sort_values(["ADDRESS", "START TIME"]).reset_index(drop=True)
        # NOTA DE COMPATIBILIDAD: en pandas >= 3.0, groupby(...).apply() ya
        # no conserva la columna de agrupación en el resultado. Se guarda
        # una copia y se restaura después si hace falta — no afecta ningún
        # valor calculado, es solo compatibilidad entre versiones.
        address_backup = df["ADDRESS"].copy()

        def calc(group):
            addr = group.name  # en pandas >= 3.0, ADDRESS ya no está en las
                               # columnas del grupo — se usa group.name,
                               # que pandas sí conserva siempre.
            n = self.component_n_eventos_map.get(addr, self.cfg.default_n_eventos)

            group["tiempo_desde_ultima_falla"] = (
                group["START TIME"].diff().dt.total_seconds() / 3600
            )
            group["media_duracion_3_previas"] = (
                group["DURACION_MIN"].shift(1).rolling(3, min_periods=1).mean()
            )
            group["max_duracion_5_previas"] = (
                group["DURACION_MIN"].shift(1).rolling(5, min_periods=1).max()
            )
            group["min_duracion_5_previas"] = (
                group["DURACION_MIN"].shift(1).rolling(5, min_periods=1).min()
            )
            group["tendencia_duracion"] = (
                group["DURACION_MIN"].shift(1).rolling(3, min_periods=2).apply(
                    lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) >= 2 else 0,
                    raw=True,
                )
            )
            group["conteo_criticas_previas"] = (
                group["ES_CRITICA"].shift(1).rolling(n, min_periods=1).sum()
            )
            group["proporcion_criticas_previas"] = (
                group["ES_CRITICA"].shift(1).rolling(n, min_periods=1).mean()
            )
            group["ultima_fue_critica"] = group["ES_CRITICA"].shift(1).fillna(0).astype(int)
            group["misma_categoria_anterior"] = (
                (group["CATEGORIA"] == group["CATEGORIA"].shift(1)).astype(int)
            )
            # Features relativos (nuevos — generalizan a componentes nuevos)
            rec = group["ES_CRITICA"].shift(1).rolling(7, min_periods=1).mean()
            hist = group["ES_CRITICA"].shift(1).rolling(30, min_periods=1).mean()
            group["ratio_reciente_historico"] = (rec / (hist + 1e-6)).clip(0, 10)
            group["mediana_tiempo_entre_fallas"] = (
                group["tiempo_desde_ultima_falla"].rolling(10, min_periods=1).median()
            )
            group["aceleracion_fallas"] = (
                group["tiempo_desde_ultima_falla"].diff().fillna(0)
            )
            return group

        df = df.groupby("ADDRESS", group_keys=False).apply(calc)
        if "ADDRESS" not in df.columns:
            df["ADDRESS"] = address_backup.reindex(df.index)

        # Fallas en ventana de 7 días
        df["temp_uid"] = df["START TIME"] + df.groupby(
            ["ADDRESS", "START TIME"]
        ).cumcount().astype("timedelta64[ns]")
        df["fallas_componente_7d"] = (
            df.groupby("ADDRESS")["DURACION_MIN"]
              .transform(
                  lambda x: x.shift(1)
                              .set_axis(df.loc[x.index, "temp_uid"])
                              .rolling("7D")
                              .count()
                              .reindex(x.index)
                              .values
              )
        )
        df = df.drop(columns=["temp_uid"])
        df["n_falla_componente"] = df.groupby("ADDRESS").cumcount()

        # Encoders seguros
        df["categoria_cod"] = self.encoder_categoria.transform(df["CATEGORIA"])
        df["panel_cod"]     = self.encoder_panel.transform(df["PANEL"])

        emb_cols = [f"emb_{i}" for i in range(self.cfg.n_pca_components)]
        self.feature_cols = [
            "hora_del_dia", "dia_semana", "es_fin_semana", "mes", "turno_cod",
            "tiempo_desde_ultima_falla",
            "media_duracion_3_previas", "max_duracion_5_previas",
            "min_duracion_5_previas", "tendencia_duracion",
            "conteo_criticas_previas", "proporcion_criticas_previas",
            "ultima_fue_critica", "misma_categoria_anterior",
            "fallas_componente_7d", "n_falla_componente",
            "categoria_cod", "panel_cod",
            "ratio_reciente_historico", "mediana_tiempo_entre_fallas",
            "aceleracion_fallas",
            *emb_cols,
        ]

        df_model = df[self.feature_cols + ["TARGET", "START TIME", "ADDRESS", "PANEL"]].copy()
        df_model = df_model.dropna(subset=["TARGET"])
        df_model[self.feature_cols] = df_model[self.feature_cols].fillna(0)
        df_model = df_model.sort_values("START TIME").reset_index(drop=True)

        X    = df_model[self.feature_cols]
        y    = df_model["TARGET"].astype(int)
        meta = df_model[["START TIME", "ADDRESS", "PANEL"]]
        return X, y, meta, df_model

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(self, df_raw: pd.DataFrame,
            tune_hyperparams: bool = False) -> dict:
        t0 = time.time()
        logger.info("Iniciando entrenamiento...")

        logger.info("  [1/5] Preprocesando datos...")
        df = self._preparar(df_raw)

        logger.info("  [2/5] Inicializando encoders...")
        categorias = sorted(df["CATEGORIA"].unique().tolist())
        for c in TODAS_CATEGORIAS:
            if c not in categorias:
                categorias.append(c)
        paneles = sorted(df["PANEL"].unique().tolist())
        self.encoder_categoria = SafeEncoder("CATEGORIA", categorias)
        self.encoder_panel     = SafeEncoder("PANEL",     paneles)

        logger.info("  [3/5] Generando embeddings + PCA (solo en train)...")
        split_idx = int(len(df) * self.cfg.split_ratio)
        df_train = df.iloc[:split_idx].copy()
        df_test  = df.iloc[split_idx:].copy()

        df_train = self._generar_embeddings(df_train, fit_pca=True)
        df_test  = self._generar_embeddings(df_test,  fit_pca=False)

        logger.info("  [4/5] Construyendo features...")
        X_train, y_train, _, df_train_model = self._build_features(df_train)
        X_test,  y_test,  _, _              = self._build_features(df_test)

        logger.info("  [5/5] Entrenando modelo global...")
        if tune_hyperparams and len(X_train) >= 200:
            params = self._tune(X_train, y_train)
        else:
            params = {
                "n_estimators": 300, "max_depth": 20,
                "min_samples_leaf": 3, "class_weight": "balanced",
                "random_state": self.cfg.random_state, "n_jobs": -1,
            }

        self.modelo_global = RandomForestClassifier(**params)
        self.modelo_global.fit(X_train, y_train)

        if self.strategy in (ModelStrategy.LOCAL_RF, ModelStrategy.HYBRID):
            logger.info("       → Entrenando modelos locales...")
            self._entrenar_locales(df_train_model)

        self._calcular_priors(df_train)

        metricas = self._evaluar(self.modelo_global, X_test, y_test)
        metricas["tiempo_entrenamiento_sec"] = round(time.time() - t0, 1)
        logger.info(f"Entrenamiento completado en {metricas['tiempo_entrenamiento_sec']}s")
        return metricas

    def _tune(self, X_train, y_train) -> dict:
        logger.info("       → Ajuste de hiperparámetros (RandomizedSearchCV)...")
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }
        rs = RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced",
                                   random_state=self.cfg.random_state, n_jobs=-1),
            param_dist, n_iter=10, cv=3,
            scoring="roc_auc", random_state=self.cfg.random_state,
            verbose=0, n_jobs=-1,
        )
        rs.fit(X_train, y_train)
        logger.info(f"Mejores params: {rs.best_params_}")
        best = rs.best_params_
        best["class_weight"] = "balanced"
        best["random_state"]  = self.cfg.random_state
        best["n_jobs"]        = -1
        return best

    def _entrenar_locales(self, df_train: pd.DataFrame):
        n_local = 0
        for address, group in df_train.groupby("ADDRESS"):
            if len(group) < self.cfg.min_eventos_modelo_local:
                continue
            X_loc = group[self.feature_cols].fillna(0)
            y_loc = group["TARGET"].astype(int)
            if y_loc.nunique() < 2:
                continue
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.cfg.random_state, n_jobs=-1,
            )
            rf.fit(X_loc, y_loc)
            self.modelos_locales[address] = rf
            n_local += 1
        logger.info(f"       → {n_local} modelos locales entrenados.")

    def _calcular_priors(self, df_train: pd.DataFrame):
        if "TARGET" in df_train.columns:
            self.prior_categoria = df_train.groupby("CATEGORIA")["TARGET"].mean().to_dict()
            self.prior_panel     = df_train.groupby("PANEL")["TARGET"].mean().to_dict()

    def _evaluar(self, modelo, X_test, y_test) -> dict:
        y_prob = modelo.predict_proba(X_test)[:, 1]
        metricas = {
            "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
            "auc_pr":  round(average_precision_score(y_test, y_prob), 4),
            "n_test":  len(y_test),
            "pos_rate_test": round(y_test.mean(), 4),
        }
        logger.info(f"  Métricas en test → AUC-ROC: {metricas['auc_roc']} | AUC-PR: {metricas['auc_pr']}")
        return metricas

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict_all(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Genera ranking de riesgo para todos los componentes."""
        df = self._preparar(df_raw)
        df = self._generar_embeddings(df, fit_pca=False)
        X_all, _, meta, df_model = self._build_features(df)

        resultados = []
        for address in meta["ADDRESS"].unique():
            mask   = meta["ADDRESS"] == address
            X_comp = X_all[mask]
            if len(X_comp) == 0:
                continue
            X_ult  = X_comp.iloc[[-1]].fillna(0)

            n_eventos = int(mask.sum())
            if n_eventos < 3:
                resultado = self._prediccion_heuristica(df[df["ADDRESS"] == address])
                resultado["ADDRESS"] = address
                resultado["N_EVENTOS"] = n_eventos
                resultados.append(resultado)
                continue

            modelo, modo = self._seleccionar_modelo(address)
            prob   = float(modelo.predict_proba(X_ult)[0, 1])
            inc    = self._incertidumbre(modelo, X_ult)
            panel  = meta[mask]["PANEL"].iloc[-1]

            resultados.append({
                "ADDRESS":         address,
                "PANEL":           panel,
                "PROB_FALLA":      round(prob, 4),
                "INCERTIDUMBRE":   round(inc, 4),
                "NIVEL_RIESGO":    self._nivel_riesgo(prob),
                "NIVEL_CONFIANZA": self._nivel_confianza(inc, n_eventos),
                "MODO":            modo,
                "N_EVENTOS":       n_eventos,
            })

        return (
            pd.DataFrame(resultados)
              .sort_values("PROB_FALLA", ascending=False)
              .reset_index(drop=True)
        )

    def _prediccion_heuristica(self, df_comp: pd.DataFrame) -> dict:
        categoria = df_comp["CATEGORIA"].iloc[-1] if len(df_comp) > 0 else "OTRO"
        panel     = df_comp["PANEL"].iloc[-1]     if len(df_comp) > 0 else "PANEL_DESCONOCIDO"
        prob = (
            self.prior_categoria.get(categoria, 0.3) * 0.5 +
            self.prior_panel.get(panel, 0.3) * 0.5
        )
        return {
            "PANEL": panel,
            "PROB_FALLA":      round(float(prob), 4),
            "INCERTIDUMBRE":   0.5,
            "NIVEL_RIESGO":    self._nivel_riesgo(prob),
            "NIVEL_CONFIANZA": "BAJO",
            "MODO":            "HEURÍSTICO",
        }

    def _seleccionar_modelo(self, address: str) -> tuple:
        if address in self.modelos_locales:
            return self.modelos_locales[address], "LOCAL"
        return self.modelo_global, "GLOBAL"

    def _incertidumbre(self, modelo: RandomForestClassifier, X) -> float:
        preds = np.array([t.predict_proba(X)[0, 1] for t in modelo.estimators_])
        return float(np.std(preds))

    def _nivel_riesgo(self, prob: float) -> str:
        if prob >= 0.6:   return "🔴 ALTO"
        elif prob >= 0.3: return "🟡 MEDIO"
        return "🟢 BAJO"

    def _nivel_confianza(self, inc: float, n: int) -> str:
        if n < 10 or inc > self.cfg.umbral_incertidumbre: return "BAJO"
        if inc > 0.15: return "MEDIO"
        return "ALTO"

    def report_schema(self) -> ReportSchema:
        return ReportSchema(
            domain_label="Predicción de fallas — eventos PLC",
            id_col="ADDRESS", id_label="Componente",
            group_col="PANEL", group_label="Panel / línea",
            primary_metric_col="PROB_FALLA", primary_metric_label="Probabilidad de falla",
            status_col="NIVEL_RIESGO",
            status_levels=[("🔴 ALTO", "#dc2626"), ("🟡 MEDIO", "#d97706"), ("🟢 BAJO", "#059669")],
            confidence_col="NIVEL_CONFIANZA",
            extra_cols=[("MODO", "Modelo usado"), ("N_EVENTOS", "N eventos"), ("INCERTIDUMBRE", "Incertidumbre")],
            higher_is_worse=True,
        )

    def feature_importance(self) -> dict:
        if self.modelo_global is None:
            return {}
        return dict(
            sorted(
                zip(self.feature_cols, self.modelo_global.feature_importances_),
                key=lambda x: x[1], reverse=True,
            )
        )

    def update(self, df_nuevos: pd.DataFrame):
        """Actualiza modelos locales sin reentrenar el modelo global."""
        df = self._preparar(df_nuevos)
        df = self._generar_embeddings(df, fit_pca=False)
        X_new, y_new, meta, _ = self._build_features(df)
        actualizados = 0
        for address, group_meta in meta.groupby("ADDRESS"):
            idx = group_meta.index
            if len(idx) < self.cfg.min_eventos_modelo_local:
                continue
            X_loc = X_new.loc[idx].fillna(0)
            y_loc = y_new.loc[idx]
            if y_loc.nunique() < 2:
                continue
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.cfg.random_state, n_jobs=-1,
            )
            rf.fit(X_loc, y_loc)
            self.modelos_locales[address] = rf
            actualizados += 1
        logger.info(f"Update completado: {actualizados} modelos locales actualizados.")
        return actualizados

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Predictor guardado en: {path}")

    @classmethod
    def load(cls, path: str) -> "AdaptiveFailurePredictor":
        return joblib.load(path)
