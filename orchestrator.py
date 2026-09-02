"""
orchestrator.py — Orquestador del sistema de agentes.

Punto de entrada único: coordina los 4 agentes y expone
una API simple para uso en producción.

Uso mínimo:
    from orchestrator import PLCFailureAgentSystem

    system = PLCFailureAgentSystem()
    report_path = system.run("datos_cliente.xlsx")

Uso avanzado:
    system = PLCFailureAgentSystem(config=AgentConfig(umbral_critico=20))
    report_path = system.run(
        source="datos.csv",
        client_id="empresa_xyz",
        column_map={"col_component": "TAG_ID", "col_panel": "LINEA"}
    )
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union, Optional

import pandas as pd

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import AgentConfig, InferredSchema, StrategyPlan, ExecutionResult
from agents.agent1_analyzer  import DataAnalyzerAgent
from agents.agent2_planner   import StrategyPlannerAgent
from agents.agent3_executor  import PipelineExecutorAgent
from agents.agent4_reporter  import ReportWriterAgent
from agents.domain_router    import detect_domain, Domain
from agents.agent1_tool_wear_loader import ToolWearDataLoaderAgent
from agents.agent2_tool_wear_planner import ToolWearStrategyPlanner
from agents.agent3_tool_wear_executor import ToolWearExecutorAgent
from core.tool_wear_predictor import ToolWearPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════╗
║   Sistema de Predicción de Fallas PLC                ║
║   Agente adaptable multi-fuente                      ║
╚══════════════════════════════════════════════════════╝"""


class PLCFailureAgentSystem:
    """
    Orquestador principal del sistema de 4 agentes.

    Flujo:
        Agente 1 → Analiza la fuente y detecta schema
        Agente 2 → Planifica la estrategia de modelo
        Agente 3 → Ejecuta pipeline (preprocesa, entrena, predice)
        Agente 4 → Genera reporte HTML

    El sistema es stateful: guarda el predictor en disco para
    que llamadas futuras (update) puedan cargarlo sin reentrenar.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.cfg = config or AgentConfig()
        self.agent1 = DataAnalyzerAgent(self.cfg)
        self.agent2 = StrategyPlannerAgent(self.cfg)
        self.agent3 = PipelineExecutorAgent(self.cfg)
        self.agent4 = ReportWriterAgent(self.cfg)

        # Estado interno (útil para inspección o tests)
        self.last_schema:  Optional[InferredSchema]  = None
        self.last_plan:    Optional[StrategyPlan]    = None
        self.last_result:  Optional[ExecutionResult] = None
        self.last_report:  Optional[str]             = None

    # ── API principal ─────────────────────────────────────────────────────────

    def run(
        self,
        source: Union[str, Path, pd.DataFrame, tuple],
        client_id: str = "default",
        column_map: Optional[dict] = None,
        verbose: bool = True,
    ) -> str:
        """
        Ejecuta el pipeline completo y devuelve la ruta del reporte.

        Parameters
        ----------
        source     : archivo (xlsx, csv, json, parquet), DataFrame, o
                     (connection_string, sql_query) para bases de datos
        client_id  : identificador del cliente (aísla modelos entre clientes)
        column_map : dict opcional para forzar mapeo de columnas, ej.:
                     {"col_component": "TAG_ID", "col_panel": "LINEA"}
        verbose    : si mostrar reportes de cada agente en consola

        Returns
        -------
        str: ruta del archivo HTML generado
        """
        if verbose:
            logger.info(BANNER)

        # ── AGENTE 1: Analizar datos ──────────────────────────────────────────
        logger.info("\n[1/4] Agente 1 — Analizando datos...")
        df, schema = self.agent1.analyze(source, column_map=column_map)
        self.last_schema = schema
        if verbose:
            self.agent1.print_schema_report(schema)

        if not schema.is_valid:
            logger.error("✗ Error en datos de entrada. Revisa las advertencias arriba.")
            return self._generar_reporte_error(schema, client_id)

        # ── AGENTE 2: Planificar estrategia ───────────────────────────────────
        logger.info("[2/4] Agente 2 — Planificando estrategia...")
        plan = self.agent2.plan(df, schema)
        self.last_plan = plan
        if verbose:
            self.agent2.print_plan_report(plan)

        # ── AGENTE 3: Ejecutar pipeline ───────────────────────────────────────
        logger.info("[3/4] Agente 3 — Ejecutando pipeline de ML...")
        result = self.agent3.execute(
            df, plan,
            has_text_logs=schema.has_text_logs,
            client_id=client_id,
        )
        self.last_result = result
        if verbose:
            self.agent3.print_execution_report(result)
            if result.ranking is not None and len(result.ranking) > 0:
                self._print_ranking_consola(result)

        # ── AGENTE 4: Generar reporte ─────────────────────────────────────────
        logger.info("[4/4] Agente 4 — Generando reporte...")
        report_path = self.agent4.generate(
            schema=schema, plan=plan, result=result, client_id=client_id
        )
        self.last_report = report_path
        logger.info(f"\n✓ Reporte generado: {report_path}")
        return report_path

    def update(
        self,
        source: Union[str, Path, pd.DataFrame, tuple],
        client_id: str = "default",
        column_map: Optional[dict] = None,
    ) -> str:
        """
        Actualización incremental: añade datos nuevos sin reentrenar desde cero.
        Solo actualiza modelos locales de los componentes presentes en los nuevos datos.

        Returns
        -------
        str: ruta del reporte actualizado
        """
        logger.info("\n[Actualización incremental]")

        df, schema = self.agent1.analyze(source, column_map=column_map)
        if not schema.is_valid:
            logger.error("✗ Error en datos de actualización.")
            return ""

        n_actualizados = self.agent3.update(df, client_id=client_id)
        logger.info(f"  → {n_actualizados} modelos locales actualizados.")

        # Regenerar predicciones y reporte
        result = self.agent3.execute(
            df,
            self.last_plan or StrategyPlan(
                data_quality=__import__('core.config', fromlist=['DataQuality']).DataQuality.MODERATE,
                model_strategy=__import__('core.config', fromlist=['ModelStrategy']).ModelStrategy.HYBRID,
                can_train=False,
            ),
            has_text_logs=schema.has_text_logs,
            client_id=client_id,
        )

        return self.agent4.generate(
            schema=self.last_schema or schema,
            plan=self.last_plan or StrategyPlan(
                data_quality=__import__('core.config', fromlist=['DataQuality']).DataQuality.MODERATE,
                model_strategy=__import__('core.config', fromlist=['ModelStrategy']).ModelStrategy.HYBRID,
                can_train=False,
            ),
            result=result,
            client_id=client_id,
        )

    # ── Utilidades ────────────────────────────────────────────────────────────

    def get_ranking(self) -> Optional[pd.DataFrame]:
        """Devuelve el último ranking generado."""
        if self.last_result:
            return self.last_result.ranking
        return None

    def get_metrics(self) -> dict:
        """Devuelve las métricas del último modelo entrenado."""
        if self.last_result:
            return self.last_result.metrics
        return {}

    def _print_ranking_consola(self, result: ExecutionResult):
        ranking = result.ranking
        lineas = [
            f"\n{'─'*60}",
            f"  RANKING DE RIESGO — Top {self.cfg.top_n_ranking}",
            f"{'─'*60}",
            f"  {'COMPONENTE':<20} {'PANEL':<15} {'PROB':>6} {'NIVEL':<12} {'CONF':<8} {'MODO'}",
            f"  {'─'*56}",
        ]
        for row in ranking.head(self.cfg.top_n_ranking).itertuples():
            lineas.append(
                f"  {row.ADDRESS:<20} "
                f"{str(row.PANEL):<15} "
                f"{row.PROB_FALLA:>6.3f} "
                f"{row.NIVEL_RIESGO:<12} "
                f"{row.NIVEL_CONFIANZA:<8} "
                f"{row.MODO}"
            )
        lineas.append(f"{'─'*60}\n")
        logger.info("\n".join(lineas))

    def _generar_reporte_error(
        self, schema: InferredSchema, client_id: str
    ) -> str:
        """Genera un reporte mínimo cuando los datos son inválidos."""
        from datetime import datetime
        Path(self.cfg.report_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(Path(self.cfg.report_dir) / f"error_{client_id}_{ts}.html")
        warns = "\n".join(f"<li>{w}</li>" for w in schema.warnings)
        html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
        <title>Error — {client_id}</title></head>
        <body style="font-family:sans-serif;padding:32px">
        <h1>No se pudieron generar predicciones</h1>
        <p>Los datos no son válidos para el análisis. Revisa las advertencias:</p>
        <ul style="color:#c0392b">{warns}</ul>
        </body></html>"""
        Path(path).write_text(html, encoding="utf-8")
        return path


# ─── CLI simple ───────────────────────────────────────────────────────────────

def main():
    """Punto de entrada por línea de comandos."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sistema de Predicción de Fallas PLC"
    )
    parser.add_argument("source",      help="Ruta al archivo de datos (xlsx, csv, json)")
    parser.add_argument("--client",    default="default", help="ID del cliente")
    parser.add_argument("--umbral",    type=int, default=30, help="Umbral crítico en minutos")
    parser.add_argument("--ventana",   type=int, default=7,  help="Ventana de predicción en días")
    parser.add_argument("--output",    default="output",     help="Directorio de salida")
    parser.add_argument("--col-component", help="Nombre de la columna de componente")
    parser.add_argument("--col-start",     help="Nombre de la columna de timestamp inicio")
    parser.add_argument("--col-panel",     help="Nombre de la columna de panel")
    parser.add_argument("--col-log",       help="Nombre de la columna de texto de log")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Nivel de detalle en consola (también configurable con "
                             "la variable de entorno PLC_AGENT_LOG_LEVEL)")
    parser.add_argument("--log-file", default=None,
                        help="Si se indica, además de consola, guarda el log completo en este archivo")
    args = parser.parse_args()

    from core.logging_config import setup_logging
    setup_logging(level=args.log_level, log_file=args.log_file)

    from pydantic import ValidationError
    try:
        config = AgentConfig(
            umbral_critico = args.umbral,
            ventana_dias   = args.ventana,
            output_dir     = args.output,
            model_dir      = f"{args.output}/models",
            report_dir     = f"{args.output}/reports",
        )
    except ValidationError as e:
        logger.error("Configuración inválida — revisa los parámetros que pasaste:")
        for err in e.errors():
            campo = ".".join(str(p) for p in err["loc"])
            logger.error(f"  • {campo}: {err['msg']} (recibido: {err.get('input')!r})")
        sys.exit(1)

    column_map = {}
    if args.col_component: column_map["col_component"]       = args.col_component
    if args.col_start:     column_map["col_timestamp_start"] = args.col_start
    if args.col_panel:     column_map["col_panel"]           = args.col_panel
    if args.col_log:       column_map["col_log_text"]        = args.col_log

    system = PLCFailureAgentSystem(config=config)
    report = system.run(
        source     = args.source,
        client_id  = args.client,
        column_map = column_map or None,
    )
    logger.info(f"\nReporte disponible en: {report}")


class MultiVerticalAgentSystem:
    """
    Orquestador multi-dominio: detecta automáticamente si los datos
    corresponden a la vertical de fallas PLC (eventos discretos) o a la
    vertical de desgaste de herramienta (señales de sensor, PHM 2010), y
    enruta al pipeline correspondiente.

    Ambas verticales comparten:
      - la interfaz BasePredictor (fit / predict_all / report_schema)
      - el Agente 4 genérico (generate_generic), dirigido por ReportSchema
      - el mismo AgentConfig y la misma convención de client_id

    Uso:
        system = MultiVerticalAgentSystem()
        report = system.run("logs_plc.xlsx", client_id="cliente_a")
        report = system.run(features_phm2010_df, client_id="cliente_b")
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.cfg = config or AgentConfig()
        self.plc_system = PLCFailureAgentSystem(config=self.cfg)
        self.tool_wear_loader = ToolWearDataLoaderAgent()
        self.tool_wear_planner = ToolWearStrategyPlanner(vb_threshold=0.2)
        self.tool_wear_executor = ToolWearExecutorAgent(self.cfg)
        self.agent4 = ReportWriterAgent(self.cfg)
        self.last_domain: Optional[Domain] = None
        self.last_predictor: Optional[ToolWearPredictor] = None
        self.last_result_df = None

    def run(
        self,
        source,
        client_id: str = "default",
        column_map: Optional[dict] = None,
        force_domain: Optional[str] = None,
        verbose: bool = True,
    ) -> str:
        domain = Domain(force_domain) if force_domain else detect_domain(source, column_map)
        self.last_domain = domain

        if verbose:
            logger.info(f"\n[Router] Dominio detectado: {domain.value}")

        if domain == Domain.PLC_FAILURE:
            report = self.plc_system.run(source, client_id=client_id,
                                         column_map=column_map, verbose=verbose)
            self.last_result_df = self.plc_system.get_ranking()
            return report

        if domain == Domain.TOOL_WEAR:
            return self._run_tool_wear(source, client_id, column_map, verbose)

        # UNKNOWN — por defecto se intenta como PLC (vertical original del producto)
        # ya que su detección de schema es más tolerante a datos ambiguos.
        if verbose:
            logger.info("[Router] Dominio ambiguo — intentando como fallas PLC por defecto. "
                       "Usa force_domain='tool_wear' si corresponde a señales de sensor.")
        report = self.plc_system.run(source, client_id=client_id,
                                     column_map=column_map, verbose=verbose)
        self.last_result_df = self.plc_system.get_ranking()
        return report

    def _run_tool_wear(self, source, client_id: str,
                       column_map: Optional[dict], verbose: bool) -> str:
        logger.info("\n[1/4] Agente 1 (desgaste) — cargando y validando señales...")
        df, schema = self.tool_wear_loader.analyze(source, column_map=column_map)
        if verbose:
            self.tool_wear_loader.print_schema_report(schema)

        if not schema.is_valid:
            logger.error("✗ Datos inválidos para la vertical de desgaste de herramienta.")
            return self._generic_error_report(schema, client_id)

        logger.info("[2/4] Agente 2 (desgaste) — planificando estrategia...")
        plan = self.tool_wear_planner.plan(df, schema)
        if verbose:
            self.tool_wear_planner.print_plan_report(plan)

        if not plan.can_train:
            logger.error("✗ El Agente 2 determinó que no se puede entrenar con estos datos.")
            schema.warnings.extend(plan.rationale)
            return self._generic_error_report(schema, client_id)

        logger.info("[3/4] Agente 3 (desgaste) — entrenando y prediciendo...")
        result = self.tool_wear_executor.execute(df, plan, client_id=client_id)
        self.last_predictor = self.tool_wear_executor.predictor
        self.last_result_df = result.result_df

        logger.info("[4/4] Agente 4 — generando reporte...")
        report_path = self.agent4.generate_generic(
            report_schema=self.tool_wear_executor.report_schema(),
            result_df=result.result_df,
            metrics=result.metrics,
            feature_importances=result.feature_importances,
            warnings=schema.warnings + plan.warnings,
            client_id=client_id,
        )
        logger.info(f"\n✓ Reporte generado: {report_path}")
        return report_path

    def _generic_error_report(self, schema: InferredSchema, client_id: str) -> str:
        from datetime import datetime
        Path(self.cfg.report_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(Path(self.cfg.report_dir) / f"error_{client_id}_{ts}.html")
        warns = "\n".join(f"<li>{w}</li>" for w in schema.warnings)
        html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
        <title>Error — {client_id}</title></head>
        <body style="font-family:sans-serif;padding:32px">
        <h1>No se pudieron generar predicciones</h1>
        <ul style="color:#c0392b">{warns}</ul>
        </body></html>"""
        Path(path).write_text(html, encoding="utf-8")
        return path

    # ── Inferencia pura — sin reentrenar (modelos ya entrenados) ───────────────

    def predict_only(
        self,
        source,
        client_id: str = "default",
        domain: Optional[str] = None,
        model_path: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
        column_map: Optional[dict] = None,
        verbose: bool = True,
        legacy: bool = False,
    ) -> str:
        """
        Genera predicciones usando un modelo YA ENTRENADO, sin reentrenar
        nada. Pensado para el flujo: entrenas en Kaggle (o donde sea) →
        descargas los artefactos → aquí solo infieres.

        Parameters
        ----------
        source        : datos nuevos a predecir (mismo formato que run())
        client_id     : identificador del cliente / modelo
        domain        : "plc_failure" o "tool_wear". Si se omite, se
                        detecta automáticamente a partir de source.
        model_path    : ruta a un .pkl de predictor completo, guardado con
                        AdaptiveFailurePredictor.save() o
                        ToolWearPredictor.save() (BasePredictor.save()).
                        Si se omite, se busca en
                        {model_dir}/predictor_{client_id}.pkl (PLC) o
                        {model_dir}/tool_wear_predictor_{client_id}.pkl
                        (desgaste).
        artifacts_dir : SOLO para desgaste de herramienta — carpeta con los
                        archivos sueltos exportados desde Kaggle
                        (model_rf.joblib, model_xgb.joblib, scaler.joblib,
                        metadata.json). Tiene prioridad sobre model_path
                        si ambos se pasan.
        legacy        : SOLO para domain="plc_failure". Si True, usa
                        LegacyPLCPredictor (réplica exacta del repositorio
                        luisroberto-maker/PLC-failure-prediction-pipeline)
                        en vez de AdaptiveFailurePredictor. Úsalo si
                        model_path apunta a un .pkl entrenado con ESE
                        repositorio (un estimador sklearn "pelado", no un
                        objeto AdaptiveFailurePredictor completo).

        Returns
        -------
        str: ruta del reporte HTML generado.
        """
        resolved_domain = Domain(domain) if domain else detect_domain(source, column_map)
        self.last_domain = resolved_domain

        if verbose:
            logger.info(f"[Router — solo inferencia] Dominio: {resolved_domain.value}"
                       f"{' (modelo legado)' if legacy else ''}")

        if resolved_domain == Domain.TOOL_WEAR:
            return self._predict_only_tool_wear(
                source, client_id, model_path, artifacts_dir, column_map, verbose
            )
        elif resolved_domain == Domain.PLC_FAILURE:
            if legacy:
                return self._predict_only_plc_legacy(source, client_id, model_path, column_map, verbose)
            return self._predict_only_plc(source, client_id, model_path, column_map, verbose)
        else:
            raise ValueError(
                "No se pudo determinar el dominio automáticamente. "
                "Especifica domain='plc_failure' o domain='tool_wear' explícitamente."
            )

    def _predict_only_tool_wear(
        self, source, client_id, model_path, artifacts_dir, column_map, verbose
    ) -> str:
        logger.info("[1/3] Agente 1 (desgaste) — cargando y validando datos nuevos...")
        df, schema = self.tool_wear_loader.analyze(source, column_map=column_map)
        if verbose:
            self.tool_wear_loader.print_schema_report(schema)
        if not schema.is_valid:
            logger.error("✗ Datos inválidos.")
            return self._generic_error_report(schema, client_id)

        logger.info("[2/3] Agente 3 (desgaste) — cargando modelo y prediciendo...")
        result = self.tool_wear_executor.load_and_predict(
            df, client_id=client_id, model_path=model_path, artifacts_dir=artifacts_dir,
        )
        self.last_predictor = self.tool_wear_executor.predictor
        self.last_result_df = result.result_df

        logger.info("[3/3] Agente 4 — generando reporte...")
        report_path = self.agent4.generate_generic(
            report_schema=self.tool_wear_executor.report_schema(),
            result_df=result.result_df,
            metrics=result.metrics,
            feature_importances=result.feature_importances,
            warnings=schema.warnings + ["Predicciones generadas con un modelo pre-entrenado, sin reentrenamiento."],
            client_id=client_id,
        )
        logger.info(f"\n✓ Reporte generado: {report_path}")
        return report_path

    def _predict_only_plc(self, source, client_id, model_path, column_map, verbose) -> str:
        from core.predictor import AdaptiveFailurePredictor

        model_path = model_path or str(Path(self.cfg.model_dir) / f"predictor_{client_id}.pkl")
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No se encontró un modelo entrenado en {model_path}. "
                f"Entrena primero con system.run(...)."
            )
        logger.info(f"[1/3] Cargando modelo entrenado desde {model_path}...")
        predictor = AdaptiveFailurePredictor.load(model_path)

        logger.info("[2/3] Agente 1 — cargando y validando datos nuevos...")
        df, schema = self.plc_system.agent1.analyze(source, column_map=column_map)
        if verbose:
            self.plc_system.agent1.print_schema_report(schema)
        if not schema.is_valid:
            logger.error("✗ Datos inválidos.")
            return self._generic_error_report(schema, client_id)

        result_df = predictor.predict_all(df)
        self.last_result_df = result_df

        logger.info("[3/3] Agente 4 — generando reporte...")
        report_path = self.agent4.generate_generic(
            report_schema=predictor.report_schema(),
            result_df=result_df,
            metrics={"nota": "modelo cargado sin reentrenar — sin métricas de test en esta corrida"},
            feature_importances=predictor.feature_importance(),
            warnings=schema.warnings + ["Predicciones generadas con un modelo pre-entrenado, sin reentrenamiento."],
            client_id=client_id,
        )
        logger.info(f"\n✓ Reporte generado: {report_path}")
        return report_path

    def _predict_only_plc_legacy(self, source, client_id, model_path, column_map, verbose) -> str:
        """
        NOTA DE DISEÑO: esta ruta deliberadamente NO pasa por un Agente 2
        (planificador). A diferencia de las otras dos verticales, aquí no
        hay ninguna decisión de estrategia que tomar — el repositorio
        original que se replica (luisroberto-maker/PLC-failure-prediction-
        pipeline) no tiene un script de entrenamiento propio, solo
        inferencia con un modelo ya fijo. Forzar un Agente 2 aquí sería un
        paso vacío sin ninguna decisión real detrás — se documenta esta
        excepción en vez de fingir simetría donde no la hay.
        """
        from core.legacy_plc_predictor import LegacyPLCPredictor

        model_path = model_path or str(Path(self.cfg.model_dir) / f"modelo_legacy_{client_id}.pkl")
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No se encontró un modelo en {model_path}. Este modo espera el "
                f".pkl entrenado con luisroberto-maker/PLC-failure-prediction-pipeline "
                f"(un estimador sklearn, no un objeto AdaptiveFailurePredictor)."
            )
        logger.info(f"[1/3] Cargando modelo legado desde {model_path}...")
        predictor = LegacyPLCPredictor.from_pretrained(
            model_path,
            umbral_critico=self.cfg.umbral_critico,
            ventana_dias=self.cfg.ventana_dias,
            n_eventos=self.cfg.default_n_eventos,
            n_componentes_pca=self.cfg.n_pca_components,
        )

        logger.info("[2/3] Agente 1 — cargando y validando datos nuevos...")
        df, schema = self.plc_system.agent1.analyze(source, column_map=column_map)
        if verbose:
            self.plc_system.agent1.print_schema_report(schema)
        if not schema.is_valid:
            logger.error("✗ Datos inválidos.")
            return self._generic_error_report(schema, client_id)

        result_df = predictor.predict_all(df)
        self.last_predictor = predictor
        self.last_result_df = result_df

        logger.info("[3/3] Agente 4 — generando reporte...")
        report_path = self.agent4.generate_generic(
            report_schema=predictor.report_schema(),
            result_df=result_df,
            metrics={"nota": "modelo legado — sin métricas de test disponibles en esta corrida"},
            feature_importances=predictor.feature_importance(),
            warnings=schema.warnings + [
                "Predicciones generadas con LegacyPLCPredictor (réplica exacta del "
                "repositorio original). El PCA de embeddings y el LabelEncoder de "
                "categorías se reajustan en cada corrida, igual que en el repo original.",
            ],
            client_id=client_id,
        )
        logger.info(f"\n✓ Reporte generado: {report_path}")
        return report_path

    def get_ranking(self):
        return self.last_result_df


if __name__ == "__main__":
    main()
