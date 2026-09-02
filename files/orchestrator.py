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
            print(BANNER)

        # ── AGENTE 1: Analizar datos ──────────────────────────────────────────
        print("\n[1/4] Agente 1 — Analizando datos...")
        df, schema = self.agent1.analyze(source, column_map=column_map)
        self.last_schema = schema
        if verbose:
            self.agent1.print_schema_report(schema)

        if not schema.is_valid:
            print("✗ Error en datos de entrada. Revisa las advertencias arriba.")
            return self._generar_reporte_error(schema, client_id)

        # ── AGENTE 2: Planificar estrategia ───────────────────────────────────
        print("[2/4] Agente 2 — Planificando estrategia...")
        plan = self.agent2.plan(df, schema)
        self.last_plan = plan
        if verbose:
            self.agent2.print_plan_report(plan)

        # ── AGENTE 3: Ejecutar pipeline ───────────────────────────────────────
        print("[3/4] Agente 3 — Ejecutando pipeline de ML...")
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
        print("[4/4] Agente 4 — Generando reporte...")
        report_path = self.agent4.generate(
            schema=schema, plan=plan, result=result, client_id=client_id
        )
        self.last_report = report_path
        print(f"\n✓ Reporte generado: {report_path}")
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
        print("\n[Actualización incremental]")

        df, schema = self.agent1.analyze(source, column_map=column_map)
        if not schema.is_valid:
            print("✗ Error en datos de actualización.")
            return ""

        n_actualizados = self.agent3.update(df, client_id=client_id)
        print(f"  → {n_actualizados} modelos locales actualizados.")

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
        print(f"\n{'─'*60}")
        print(f"  RANKING DE RIESGO — Top {self.cfg.top_n_ranking}")
        print(f"{'─'*60}")
        print(f"  {'COMPONENTE':<20} {'PANEL':<15} {'PROB':>6} {'NIVEL':<12} {'CONF':<8} {'MODO'}")
        print(f"  {'─'*56}")
        for row in ranking.head(self.cfg.top_n_ranking).itertuples():
            print(
                f"  {row.ADDRESS:<20} "
                f"{str(row.PANEL):<15} "
                f"{row.PROB_FALLA:>6.3f} "
                f"{row.NIVEL_RIESGO:<12} "
                f"{row.NIVEL_CONFIANZA:<8} "
                f"{row.MODO}"
            )
        print(f"{'─'*60}\n")

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
    args = parser.parse_args()

    config = AgentConfig(
        umbral_critico = args.umbral,
        ventana_dias   = args.ventana,
        output_dir     = args.output,
        model_dir      = f"{args.output}/models",
        report_dir     = f"{args.output}/reports",
    )

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
    print(f"\nReporte disponible en: {report}")


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
            print(f"\n[Router] Dominio detectado: {domain.value}")

        if domain == Domain.PLC_FAILURE:
            return self.plc_system.run(source, client_id=client_id,
                                       column_map=column_map, verbose=verbose)

        if domain == Domain.TOOL_WEAR:
            return self._run_tool_wear(source, client_id, column_map, verbose)

        # UNKNOWN — por defecto se intenta como PLC (vertical original del producto)
        # ya que su detección de schema es más tolerante a datos ambiguos.
        if verbose:
            print("[Router] Dominio ambiguo — intentando como fallas PLC por defecto. "
                  "Usa force_domain='tool_wear' si corresponde a señales de sensor.")
        return self.plc_system.run(source, client_id=client_id,
                                   column_map=column_map, verbose=verbose)

    def _run_tool_wear(self, source, client_id: str,
                       column_map: Optional[dict], verbose: bool) -> str:
        print("\n[1/3] Agente 1 (desgaste) — cargando y validando señales...")
        df, schema = self.tool_wear_loader.analyze(source, column_map=column_map)
        if verbose:
            self.tool_wear_loader.print_schema_report(schema)

        if not schema.is_valid:
            print("✗ Datos inválidos para la vertical de desgaste de herramienta.")
            return self._generic_error_report(schema, client_id)

        print("[2/3] Agente 3 (desgaste) — entrenando y prediciendo...")
        predictor = ToolWearPredictor()
        metrics = predictor.fit(df)
        result_df = predictor.predict_all(df)
        self.last_predictor = predictor
        self.last_result_df = result_df

        model_dir = Path(self.cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        predictor.save(str(model_dir / f"tool_wear_predictor_{client_id}.pkl"))

        print("[3/3] Agente 4 — generando reporte...")
        report_path = self.agent4.generate_generic(
            report_schema=predictor.report_schema(),
            result_df=result_df,
            metrics=metrics,
            feature_importances=predictor.feature_importance(),
            warnings=schema.warnings,
            client_id=client_id,
        )
        print(f"\n✓ Reporte generado: {report_path}")
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

    def get_ranking(self):
        if self.last_domain == Domain.TOOL_WEAR:
            return self.last_result_df
        return self.plc_system.get_ranking()


if __name__ == "__main__":
    main()
