"""
agents/agent4_reporter.py — Agente 4: Redactor de reporte.

Responsabilidad: generar un reporte HTML/PDF legible para el ingeniero de
mantenimiento. Incluye: ranking de riesgo, métricas del modelo,
importancia de features, advertencias del sistema, y recomendaciones
en lenguaje natural.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from core.config import (
    AgentConfig, InferredSchema, StrategyPlan,
    ExecutionResult, DataQuality, ModelStrategy,
)
from core.base_predictor import ReportSchema

logger = logging.getLogger(__name__)


class _FakeResult:
    """Adaptador mínimo para reutilizar _tabla_features() fuera del flujo PLC."""
    def __init__(self, feature_importances: dict):
        self.feature_importances = feature_importances


_SHARED_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       color: #1a1a2e; background: #f4f6f9; font-size: 14px; line-height: 1.6; }
.page { max-width: 1100px; margin: 0 auto; padding: 24px; }
header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
         color: white; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; }
header h1 { font-size: 22px; font-weight: 600; }
header p  { opacity: 0.75; font-size: 13px; margin-top: 4px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.card { background: white; border-radius: 10px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 20px; }
.card h2 { font-size: 15px; font-weight: 600; margin-bottom: 14px;
           padding-bottom: 8px; border-bottom: 2px solid #e5e7eb; color: #111827; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: #f9fafb; text-align: left; padding: 8px 12px; font-weight: 600;
     color: #374151; border-bottom: 2px solid #e5e7eb; }
td { padding: 7px 12px; border-bottom: 1px solid #f3f4f6; color: #111827; }
tr:hover td { background: #f9fafb; }
.warn-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px;
            border-radius: 0 8px 8px 0; margin-bottom: 10px; font-size: 13px; color: #92400e; }
.section { margin-bottom: 24px; }
footer { text-align: center; color: #9ca3af; font-size: 12px; margin-top: 32px; }
"""


class ReportWriterAgent:
    """
    Agente 4: Genera el reporte final en HTML (imprimible como PDF).

    El reporte incluye:
      - Resumen ejecutivo
      - Tabla de ranking de riesgo (top N componentes)
      - Métricas del modelo con interpretación
      - Importancia de features (top 10)
      - Métricas por panel
      - Advertencias y recomendaciones
      - Ficha técnica del sistema (schema detectado, estrategia)
    """

    def __init__(self, config: AgentConfig):
        self.cfg = config

    def generate(
        self,
        schema:   InferredSchema,
        plan:     StrategyPlan,
        result:   ExecutionResult,
        client_id: str = "default",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Genera el reporte HTML y lo guarda en disco.

        Returns
        -------
        str: ruta del archivo generado
        """
        html = self._build_html(schema, plan, result, client_id)

        if output_path is None:
            report_dir = Path(self.cfg.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(report_dir / f"reporte_{client_id}_{ts}.html")

        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Reporte generado: {output_path}")
        return output_path

    # ── Reporte genérico dirigido por ReportSchema (multi-vertical) ────────────

    def generate_generic(
        self,
        report_schema: ReportSchema,
        result_df: pd.DataFrame,
        metrics: dict,
        feature_importances: dict,
        warnings: list[str],
        client_id: str = "default",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Genera un reporte HTML para CUALQUIER vertical del framework,
        usando únicamente lo que describe report_schema — sin asumir nada
        específico de PLC ni de desgaste de herramienta. Reutiliza el mismo
        CSS y esqueleto visual que generate().
        """
        ts = datetime.now().strftime("%d/%m/%Y %H:%M")
        rows_html = self._generic_table_rows(report_schema, result_df)
        metrics_html = self._generic_metrics_block(metrics)
        warn_html = "".join(
            f'<div class="warn-box">⚠ {w}</div>' for w in warnings
        ) or "<p style='color:#6b7280'>Sin advertencias.</p>"
        fi_html = self._tabla_features(_FakeResult(feature_importances))

        extra_headers = "".join(f"<th>{label}</th>" for _, label in report_schema.extra_cols)

        html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>{report_schema.domain_label} — {client_id}</title>
<style>
{_SHARED_CSS}
</style></head>
<body><div class="page">
  <header>
    <h1>{report_schema.domain_label}</h1>
    <p>Cliente: <strong>{client_id}</strong> &nbsp;·&nbsp; Generado: <strong>{ts}</strong></p>
  </header>

  <div class="section card">
    <h2>Métricas del modelo</h2>
    {metrics_html}
  </div>

  <div class="section card">
    <h2>Resultados — {report_schema.id_label}</h2>
    <table>
      <thead><tr>
        <th>#</th><th>{report_schema.id_label}</th><th>{report_schema.group_label}</th>
        <th>{report_schema.primary_metric_label}</th><th>Estado</th>
        <th>Confianza</th>{extra_headers}
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  {fi_html}

  <div class="section card">
    <h2>Advertencias</h2>
    {warn_html}
  </div>

  <footer><p>Generado por el framework multi-vertical de predicción — VecTech</p></footer>
</div></body></html>"""

        if output_path is None:
            report_dir = Path(self.cfg.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(report_dir / f"reporte_{client_id}_{ts_file}.html")

        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Reporte genérico generado: {output_path}")
        return output_path

    def _generic_table_rows(self, schema: ReportSchema, df: pd.DataFrame) -> str:
        if df is None or len(df) == 0:
            return "<tr><td colspan='10'>Sin resultados.</td></tr>"

        top = df.head(self.cfg.top_n_ranking)
        rows = []
        for i, row in enumerate(top.itertuples(), 1):
            rid = getattr(row, schema.id_col, "—")
            grp = getattr(row, schema.group_col, "—")
            metric = getattr(row, schema.primary_metric_col, 0)
            status = getattr(row, schema.status_col, "—")
            conf = getattr(row, schema.confidence_col, "—")

            color = "#6b7280"
            for level, hexcolor in schema.status_levels:
                if level == status:
                    color = hexcolor
                    break

            extra_cells = "".join(
                f"<td>{getattr(row, col, '—')}</td>" for col, _ in schema.extra_cols
            )
            rows.append(f"""
        <tr>
          <td>{i}</td><td><strong>{rid}</strong></td><td>{grp}</td>
          <td>{metric}</td>
          <td><span class="badge" style="background:{color}20;color:{color}">{status}</span></td>
          <td>{conf}</td>{extra_cells}
        </tr>""")
        return "".join(rows)

    def _generic_metrics_block(self, metrics: dict) -> str:
        if not metrics:
            return "<p style='color:#6b7280'>Sin métricas disponibles.</p>"
        items = []
        for k, v in metrics.items():
            if isinstance(v, dict):
                sub = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
                items.append(f"<div><strong>{k}</strong>: {sub}</div>")
            else:
                items.append(f"<div><strong>{k}</strong>: {v}</div>")
        return "<div style='font-size:13px;line-height:1.8'>" + "".join(items) + "</div>"

    # ── HTML builder (PLC) ──────────────────────────────────────────────────────

    def _build_html(
        self,
        schema: InferredSchema,
        plan:   StrategyPlan,
        result: ExecutionResult,
        client_id: str,
    ) -> str:
        ts = datetime.now().strftime("%d/%m/%Y %H:%M")
        resumen = self._resumen_ejecutivo(plan, result)
        tabla_ranking = self._tabla_ranking(result)
        tabla_panel   = self._tabla_panel(result)
        tabla_features = self._tabla_features(result)
        advertencias  = self._seccion_advertencias(schema, plan, result)
        ficha_tecnica = self._ficha_tecnica(schema, plan, result)

        return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reporte de Predicción de Fallas — {client_id}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #1a1a2e; background: #f4f6f9; font-size: 14px; line-height: 1.6; }}
    .page {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
              color: white; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; }}
    header h1 {{ font-size: 22px; font-weight: 600; }}
    header p  {{ opacity: 0.75; font-size: 13px; margin-top: 4px; }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
              font-size: 12px; font-weight: 600; }}
    .badge-green  {{ background: #d1fae5; color: #065f46; }}
    .badge-yellow {{ background: #fef3c7; color: #92400e; }}
    .badge-red    {{ background: #fee2e2; color: #991b1b; }}
    .badge-blue   {{ background: #dbeafe; color: #1e40af; }}
    .badge-gray   {{ background: #f3f4f6; color: #374151; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
               margin-bottom: 24px; }}
    .card {{ background: white; border-radius: 10px; padding: 20px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .card h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 14px;
                padding-bottom: 8px; border-bottom: 2px solid #e5e7eb; color: #111827; }}
    .metric-val {{ font-size: 32px; font-weight: 700; color: #1a1a2e; }}
    .metric-lbl {{ font-size: 12px; color: #6b7280; margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ background: #f9fafb; text-align: left; padding: 8px 12px;
          font-weight: 600; color: #374151; border-bottom: 2px solid #e5e7eb; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #f3f4f6; color: #111827; }}
    tr:hover td {{ background: #f9fafb; }}
    .warn-box {{ background: #fffbeb; border-left: 4px solid #f59e0b;
                 padding: 12px 16px; border-radius: 0 8px 8px 0;
                 margin-bottom: 10px; font-size: 13px; color: #92400e; }}
    .info-box {{ background: #eff6ff; border-left: 4px solid #3b82f6;
                 padding: 12px 16px; border-radius: 0 8px 8px 0;
                 margin-bottom: 10px; font-size: 13px; color: #1e40af; }}
    .section {{ margin-bottom: 24px; }}
    footer {{ text-align: center; color: #9ca3af; font-size: 12px; margin-top: 32px; }}
    .progress {{ background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden; }}
    .progress-bar {{ height: 100%; border-radius: 4px; background: #3b82f6; }}
    @media print {{
      body {{ background: white; }}
      .page {{ padding: 0; }}
      header {{ border-radius: 0; }}
    }}
  </style>
</head>
<body>
<div class="page">

  <header>
    <h1>Reporte de Predicción de Fallas en Componentes PLC</h1>
    <p>Cliente: <strong>{client_id}</strong> &nbsp;·&nbsp;
       Generado: <strong>{ts}</strong> &nbsp;·&nbsp;
       Horizonte de predicción: <strong>{self.cfg.ventana_dias} días</strong>
    </p>
  </header>

  <!-- RESUMEN EJECUTIVO -->
  <div class="grid-3">
    {resumen}
  </div>

  <!-- RANKING DE RIESGO -->
  <div class="section card">
    <h2>Ranking de Riesgo — Top {self.cfg.top_n_ranking} componentes</h2>
    {tabla_ranking}
  </div>

  <!-- MÉTRICAS POR PANEL -->
  {tabla_panel}

  <!-- IMPORTANCIA DE FEATURES -->
  {tabla_features}

  <!-- ADVERTENCIAS Y RECOMENDACIONES -->
  {advertencias}

  <!-- FICHA TÉCNICA -->
  <div class="section card">
    <h2>Ficha Técnica del Sistema</h2>
    {ficha_tecnica}
  </div>

  <footer>
    <p>Generado por el Agente de Predicción de Fallas PLC &nbsp;|&nbsp;
       Para uso interno del equipo de mantenimiento.</p>
  </footer>

</div>
</body>
</html>"""

    # ── Secciones ─────────────────────────────────────────────────────────────

    def _resumen_ejecutivo(self, plan: StrategyPlan, result: ExecutionResult) -> str:
        ranking = result.ranking
        n_alto = n_medio = n_bajo = 0
        if ranking is not None and len(ranking) > 0 and "NIVEL_RIESGO" in ranking.columns:
            n_alto  = int((ranking["NIVEL_RIESGO"] == "🔴 ALTO").sum())
            n_medio = int((ranking["NIVEL_RIESGO"] == "🟡 MEDIO").sum())
            n_bajo  = int((ranking["NIVEL_RIESGO"] == "🟢 BAJO").sum())

        auc = result.metrics.get("auc_roc", "N/A")
        auc_badge = ""
        if isinstance(auc, float):
            if auc >= 0.80:   auc_badge = f'<span class="badge badge-green">Bueno</span>'
            elif auc >= 0.65: auc_badge = f'<span class="badge badge-yellow">Aceptable</span>'
            else:             auc_badge = f'<span class="badge badge-red">Bajo</span>'

        return f"""
    <div class="card">
      <h2>Componentes por Nivel de Riesgo</h2>
      <div style="display:flex;gap:16px;align-items:center;margin-top:8px">
        <div><div class="metric-val" style="color:#dc2626">{n_alto}</div>
             <div class="metric-lbl">🔴 Alto riesgo</div></div>
        <div><div class="metric-val" style="color:#d97706">{n_medio}</div>
             <div class="metric-lbl">🟡 Riesgo medio</div></div>
        <div><div class="metric-val" style="color:#059669">{n_bajo}</div>
             <div class="metric-lbl">🟢 Bajo riesgo</div></div>
      </div>
    </div>
    <div class="card">
      <h2>Métricas del Modelo</h2>
      <div class="metric-val">{auc if auc != 'N/A' else '—'} {auc_badge}</div>
      <div class="metric-lbl">AUC-ROC en test</div>
      <div style="margin-top:8px">
        <div class="metric-lbl">AUC-PR: <strong>{result.metrics.get('auc_pr', '—')}</strong></div>
        <div class="metric-lbl">Eventos en test: <strong>{result.metrics.get('n_test', '—')}</strong></div>
      </div>
    </div>
    <div class="card">
      <h2>Resumen del Análisis</h2>
      <div style="font-size:13px;color:#374151;line-height:1.8">
        <div>Estrategia: <strong>{result.strategy_used.value.upper()}</strong></div>
        <div>Componentes analizados: <strong>{result.n_components_predicted}</strong></div>
        <div>Tiempo de ejecución: <strong>{result.execution_time_sec}s</strong></div>
        <div>Umbral crítico: <strong>{self.cfg.umbral_critico} min</strong></div>
      </div>
    </div>"""

    def _tabla_ranking(self, result: ExecutionResult) -> str:
        ranking = result.ranking
        if ranking is None or len(ranking) == 0:
            return "<p style='color:#6b7280'>No hay predicciones disponibles.</p>"

        top = ranking.head(self.cfg.top_n_ranking)
        rows = []
        for i, row in enumerate(top.itertuples(), 1):
            prob    = getattr(row, "PROB_FALLA", 0)
            inc     = getattr(row, "INCERTIDUMBRE", 0)
            nivel   = getattr(row, "NIVEL_RIESGO", "")
            confianza = getattr(row, "NIVEL_CONFIANZA", "")
            modo    = getattr(row, "MODO", "")
            n_ev    = getattr(row, "N_EVENTOS", "—")
            panel   = getattr(row, "PANEL", "—")
            address = getattr(row, "ADDRESS", "—")

            badge_nivel = (
                'badge-red' if 'ALTO' in nivel
                else ('badge-yellow' if 'MEDIO' in nivel else 'badge-green')
            )
            badge_conf = (
                'badge-blue' if confianza == 'ALTO'
                else ('badge-yellow' if confianza == 'MEDIO' else 'badge-gray')
            )
            badge_modo = 'badge-blue' if modo == 'LOCAL' else 'badge-gray'

            pct = int(prob * 100)
            rows.append(f"""
        <tr>
          <td>{i}</td>
          <td><strong>{address}</strong></td>
          <td>{panel}</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div class="progress" style="width:80px">
                <div class="progress-bar" style="width:{pct}%;
                  background:{'#dc2626' if pct>=60 else ('#d97706' if pct>=30 else '#059669')}">
                </div>
              </div>
              <span>{prob:.3f}</span>
            </div>
          </td>
          <td><span class="badge {badge_nivel}">{nivel}</span></td>
          <td>{inc:.3f}</td>
          <td><span class="badge {badge_conf}">{confianza}</span></td>
          <td><span class="badge {badge_modo}">{modo}</span></td>
          <td>{n_ev}</td>
        </tr>""")

        return f"""
    <table>
      <thead>
        <tr>
          <th>#</th><th>Componente</th><th>Panel</th>
          <th>Prob. Falla</th><th>Nivel Riesgo</th>
          <th>Incertidumbre</th><th>Confianza</th>
          <th>Modelo</th><th>N Eventos</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""

    def _tabla_panel(self, result: ExecutionResult) -> str:
        pm = result.panel_metrics
        if pm is None or len(pm) == 0:
            return ""

        rows = []
        for row in pm.itertuples():
            panel   = getattr(row, "PANEL", "—")
            n_comp  = getattr(row, "n_componentes", 0)
            prob_m  = getattr(row, "prob_media", 0)
            n_alto  = getattr(row, "n_alto_riesgo", 0)
            rows.append(f"""
        <tr>
          <td><strong>{panel}</strong></td>
          <td>{n_comp}</td>
          <td>{prob_m:.3f}</td>
          <td><span class="badge {'badge-red' if n_alto > 0 else 'badge-green'}">{n_alto}</span></td>
        </tr>""")

        return f"""
  <div class="section card">
    <h2>Análisis por Panel / Línea</h2>
    <table>
      <thead>
        <tr><th>Panel</th><th>Componentes</th>
            <th>Prob. Media Falla</th><th>En Alto Riesgo</th></tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>"""

    def _tabla_features(self, result: ExecutionResult) -> str:
        fi = result.feature_importances
        if not fi:
            return ""

        top10 = list(fi.items())[:10]
        max_v = top10[0][1] if top10 else 1

        rows = []
        for feat, imp in top10:
            pct = int(imp / max_v * 100)
            rows.append(f"""
        <tr>
          <td style="font-family:monospace">{feat}</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div class="progress" style="width:120px">
                <div class="progress-bar" style="width:{pct}%"></div>
              </div>
              <span>{imp:.4f}</span>
            </div>
          </td>
        </tr>""")

        return f"""
  <div class="section card">
    <h2>Importancia de Variables (Top 10)</h2>
    <table>
      <thead><tr><th>Variable</th><th>Importancia relativa</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>"""

    def _seccion_advertencias(
        self,
        schema: InferredSchema,
        plan:   StrategyPlan,
        result: ExecutionResult,
    ) -> str:
        advertencias = []
        for w in schema.warnings + plan.warnings:
            advertencias.append(f'<div class="warn-box">⚠ {w}</div>')

        # Recomendaciones automáticas basadas en el resultado
        recomendaciones = []
        if result.ranking is not None and len(result.ranking) > 0:
            alto_riesgo = result.ranking[result.ranking["NIVEL_RIESGO"] == "🔴 ALTO"]
            if len(alto_riesgo) > 0:
                componentes = ", ".join(alto_riesgo["ADDRESS"].head(5).tolist())
                recomendaciones.append(
                    f"Programar inspección preventiva en los próximos "
                    f"{self.cfg.ventana_dias} días para: <strong>{componentes}</strong>"
                )

            baja_confianza = result.ranking[result.ranking["NIVEL_CONFIANZA"] == "BAJO"]
            if len(baja_confianza) > 0:
                recomendaciones.append(
                    f"{len(baja_confianza)} componentes tienen predicciones de baja "
                    "confianza (datos insuficientes). Aumentar el registro histórico "
                    "mejorará la precisión."
                )

        metricas = result.metrics
        auc = metricas.get("auc_roc", 0)
        if isinstance(auc, float) and auc < 0.65:
            recomendaciones.append(
                f"El AUC-ROC ({auc:.3f}) es bajo. Considerar: aumentar el dataset, "
                "revisar el umbral_critico, o añadir más variables de contexto."
            )

        bloques = []
        if advertencias:
            bloques.append(f"""
  <div class="section card">
    <h2>Advertencias del Sistema</h2>
    {''.join(advertencias)}
  </div>""")

        if recomendaciones:
            recs_html = "".join(
                f'<div class="info-box">💡 {r}</div>' for r in recomendaciones
            )
            bloques.append(f"""
  <div class="section card">
    <h2>Recomendaciones</h2>
    {recs_html}
  </div>""")

        return "\n".join(bloques)

    def _ficha_tecnica(
        self,
        schema: InferredSchema,
        plan:   StrategyPlan,
        result: ExecutionResult,
    ) -> str:
        return f"""
    <table>
      <tbody>
        <tr><td><strong>Calidad de datos</strong></td>
            <td>{plan.data_quality.value.upper()}</td></tr>
        <tr><td><strong>Estrategia de modelo</strong></td>
            <td>{result.strategy_used.value.upper()}</td></tr>
        <tr><td><strong>Total de eventos</strong></td>
            <td>{schema.n_rows:,}</td></tr>
        <tr><td><strong>Rango temporal</strong></td>
            <td>{schema.date_range_days:.0f} días</td></tr>
        <tr><td><strong>Componentes únicos</strong></td>
            <td>{schema.n_components:,}</td></tr>
        <tr><td><strong>Componentes con modelo local</strong></td>
            <td>{plan.n_components_with_enough_data}</td></tr>
        <tr><td><strong>Umbral crítico usado</strong></td>
            <td>{plan.recommended_umbral_critico} min</td></tr>
        <tr><td><strong>Ventana de predicción</strong></td>
            <td>{self.cfg.ventana_dias} días</td></tr>
        <tr><td><strong>N eventos por componente</strong></td>
            <td>{plan.recommended_n_eventos}</td></tr>
        <tr><td><strong>Embeddings de texto</strong></td>
            <td>{'Activados' if schema.has_text_logs else 'Desactivados'}</td></tr>
        <tr><td><strong>Mapeo de columnas detectado</strong></td>
            <td>START TIME←{schema.col_timestamp_start} |
                ADDRESS←{schema.col_component} |
                FAIL COMMENT←{schema.col_log_text} |
                PANEL←{schema.col_panel}</td></tr>
      </tbody>
    </table>"""
