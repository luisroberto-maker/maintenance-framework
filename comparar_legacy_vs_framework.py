"""
comparar_legacy_vs_framework.py — Compara las predicciones del modelo
legado (LegacyPLCPredictor, que hereda el LabelEncoder y el PCA que se
reajustan en cada corrida) contra un AdaptiveFailurePredictor entrenado
desde cero con los mismos datos reales.

Sirve para diagnosticar si un patrón como "muchos componentes en ALTO con
confianza BAJA" viene de una limitación conocida del modelo legado, o si
es una señal real que ambos modelos coinciden en ver.

NOTA: este archivo es una PLANTILLA pensada para que la edites con tus
propias rutas — por eso NO está en CHECKSUMS.txt.

Uso:
    1. Ajusta la sección "CONFIGURA AQUÍ" abajo.
    2. Ejecuta:  python comparar_legacy_vs_framework.py

Qué hace:
    1. Entrena un AdaptiveFailurePredictor nuevo con tu Excel real
       (SafeEncoder con vocabulario fijo + PCA ajustado solo en train —
       sin las dos limitaciones conocidas del modelo legado).
    2. Vuelve a predecir con tu modelo legado ya entrenado, sobre los
       mismos datos.
    3. Muestra ambos resultados lado a lado, por componente, y un
       resumen de cuántos componentes quedan con confianza baja en
       cada uno.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from core.logging_config import setup_logging
setup_logging(level="WARNING")  # silencioso — solo nos interesa la tabla final

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURA AQUÍ
# ══════════════════════════════════════════════════════════════════════════
RUTA_EXCEL         = "ruta/a/tu/Registro_PLC.xlsx"
RUTA_MODELO_LEGACY = "ruta/a/tu/modelo.pkl"
CLIENT_ID          = "comparacion"
TOP_N              = 20
# ══════════════════════════════════════════════════════════════════════════


def main():
    if not Path(RUTA_EXCEL).exists():
        print(f"✗ No encuentro el archivo Excel: {RUTA_EXCEL}")
        sys.exit(1)
    if not Path(RUTA_MODELO_LEGACY).exists():
        print(f"✗ No encuentro el modelo legado: {RUTA_MODELO_LEGACY}")
        sys.exit(1)

    from orchestrator import MultiVerticalAgentSystem

    print("=== 1/2: Entrenando AdaptiveFailurePredictor (framework) desde cero ===")
    print("    (esto entrena un modelo nuevo — puede tardar unos minutos)")
    system_framework = MultiVerticalAgentSystem()
    system_framework.run(RUTA_EXCEL, client_id=CLIENT_ID, verbose=False)
    ranking_framework = system_framework.get_ranking().copy()
    ranking_framework = ranking_framework.rename(columns={
        "ADDRESS": "COMPONENTE",
        "PROB_FALLA": "PROB_FRAMEWORK",
        "NIVEL_RIESGO": "RIESGO_FRAMEWORK",
        "NIVEL_CONFIANZA": "CONFIANZA_FRAMEWORK",
    })[["COMPONENTE", "PROB_FRAMEWORK", "RIESGO_FRAMEWORK", "CONFIANZA_FRAMEWORK"]]
    print(f"    → {len(ranking_framework)} componentes evaluados por el framework")

    print()
    print("=== 2/2: Prediciendo con el modelo legado ya entrenado ===")
    system_legacy = MultiVerticalAgentSystem()
    system_legacy.predict_only(
        source=RUTA_EXCEL,
        client_id=f"{CLIENT_ID}_legacy",
        domain="plc_failure",
        legacy=True,
        model_path=RUTA_MODELO_LEGACY,
        verbose=False,
    )
    ranking_legacy = system_legacy.get_ranking().copy()
    ranking_legacy = ranking_legacy.rename(columns={
        "PROBABILIDAD_FALLA": "PROB_LEGACY",
        "NIVEL_RIESGO": "RIESGO_LEGACY",
        "NIVEL_CONFIANZA": "CONFIANZA_LEGACY",
    })[["COMPONENTE", "PROB_LEGACY", "RIESGO_LEGACY", "CONFIANZA_LEGACY"]]
    print(f"    → {len(ranking_legacy)} componentes evaluados por el modelo legado")

    # ── Comparación lado a lado ──────────────────────────────────────────────
    comparacion = ranking_legacy.merge(ranking_framework, on="COMPONENTE", how="outer")
    comparacion = comparacion.sort_values("PROB_LEGACY", ascending=False, na_position="last")

    print()
    print("=" * 100)
    print(f"COMPARACIÓN — Top {TOP_N} por probabilidad según el modelo legado")
    print("=" * 100)
    print(comparacion.head(TOP_N).to_string(index=False))

    # ── Resumen de confianza ─────────────────────────────────────────────────
    n_legacy_bajo    = (ranking_legacy["CONFIANZA_LEGACY"] == "BAJO").sum()
    n_framework_bajo = (ranking_framework["CONFIANZA_FRAMEWORK"] == "BAJO").sum()
    n_legacy_alto_riesgo    = (ranking_legacy["RIESGO_LEGACY"] == "🔴 ALTO").sum()
    n_framework_alto_riesgo = (ranking_framework["RIESGO_FRAMEWORK"] == "🔴 ALTO").sum()

    print()
    print("=" * 100)
    print("RESUMEN")
    print("=" * 100)
    print(f"  Componentes en 🔴 ALTO   — legado: {n_legacy_alto_riesgo}/{len(ranking_legacy)}"
          f"  |  framework: {n_framework_alto_riesgo}/{len(ranking_framework)}")
    print(f"  Componentes confianza BAJA — legado: {n_legacy_bajo}/{len(ranking_legacy)}"
          f"  |  framework: {n_framework_bajo}/{len(ranking_framework)}")
    print()
    if n_framework_bajo < n_legacy_bajo * 0.5:
        print("  → El framework muestra bastante menos incertidumbre que el modelo legado.")
        print("    Esto es consistente con las limitaciones conocidas del legado (LabelEncoder")
        print("    y PCA reajustados en cada corrida) — no necesariamente significa que tus")
        print("    componentes estén más sanos de lo que el legado sugiere, sino que el")
        print("    framework tiene más certeza sobre su propia predicción.")
    else:
        print("  → Ambos modelos muestran niveles de confianza similares. El patrón de")
        print("    riesgo/confianza que viste no parece explicarse solo por las limitaciones")
        print("    conocidas del modelo legado — vale la pena revisar esos componentes con")
        print("    más atención (o con más datos históricos, si los tienes).")


if __name__ == "__main__":
    main()
