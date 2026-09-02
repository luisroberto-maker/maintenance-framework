"""
probar_modelo_real.py — Prueba tu modelo .pkl ya entrenado con tu Excel real.

NOTA: este archivo es una PLANTILLA pensada para que la edites con tus
propias rutas — por eso NO está en CHECKSUMS.txt. Es normal y esperado
que tu copia sea distinta a la original una vez que la personalices.

Uso:
    1. Ajusta las 2 rutas de la sección "CONFIGURA AQUÍ" abajo.
    2. Ejecuta:  python probar_modelo_real.py
    3. El script detecta automáticamente si tu .pkl es un
       AdaptiveFailurePredictor (framework) o un modelo legado (repo
       original) y llama a predict_only() con los parámetros correctos.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import joblib

from core.logging_config import setup_logging
setup_logging(level="INFO")  # sin esto, los mensajes de progreso no se ven

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURA AQUÍ
# ══════════════════════════════════════════════════════════════════════════
RUTA_EXCEL   = "ruta/a/tu/Registro_PLC.xlsx"     # ← tu archivo de logs
RUTA_MODELO  = "ruta/a/tu/modelo.pkl"            # ← tu modelo ya entrenado
CLIENT_ID    = "prueba_real"                     # ← identificador libre
# ══════════════════════════════════════════════════════════════════════════


def detectar_tipo_de_modelo(ruta_pkl: str) -> str:
    """
    Distingue entre:
      - 'framework' → AdaptiveFailurePredictor guardado con .save() de
        este framework (objeto completo: encoders, PCA, RF, etc.)
      - 'legacy'    → estimador de scikit-learn "pelado" (RandomForestClassifier,
        etc.), tal como lo guarda el repositorio original
        luisroberto-maker/PLC-failure-prediction-pipeline
    """
    obj = joblib.load(ruta_pkl)
    tipo = type(obj).__name__
    modulo = type(obj).__module__

    if "AdaptiveFailurePredictor" in tipo:
        return "framework"
    elif "sklearn" in modulo or hasattr(obj, "predict_proba"):
        return "legacy"
    else:
        raise ValueError(
            f"No reconozco el tipo de objeto guardado en {ruta_pkl}: "
            f"{modulo}.{tipo}. Revisa manualmente con qué se entrenó."
        )


def main():
    if not Path(RUTA_EXCEL).exists():
        print(f"✗ No encuentro el archivo Excel: {RUTA_EXCEL}")
        print("  Edita RUTA_EXCEL en este script con la ruta correcta.")
        sys.exit(1)

    if not Path(RUTA_MODELO).exists():
        print(f"✗ No encuentro el modelo: {RUTA_MODELO}")
        print("  Edita RUTA_MODELO en este script con la ruta correcta.")
        sys.exit(1)

    print(f"Excel:  {RUTA_EXCEL}")
    print(f"Modelo: {RUTA_MODELO}")
    print()

    print("Detectando tipo de modelo...")
    tipo = detectar_tipo_de_modelo(RUTA_MODELO)
    print(f"  → Tipo detectado: {tipo}")
    print()

    from core.config import AgentConfig
    from orchestrator import MultiVerticalAgentSystem

    config = AgentConfig(
        output_dir="output",
        model_dir="output/models",
        report_dir="output/reports",
    )
    system = MultiVerticalAgentSystem(config=config)

    if tipo == "framework":
        reporte = system.predict_only(
            source=RUTA_EXCEL,
            client_id=CLIENT_ID,
            domain="plc_failure",
            model_path=RUTA_MODELO,
        )
    else:  # legacy
        reporte = system.predict_only(
            source=RUTA_EXCEL,
            client_id=CLIENT_ID,
            domain="plc_failure",
            legacy=True,
            model_path=RUTA_MODELO,
        )

    print()
    print("═" * 60)
    print(f"✓ Reporte generado: {reporte}")
    print("═" * 60)

    ranking = system.get_ranking()
    if ranking is not None and len(ranking) > 0:
        print(f"\nTop 10 componentes por riesgo:\n")
        cols_disponibles = [c for c in
            ["ADDRESS", "COMPONENTE", "PROB_FALLA", "PROBABILIDAD_FALLA",
             "NIVEL_RIESGO", "NIVEL_CONFIANZA", "MODO"]
            if c in ranking.columns]
        print(ranking[cols_disponibles].head(10).to_string(index=False))

    print(f"\nAbre el reporte HTML en tu navegador para verlo con formato:")
    print(f"  open '{reporte}'   (macOS)")


if __name__ == "__main__":
    main()
