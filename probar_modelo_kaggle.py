"""
probar_modelo_kaggle.py — Prueba tu modelo de desgaste entrenado en Kaggle
con tus CSV de señal cruda (uno por pasada/corte).

NOTA: este archivo es una PLANTILLA pensada para que la edites con tus
propias rutas — por eso NO está en CHECKSUMS.txt. Es normal y esperado
que tu copia sea distinta a la original una vez que la personalices.

Uso:
    1. Ajusta la sección "CONFIGURA AQUÍ" abajo.
    2. Ejecuta:  python probar_modelo_kaggle.py

Formato esperado de tus archivos:
    - Los 4 artefactos de Kaggle (model_rf.joblib, model_xgb.joblib,
      scaler.joblib, metadata.json) juntos en UNA carpeta.
    - Los CSV de señal cruda (columnas fx, fy, fz, vx, vy, vz, ae, sin
      encabezado) juntos en OTRA carpeta, un archivo por pasada, con
      nombres terminados en "_<número>.csv" — ej. c_nueva_001.csv,
      c_nueva_002.csv, ... El número al final es el número de pasada/corte.
    - NO necesitas un archivo de desgaste (wear.csv) para esto — es
      justamente lo que el modelo va a predecir.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from core.logging_config import setup_logging
setup_logging(level="INFO")  # sin esto, los mensajes de progreso no se ven

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURA AQUÍ
# ══════════════════════════════════════════════════════════════════════════
CARPETA_ARTEFACTOS_KAGGLE = "ruta/a/carpeta/con/model_rf.joblib_y_los_demas"
CARPETA_CSV_SENAL         = "ruta/a/carpeta/con/tus_csv_de_senal"
NOMBRE_HERRAMIENTA        = "herramienta_nueva"   # nombre libre para identificarla en el reporte
CLIENT_ID                 = "prueba_kaggle"
# ══════════════════════════════════════════════════════════════════════════


def verificar_artefactos(carpeta: str):
    carpeta = Path(carpeta)
    requeridos = ["model_rf.joblib", "scaler.joblib"]
    opcionales = ["model_xgb.joblib", "metadata.json"]

    faltantes = [f for f in requeridos if not (carpeta / f).exists()]
    if faltantes:
        print(f"✗ Faltan archivos obligatorios en {carpeta}: {faltantes}")
        sys.exit(1)

    for f in opcionales:
        estado = "✓" if (carpeta / f).exists() else "⚠ (opcional, no está)"
        print(f"  {estado} {f}")


def verificar_csvs(carpeta: str):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        print(f"✗ No existe la carpeta: {carpeta}")
        sys.exit(1)

    csvs = sorted(carpeta.glob("*.csv"))
    if not csvs:
        print(f"✗ No hay archivos .csv en {carpeta}")
        sys.exit(1)

    print(f"  ✓ {len(csvs)} archivos CSV encontrados")

    # Verificar que al menos el primero tenga nombre parseable (_<numero>.csv)
    primero = csvs[0].stem
    try:
        int(primero.split("_")[-1])
    except ValueError:
        print(
            f"  ⚠ El archivo '{csvs[0].name}' no termina en '_<número>.csv' "
            f"— el sistema no podrá identificar el número de pasada. "
            f"Renombra tus archivos a un patrón como 'nombre_001.csv'."
        )
    return len(csvs)


def main():
    print("Verificando artefactos del modelo de Kaggle...")
    verificar_artefactos(CARPETA_ARTEFACTOS_KAGGLE)
    print()

    print("Verificando archivos CSV de señal...")
    n_csvs = verificar_csvs(CARPETA_CSV_SENAL)
    print()

    from core.config import AgentConfig
    from orchestrator import MultiVerticalAgentSystem

    config = AgentConfig(
        output_dir="output",
        model_dir="output/models",
        report_dir="output/reports",
    )
    system = MultiVerticalAgentSystem(config=config)

    # La fuente es un dict de "experimentos" — aquí solo uno, sin wear file
    # porque estamos haciendo inferencia pura (el desgaste es lo que se
    # quiere predecir, no algo que ya se sepa).
    fuente = {
        NOMBRE_HERRAMIENTA: {
            "signals": CARPETA_CSV_SENAL,
            # "wear": "..."  ← descomenta y agrega la ruta SOLO si tienes
            #                   el desgaste real medido y quieres comparar
        }
    }

    print(f"Extrayendo features de {n_csvs} pasadas y prediciendo "
          f"(esto puede tardar unos minutos)...")
    print()

    reporte = system.predict_only(
        source=fuente,
        client_id=CLIENT_ID,
        domain="tool_wear",
        artifacts_dir=CARPETA_ARTEFACTOS_KAGGLE,
    )

    print()
    print("═" * 60)
    print(f"✓ Reporte generado: {reporte}")
    print("═" * 60)

    ranking = system.get_ranking()
    if ranking is not None and len(ranking) > 0:
        print(f"\nPredicciones por pasada:\n")
        cols = [c for c in
            ["COMPONENT", "CUT", "VB_ESTIMADO", "ESTADO",
             "RUL_ESTIMADO", "INCERTIDUMBRE", "NIVEL_CONFIANZA"]
            if c in ranking.columns]
        print(ranking[cols].to_string(index=False))

    print(f"\nAbre el reporte HTML en tu navegador:")
    print(f"  open '{reporte}'   (macOS)")


if __name__ == "__main__":
    main()
