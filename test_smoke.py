"""
test_smoke.py — Prueba rápida de que el entorno local quedó bien configurado.

No usa datos reales: genera datos sintéticos con la MISMA FORMA que
PHM 2010 (133 features) y que un log de eventos PLC, y corre el sistema
completo end-to-end en ambas verticales. Si esto pasa sin errores, tu
entorno está listo para trabajar con datos reales.

Uso:
    python test_smoke.py
"""
import sys
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core.logging_config import setup_logging
setup_logging(level="INFO")  # sin esto, el progreso interno de los agentes no se ve

np.random.seed(42)

TEST_DIR = Path("/tmp/plc_agent_smoke_test")


def print_step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def simulate_tool_wear_experiment(exp_name, n_cuts=150, wear_rate=0.0012):
    """Genera pasadas sintéticas con la misma forma que el dataset PHM 2010
    (133 features correlacionadas con el desgaste real)."""
    from core.tool_wear_predictor import FEAT_COLS

    rows = []
    vb = 0.0
    for cut in range(1, n_cuts + 1):
        vb += wear_rate * (1 + 0.3 * np.sin(cut / 20)) + np.random.normal(0, 0.0005)
        vb = max(0, vb)
        feat = {c: vb * 10 + np.random.normal(0, 0.5) for c in FEAT_COLS}
        feat.update({"experiment": exp_name, "cut": cut, "VB": vb})
        rows.append(feat)
    return pd.DataFrame(rows)


def simulate_plc_events(n=300):
    """Genera eventos sintéticos con la misma forma que un log real de PLC.

    IMPORTANTE: la duración debe variar (algunos eventos superando el
    umbral_critico por defecto de 30 min) para que TARGET tenga ambas
    clases (0 y 1) presentes. Si todos los eventos duran lo mismo y por
    debajo del umbral, ningún modelo puede entrenar — RandomForestClassifier
    con una sola clase produce predict_proba() de una sola columna y
    predict_proba()[:, 1] lanza IndexError.
    """
    start = pd.date_range("2024-01-01", periods=n, freq="4h")
    # ~15% de eventos "críticos" (30-90 min), el resto breves (1-25 min)
    duracion_min = np.where(
        np.random.rand(n) < 0.15,
        np.random.uniform(30, 90, n),
        np.random.uniform(1, 25, n),
    )
    return pd.DataFrame({
        "START TIME": start,
        "END TIME": start + pd.to_timedelta(duracion_min, unit="m"),
        "ADDRESS": np.random.choice(["COMP_A", "COMP_B", "COMP_C", "COMP_D"], n),
        "FAIL COMMENT": np.random.choice(
            ["Emergency Stop", "PLC Connect Error(-1)", "C1 Heavy Error", "R/B-1 FAULT"], n
        ),
        "PANEL": np.random.choice(["LINEA_1", "LINEA_2"], n),
    })


def test_imports():
    print_step("1/6 — Verificando imports y dependencias")
    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ✗ scikit-learn NO instalado: {e}")
        return False

    try:
        import xgboost
        print(f"  ✓ xgboost {xgboost.__version__}")
    except Exception as e:
        print(f"  ⚠ xgboost no disponible ({type(e).__name__}) — ToolWearPredictor usará "
              f"solo Random Forest (OK, es opcional)")
        if "libomp" in str(e).lower() or "dylib" in str(e).lower():
            print("    → En macOS esto se resuelve con: brew install libomp")

    try:
        import sentence_transformers
        print(f"  ✓ sentence-transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ sentence-transformers NO instalado — la vertical PLC fallará: {e}")
        return False

    try:
        from core.config import AgentConfig
        from core.base_predictor import BasePredictor, ReportSchema
        from core.tool_wear_predictor import ToolWearPredictor, FEAT_COLS
        from agents.domain_router import detect_domain, Domain
        from orchestrator import MultiVerticalAgentSystem
        print("  ✓ Todos los módulos del proyecto importan correctamente")
    except ImportError as e:
        print(f"  ✗ Error importando módulos del proyecto: {e}")
        return False

    return True


def test_domain_detection():
    print_step("2/6 — Verificando detección automática de dominio")
    from agents.domain_router import detect_domain, Domain

    df_wear = simulate_tool_wear_experiment("c1", n_cuts=20)
    domain_wear = detect_domain(df_wear)
    assert domain_wear == Domain.TOOL_WEAR, f"Esperaba TOOL_WEAR, obtuve {domain_wear}"
    print(f"  ✓ Datos de señal → detectado como {domain_wear.value}")

    df_plc = simulate_plc_events(n=20)
    domain_plc = detect_domain(df_plc)
    assert domain_plc == Domain.PLC_FAILURE, f"Esperaba PLC_FAILURE, obtuve {domain_plc}"
    print(f"  ✓ Datos de eventos → detectado como {domain_plc.value}")


def test_tool_wear_pipeline():
    print_step("3/6 — Verificando pipeline completo de desgaste de herramienta")
    from core.config import AgentConfig
    from orchestrator import MultiVerticalAgentSystem

    df_wear = pd.concat([
        simulate_tool_wear_experiment("c1", 150, 0.0013),
        simulate_tool_wear_experiment("c4", 150, 0.0011),
        simulate_tool_wear_experiment("c6", 150, 0.0012),
    ], ignore_index=True)

    cfg = AgentConfig(
        output_dir=str(TEST_DIR),
        model_dir=str(TEST_DIR / "models"),
        report_dir=str(TEST_DIR / "reports"),
    )
    system = MultiVerticalAgentSystem(config=cfg)
    report_path = system.run(df_wear, client_id="smoke_test_wear", verbose=False)

    assert Path(report_path).exists(), "El reporte no se generó"
    ranking = system.get_ranking()
    assert ranking is not None and len(ranking) > 0, "El ranking está vacío"
    print(f"  ✓ Pipeline completo — {len(ranking)} predicciones generadas")
    print(f"  ✓ Reporte: {report_path}")


def test_plc_pipeline():
    print_step("4/6 — Verificando pipeline completo de fallas PLC")
    print("  (requiere descargar el modelo de embeddings ~91 MB la primera vez)")
    from core.config import AgentConfig
    from orchestrator import MultiVerticalAgentSystem

    df_plc = simulate_plc_events(n=300)
    cfg = AgentConfig(
        output_dir=str(TEST_DIR),
        model_dir=str(TEST_DIR / "models"),
        report_dir=str(TEST_DIR / "reports"),
    )
    system = MultiVerticalAgentSystem(config=cfg)
    report_path = system.run(df_plc, client_id="smoke_test_plc", verbose=False)

    assert Path(report_path).exists(), "El reporte no se generó"
    ranking = system.get_ranking()
    assert ranking is not None and len(ranking) > 0, "El ranking está vacío"
    print(f"  ✓ Pipeline completo — {len(ranking)} predicciones generadas")
    print(f"  ✓ Reporte: {report_path}")


def test_legacy_plc_pipeline():
    print_step("5/6 — Verificando LegacyPLCPredictor (modelo legado)")
    from core.legacy_plc_predictor import LegacyPLCPredictor
    from core.config import AgentConfig
    from orchestrator import MultiVerticalAgentSystem

    df_plc = simulate_plc_events(n=300)

    predictor = LegacyPLCPredictor(umbral_critico=30, ventana_dias=7, n_eventos=10, n_componentes_pca=15)
    predictor.fit(df_plc)

    import joblib
    model_path = TEST_DIR / "modelo_legacy_smoke.pkl"
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor.modelo, model_path)

    cfg = AgentConfig(
        output_dir=str(TEST_DIR),
        model_dir=str(TEST_DIR / "models"),
        report_dir=str(TEST_DIR / "reports"),
    )
    system = MultiVerticalAgentSystem(config=cfg)
    report_path = system.predict_only(
        source=df_plc.sample(50, random_state=1),
        client_id="smoke_test_legacy",
        domain="plc_failure",
        legacy=True,
        model_path=str(model_path),
        verbose=False,
    )
    assert Path(report_path).exists(), "El reporte no se generó"
    ranking = system.get_ranking()
    assert ranking is not None and len(ranking) > 0, "El ranking está vacío"
    print(f"  ✓ Modelo legado — {len(ranking)} predicciones generadas")
    print(f"  ✓ Reporte: {report_path}")


def cleanup():
    print_step("6/6 — Limpiando archivos temporales")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    print(f"  ✓ {TEST_DIR} eliminado")


def main():
    results = {}

    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n✗ Faltan dependencias críticas — corrige antes de continuar.")
        print("  Ejecuta: pip install -r requirements.txt")
        sys.exit(1)

    try:
        test_domain_detection()
        results["domain_detection"] = True
    except Exception as e:
        print(f"  ✗ Falló: {e}")
        results["domain_detection"] = False

    try:
        test_tool_wear_pipeline()
        results["tool_wear"] = True
    except Exception as e:
        print(f"  ✗ Falló: {e}")
        results["tool_wear"] = False

    try:
        test_plc_pipeline()
        results["plc"] = True
    except Exception as e:
        print(f"  ✗ Falló: {e}")
        print("    (si el error menciona 'huggingface.co' o conexión, revisa tu acceso a internet")
        print("     — es necesario solo la primera vez, para descargar el modelo de embeddings)")
        results["plc"] = False

    try:
        test_legacy_plc_pipeline()
        results["legacy_plc"] = True
    except Exception as e:
        print(f"  ✗ Falló: {e}")
        results["legacy_plc"] = False

    cleanup()

    print_step("RESUMEN")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    if all(results.values()):
        print("\n✓ Entorno listo — puedes trabajar con datos reales.")
        sys.exit(0)
    else:
        print("\n⚠ Hay pasos que fallaron — revisa los mensajes arriba antes de continuar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
