"""
tests/test_domain_router.py — Detección automática de dominio (PLC vs
desgaste de herramienta).
"""
from agents.domain_router import detect_domain, Domain


def test_detecta_eventos_plc(df_plc_events):
    assert detect_domain(df_plc_events) == Domain.PLC_FAILURE


def test_detecta_desgaste_herramienta(df_tool_wear_multi_experiment):
    assert detect_domain(df_tool_wear_multi_experiment) == Domain.TOOL_WEAR


def test_dominio_forzado_por_column_map(df_plc_events):
    """El usuario puede forzar el dominio explícitamente, saltándose la
    heurística por completo."""
    resultado = detect_domain(df_plc_events, column_map={"domain": "tool_wear"})
    assert resultado == Domain.TOOL_WEAR


def test_dataframe_ambiguo_no_falla():
    """Un DataFrame sin ninguna señal reconocible no debe lanzar excepción
    — debe devolver UNKNOWN para que el orquestador decida el fallback."""
    import pandas as pd
    df_ambiguo = pd.DataFrame({"col_x": [1, 2, 3], "col_y": ["a", "b", "c"]})
    assert detect_domain(df_ambiguo) == Domain.UNKNOWN


def test_carpeta_con_wear_csv_es_desgaste(tmp_path):
    """Una carpeta que contiene archivos *_wear.csv se reconoce como
    desgaste de herramienta, sin necesidad de inspeccionar columnas."""
    exp_dir = tmp_path / "c1"
    exp_dir.mkdir()
    (exp_dir / "c1_wear.csv").write_text("cut,flute_1,flute_2,flute_3\n1,0.05,0.04,0.05\n")
    assert detect_domain(str(tmp_path)) == Domain.TOOL_WEAR


def test_dict_de_experimentos_es_desgaste():
    """La estructura cruda {experimento: {'signals':..., 'wear':...}} se
    reconoce sin necesidad de tocar disco."""
    fuente = {"c1": {"signals": "/alguna/ruta", "wear": "/otra/ruta.csv"}}
    assert detect_domain(fuente) == Domain.TOOL_WEAR
