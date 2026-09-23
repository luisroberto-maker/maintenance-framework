"""
tests/test_errors.py — Sistema de errores y estados estructurados
(core/errors.py) y los métodos run_safe()/predict_only_safe() del
orquestador.
"""
import pandas as pd
import pytest

from core.errors import (
    Status, ErrorCode, PLCAgentError,
    InsufficientDataError, UnknownSchemaError, ModelNotFoundError,
    UnknownDomainError, error_to_response, success_response,
)


# ─── Las excepciones estructuradas ────────────────────────────────────────────

def test_cada_excepcion_tiene_su_codigo_correcto():
    assert InsufficientDataError("x").code == ErrorCode.INSUFFICIENT_DATA
    assert UnknownSchemaError("x").code == ErrorCode.UNKNOWN_SCHEMA
    assert ModelNotFoundError("x").code == ErrorCode.MODEL_NOT_FOUND
    assert UnknownDomainError("x").code == ErrorCode.UNKNOWN_DOMAIN


def test_to_dict_tiene_el_formato_esperado():
    exc = InsufficientDataError("no hay suficientes eventos", details={"n_rows": 5})
    d = exc.to_dict()
    assert d["status"] == "ERROR"
    assert d["code"] == "INSUFFICIENT_DATA"
    assert d["message"] == "no hay suficientes eventos"
    assert d["details"] == {"n_rows": 5}


def test_modelnotfounderror_sigue_siendo_capturable_como_filenotfounderror():
    """Compatibilidad hacia atrás: código existente que hace
    `except FileNotFoundError` no debe dejar de funcionar."""
    with pytest.raises(FileNotFoundError):
        raise ModelNotFoundError("no encontrado")


def test_unknowndomainerror_sigue_siendo_capturable_como_valueerror():
    with pytest.raises(ValueError):
        raise UnknownDomainError("dominio desconocido")


# ─── error_to_response() ──────────────────────────────────────────────────────

def test_error_to_response_con_excepcion_estructurada():
    exc = ModelNotFoundError("no encontrado", details={"model_path": "x.pkl"})
    r = error_to_response(exc)
    assert r["status"] == "ERROR"
    assert r["code"] == "MODEL_NOT_FOUND"
    assert r["details"]["model_path"] == "x.pkl"


def test_error_to_response_con_pydantic_validation_error():
    from pydantic import BaseModel, Field, ValidationError

    class Modelo(BaseModel):
        x: int = Field(gt=0)

    try:
        Modelo(x=-5)
    except ValidationError as e:
        r = error_to_response(e)
        assert r["status"] == "ERROR"
        assert r["code"] == "VALIDATION_ERROR"
        assert "errors" in r["details"]
        assert len(r["details"]["errors"]) == 1


def test_error_to_response_con_excepcion_no_anticipada():
    """Cualquier excepción no prevista debe convertirse igual al mismo
    formato — nunca debe dejar escapar un traceback crudo."""
    r = error_to_response(RuntimeError("algo salió mal de forma inesperada"))
    assert r["status"] == "ERROR"
    assert r["code"] == "INTERNAL_ERROR"
    assert "algo salió mal" in r["message"]
    assert r["details"]["exception_type"] == "RuntimeError"


# ─── success_response() ───────────────────────────────────────────────────────

def test_success_response_sin_advertencias_es_success():
    r = success_response("reporte.html")
    assert r["status"] == "SUCCESS"
    assert r["warnings"] == []


def test_success_response_con_advertencias_es_warning():
    r = success_response("reporte.html", warnings=["algo a tener en cuenta"])
    assert r["status"] == "WARNING"
    assert r["warnings"] == ["algo a tener en cuenta"]


# ─── run_safe() / predict_only_safe() — integración con el orquestador ──────

def test_run_safe_nunca_lanza_excepcion_con_dominio_invalido(agent_config):
    """Si no se puede determinar el dominio ni hay force_domain, run()
    lanzaría UnknownDomainError — run_safe() debe atraparla y devolver
    el dict estructurado en vez de propagarla."""
    from orchestrator import MultiVerticalAgentSystem

    system = MultiVerticalAgentSystem(config=agent_config)
    df_ambiguo = pd.DataFrame({"col_x": [1, 2, 3], "col_y": ["a", "b", "c"]})

    resultado = system.run_safe(df_ambiguo, client_id="test_ambiguo", verbose=False)
    # UNKNOWN cae por defecto a intentar como PLC (comportamiento de run()),
    # así que esto en realidad no dispara UnknownDomainError directamente,
    # pero sí debe fallar de forma controlada (esquema no reconocible) y
    # devolver un dict, nunca lanzar.
    assert isinstance(resultado, dict)
    assert resultado["status"] in ("ERROR", "SUCCESS", "WARNING")


def test_predict_only_safe_sin_modelo_devuelve_model_not_found(agent_config, df_plc_events):
    from orchestrator import MultiVerticalAgentSystem

    system = MultiVerticalAgentSystem(config=agent_config)
    resultado = system.predict_only_safe(
        source=df_plc_events,
        client_id="sin_modelo_previo",
        domain="plc_failure",
        verbose=False,
    )
    assert resultado["status"] == "ERROR"
    assert resultado["code"] == "MODEL_NOT_FOUND"
    assert "model_path" in resultado["details"]


def test_run_safe_con_datos_validos_es_success_o_warning(df_plc_events, agent_config, mock_plc_embeddings):
    from orchestrator import MultiVerticalAgentSystem

    system = MultiVerticalAgentSystem(config=agent_config)
    resultado = system.run_safe(df_plc_events, client_id="test_valido", verbose=False)

    assert resultado["status"] in ("SUCCESS", "WARNING")
    assert "report_path" in resultado
    assert isinstance(resultado["warnings"], list)


def test_run_safe_con_datos_insuficientes_no_lanza_y_da_codigo_correcto(agent_config):
    """Con muy pocos datos, el Agente 2 decide HEURISTIC/can_train limitado
    — pero incluso en el peor caso (columnas irreconocibles), run_safe()
    debe devolver un dict con código de error, nunca una excepción cruda."""
    from orchestrator import MultiVerticalAgentSystem

    system = MultiVerticalAgentSystem(config=agent_config)
    df_vacio_de_schema = pd.DataFrame({"a": [1], "b": [2]})

    resultado = system.run_safe(df_vacio_de_schema, client_id="test_insuficiente", verbose=False)
    assert isinstance(resultado, dict)
    assert resultado["status"] == "ERROR"
    assert resultado["code"] in ("UNKNOWN_SCHEMA", "INSUFFICIENT_DATA")
