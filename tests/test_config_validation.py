"""
tests/test_config_validation.py — Validación de AgentConfig con Pydantic.

AgentConfig dejó de ser un @dataclass simple para convertirse en un
pydantic.BaseModel — estas pruebas confirman que:
  1. La construcción normal (con y sin kwargs) sigue funcionando igual
     que antes, sin romper ningún llamador existente.
  2. Valores inválidos fallan AL CONSTRUIR, con un mensaje claro que
     identifica el campo y la razón — no minutos después, a mitad del
     entrenamiento.
"""
import pytest
from pydantic import ValidationError

from core.config import AgentConfig


# ─── Construcción normal — no debe cambiar de comportamiento ────────────────

def test_construccion_por_defecto():
    cfg = AgentConfig()
    assert cfg.umbral_critico == 30
    assert cfg.ventana_dias == 7
    assert cfg.split_ratio == 0.80


def test_construccion_con_kwargs():
    cfg = AgentConfig(umbral_critico=20, output_dir="resultados", top_n_ranking=25)
    assert cfg.umbral_critico == 20
    assert cfg.output_dir == "resultados"
    assert cfg.top_n_ranking == 25


def test_reconstruccion_a_partir_de_otra_instancia():
    """Patrón usado en agent3_executor.py: construir un AgentConfig nuevo
    copiando campos de uno existente vía kwargs explícitos."""
    original = AgentConfig(umbral_critico=15, ventana_dias=10)
    copia = AgentConfig(
        umbral_critico=original.umbral_critico,
        ventana_dias=original.ventana_dias,
    )
    assert copia.umbral_critico == 15
    assert copia.ventana_dias == 10


# ─── Valores inválidos — deben fallar al construir ───────────────────────────

@pytest.mark.parametrize("kwargs", [
    {"umbral_critico": -5},
    {"umbral_critico": 0},
    {"ventana_dias": 0},
    {"ventana_dias": -1},
    {"default_n_eventos": 0},
    {"split_ratio": 0.0},
    {"split_ratio": 1.0},
    {"split_ratio": 1.5},
    {"split_ratio": -0.1},
    {"n_pca_components": 0},
    {"n_pca_components": -3},
    {"min_eventos_modelo_local": 0},
    {"min_eventos_entrenamiento": 0},
    {"umbral_incertidumbre": 0.0},
    {"umbral_incertidumbre": -0.1},
    {"top_n_ranking": 0},
    {"output_dir": ""},
    {"output_dir": "   "},
    {"model_dir": ""},
    {"report_dir": ""},
    {"embedding_model": ""},
    {"idioma_reporte": "fr"},
    {"idioma_reporte": "ES"},  # sensible a mayúsculas — solo "es"/"en" literal
])
def test_valores_invalidos_fallan_al_construir(kwargs):
    with pytest.raises(ValidationError):
        AgentConfig(**kwargs)


def test_campo_desconocido_falla_por_extra_forbid():
    """Un typo en el nombre de un parámetro (ej. 'umbral_critico_mal' en
    vez de 'umbral_critico') debe fallar en vez de ignorarse en silencio."""
    with pytest.raises(ValidationError):
        AgentConfig(umbral_critico_mal=5)


def test_mensaje_de_error_identifica_el_campo_y_la_razon():
    """El mensaje debe ser accionable: qué campo, qué se recibió, por qué
    está mal — no un traceback genérico."""
    with pytest.raises(ValidationError) as exc_info:
        AgentConfig(umbral_critico=-5)

    mensaje = str(exc_info.value)
    assert "umbral_critico" in mensaje
    assert "-5" in mensaje


def test_multiples_errores_se_reportan_juntos():
    """Si varios campos son inválidos a la vez, Pydantic los reporta
    todos en un solo error — no solo el primero que encuentra."""
    with pytest.raises(ValidationError) as exc_info:
        AgentConfig(umbral_critico=-5, split_ratio=2.0)

    assert len(exc_info.value.errors()) == 2


# ─── Valores límite — deben ser aceptados (no son inválidos) ────────────────

def test_valores_limite_validos_no_fallan():
    # split_ratio muy cerca de los límites, pero dentro del rango abierto (0, 1)
    AgentConfig(split_ratio=0.01)
    AgentConfig(split_ratio=0.99)
    # random_state=0 es válido (ge=0, no gt=0)
    AgentConfig(random_state=0)
    # valores mínimos positivos
    AgentConfig(umbral_critico=1, ventana_dias=1, n_pca_components=1, top_n_ranking=1)
