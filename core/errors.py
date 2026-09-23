"""
core/errors.py — Errores y estados estructurados para consumidores
externos (como una interfaz web).

Antes de este módulo, cuando algo fallaba, el framework dejaba escapar
excepciones nativas de Python (FileNotFoundError, ValueError, o incluso
pydantic.ValidationError) directamente hacia quien lo llamaba. Eso
obliga a cualquier interfaz externa a conocer y capturar tipos de
excepción internos de Python, en vez de recibir algo que pueda mostrarle
al usuario final sin traducir.

Este módulo define:
  - Status / ErrorCode: el vocabulario de estados que puede ver una
    interfaz externa, sin acoplarse a excepciones de Python.
  - PLCAgentError y sus subclases: excepciones específicas del dominio,
    cada una con su ErrorCode ya asignado.
  - error_to_response(): convierte CUALQUIER excepción (una de las
    nuestras, una de pydantic, o cualquier otra no anticipada) al mismo
    formato de diccionario serializable a JSON.

Compatibilidad hacia atrás: las excepciones específicas heredan también
de la excepción nativa equivalente (ej. ModelNotFoundError hereda de
FileNotFoundError), así que código existente que ya capturaba
FileNotFoundError sigue funcionando exactamente igual — esto es
puramente aditivo, no rompe nada.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class Status(str, Enum):
    """Estado de alto nivel de una operación — lo primero que debería
    mirar una interfaz externa antes de decidir qué hacer con el resto
    de la respuesta."""
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ErrorCode(str, Enum):
    """Códigos específicos cuando status == ERROR. Estables entre
    versiones — si agregas uno nuevo, no reutilices ni renombres uno
    existente, ya que una interfaz externa puede estar comparando contra
    estos valores literales."""
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNKNOWN_SCHEMA = "UNKNOWN_SCHEMA"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_DOMAIN = "UNKNOWN_DOMAIN"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class PLCAgentError(Exception):
    """
    Base de todas las excepciones estructuradas del framework.

    No la lances directamente — usa una de las subclases de abajo, que
    ya traen su ErrorCode correcto asignado.
    """
    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "status": Status.ERROR.value,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class InsufficientDataError(PLCAgentError):
    """No hay suficientes eventos/pasadas para entrenar o para calcular
    una predicción confiable — un problema de CANTIDAD de datos."""
    code = ErrorCode.INSUFFICIENT_DATA


class UnknownSchemaError(PLCAgentError):
    """Los datos no tienen las columnas esperadas y no se pudieron
    inferir (ni con column_map) — un problema de ESTRUCTURA de datos,
    no de cantidad."""
    code = ErrorCode.UNKNOWN_SCHEMA


class ModelNotFoundError(PLCAgentError, FileNotFoundError):
    """No existe un modelo entrenado en la ruta esperada o indicada.
    Hereda también de FileNotFoundError por compatibilidad — código que
    ya hacía `except FileNotFoundError` lo sigue capturando igual."""
    code = ErrorCode.MODEL_NOT_FOUND


class ConfigValidationError(PLCAgentError):
    """Un valor de configuración (AgentConfig) es inválido. Para errores
    que vienen directamente de pydantic, usa error_to_response() en vez
    de esta clase — ver más abajo."""
    code = ErrorCode.VALIDATION_ERROR


class UnknownDomainError(PLCAgentError, ValueError):
    """No se pudo determinar si los datos son de la vertical PLC o de
    desgaste de herramienta, y no se especificó domain/force_domain
    explícitamente. Hereda también de ValueError por compatibilidad."""
    code = ErrorCode.UNKNOWN_DOMAIN


def error_to_response(exc: Exception) -> dict:
    """
    Convierte CUALQUIER excepción al mismo formato estructurado, para
    que una interfaz externa solo necesite un punto de conversión, sin
    importar si el error vino de este framework, de pydantic, o de algo
    completamente inesperado.

    Formato de salida (siempre el mismo shape):
        {
            "status": "ERROR",
            "code": "INSUFFICIENT_DATA" | "UNKNOWN_SCHEMA" | ... ,
            "message": "texto legible para mostrar al usuario",
            "details": {...}  # contexto adicional, varía según el error
        }
    """
    if isinstance(exc, PLCAgentError):
        return exc.to_dict()

    # pydantic.ValidationError — se detecta por nombre de clase para no
    # forzar que pydantic sea una dependencia obligatoria de este módulo
    # en particular, aunque en la práctica ya lo es para el resto del
    # framework.
    if type(exc).__name__ == "ValidationError" and hasattr(exc, "errors"):
        try:
            errores = exc.errors()
        except Exception:
            errores = []
        return {
            "status": Status.ERROR.value,
            "code": ErrorCode.VALIDATION_ERROR.value,
            "message": "Configuración inválida — revisa los parámetros.",
            "details": {"errors": errores},
        }

    # Cualquier otra excepción no anticipada — nunca debería llegar aquí
    # en operación normal, pero si pasa, igual se entrega en el mismo
    # formato en vez de dejar escapar un traceback crudo.
    return {
        "status": Status.ERROR.value,
        "code": ErrorCode.INTERNAL_ERROR.value,
        "message": str(exc) or f"Error interno inesperado ({type(exc).__name__}).",
        "details": {"exception_type": type(exc).__name__},
    }


def success_response(report_path: str, warnings: Optional[list] = None,
                     **extra) -> dict:
    """Construye la respuesta de éxito — SUCCESS si no hay advertencias,
    WARNING si las hay (la operación sí completó, pero con algo que vale
    la pena que la interfaz externa le muestre al usuario)."""
    warnings = warnings or []
    respuesta = {
        "status": Status.WARNING.value if warnings else Status.SUCCESS.value,
        "code": None,
        "message": "Completado con advertencias." if warnings else "Completado correctamente.",
        "report_path": report_path,
        "warnings": warnings,
    }
    respuesta.update(extra)
    return respuesta
