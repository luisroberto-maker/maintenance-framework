"""
core/logging_config.py — Configuración centralizada de logging.

Antes de este módulo, el código mezclaba print() con logger.info()/
logger.warning() sin ningún criterio, repartido en 8 archivos distintos.
Esto tenía dos problemas reales:

  1. No había forma de controlar cuánto se muestra sin editar el código
     — ni silenciar el progreso en producción, ni mandar los logs a un
     archivo además de consola.
  2. Si en el futuro este framework se usa dentro de otra aplicación
     (un servicio web, un notebook, un cron job), esa aplicación no
     tiene ningún control sobre la verbosidad de este paquete —
     print() no respeta ninguna configuración externa.

Con logging estándar de Python, ambos problemas se resuelven gratis:
quien importe este paquete puede controlar su verbosidad
(`logging.getLogger("plc_agent").setLevel(...)`), redirigirla a un
archivo, o silenciarla por completo — sin tocar una sola línea de
core/ ni agents/.

Uso típico (ya integrado en orchestrator.py, no hace falta llamarlo
manualmente salvo que quieras cambiar el comportamiento por defecto):

    from core.logging_config import setup_logging
    setup_logging(level="INFO")                    # consola limpia (por defecto)
    setup_logging(level="DEBUG", log_file="run.log")  # además, a archivo
    setup_logging(level="WARNING")                  # solo advertencias y errores

También se puede controlar sin tocar código, vía variable de entorno:

    PLC_AGENT_LOG_LEVEL=DEBUG python orchestrator.py datos.xlsx
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


class _CleanConsoleFormatter(logging.Formatter):
    """
    Formato de consola pensado para uso interactivo (CLI):
      - Mensajes INFO se muestran tal cual, sin timestamp ni prefijo —
        exactamente como se veían los print() que reemplaza.
      - WARNING/ERROR llevan un ícono visual, igual que antes.
      - DEBUG lleva el nombre del módulo, útil solo cuando se pide
        explícitamente más detalle.
    """

    _ICONOS = {
        logging.WARNING: "⚠",
        logging.ERROR: "✗",
        logging.CRITICAL: "✗",
    }

    def format(self, record: logging.LogRecord) -> str:
        mensaje = record.getMessage()
        if record.levelno == logging.DEBUG:
            return f"[{record.name}] {mensaje}"
        icono = self._ICONOS.get(record.levelno)
        return f"{icono} {mensaje}" if icono else mensaje


_VERBOSE_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose_console: bool = False,
) -> logging.Logger:
    """
    Configura el logging para todo el paquete. Se llama una sola vez, al
    arrancar el programa (ya lo hace orchestrator.py automáticamente).

    Parameters
    ----------
    level          : nivel mínimo a mostrar ("DEBUG", "INFO", "WARNING",
                     "ERROR"). Si no se especifica, respeta la variable
                     de entorno PLC_AGENT_LOG_LEVEL; si tampoco existe,
                     usa "INFO".
    log_file       : ruta opcional — si se da, además de consola, todo
                     se escribe ahí con timestamp y nivel completos
                     (útil para diagnosticar una corrida después).
    verbose_console: si True, la consola también usa el formato completo
                     con timestamp (útil para debugging), en vez del
                     formato limpio por defecto.

    Returns
    -------
    El logger raíz ya configurado.
    """
    nivel_efectivo = os.environ.get("PLC_AGENT_LOG_LEVEL", level).upper()

    root = logging.getLogger()
    root.setLevel(nivel_efectivo)
    root.handlers.clear()  # evita duplicar handlers si se llama más de una vez

    consola = logging.StreamHandler(sys.stdout)
    if verbose_console:
        consola.setFormatter(logging.Formatter(_VERBOSE_FILE_FORMAT, datefmt="%H:%M:%S"))
    else:
        consola.setFormatter(_CleanConsoleFormatter())
    root.addHandler(consola)

    if log_file:
        archivo = logging.FileHandler(log_file, encoding="utf-8")
        archivo.setFormatter(logging.Formatter(_VERBOSE_FILE_FORMAT))
        root.addHandler(archivo)

    return root


def get_logger(name: str) -> logging.Logger:
    """Atajo estándar — equivalente a logging.getLogger(name), para
    mantener un solo punto de import en todo el paquete."""
    return logging.getLogger(name)
