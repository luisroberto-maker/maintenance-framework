"""
core/legacy_plc_predictor.py — LegacyPLCPredictor.

Envoltorio de BasePredictor sobre el pipeline ORIGINAL del repositorio
luisroberto-maker/PLC-failure-prediction-pipeline, replicado línea por línea
(mismo MAPEO_FALLAS, mismo preprocesamiento de texto, mismas features,
mismo LabelEncoder) para poder usar el modelo .pkl ya entrenado sin alterar
absolutamente nada de su comportamiento — bugs conocidos incluidos.

Diferencias respecto a AdaptiveFailurePredictor (core/predictor.py):
  - Usa LabelEncoder (se reajusta en cada llamada) en vez de SafeEncoder con
    vocabulario fijo. Esto significa que categoria_cod puede NO ser
    consistente entre la corrida de entrenamiento original y una corrida de
    inferencia posterior si el conjunto de categorías presentes difiere.
    Esto es EXACTAMENTE el comportamiento del repositorio original — no se
    corrigió a propósito, según lo solicitado.
  - El PCA de embeddings se reajusta (fit) en cada llamada a predict_all(),
    sobre los datos que se le pasen en ese momento — no reutiliza el PCA
    del entrenamiento original. También es el comportamiento original.
  - Tiene 17 features + PCA (vs 21 + PCA en AdaptiveFailurePredictor): no
    incluye panel_cod, ratio_reciente_historico, mediana_tiempo_entre_fallas
    ni aceleracion_fallas.
  - No tiene niveles LOCAL/GLOBAL/HEURÍSTICO ni incertidumbre por dispersión
    de árboles — es un único modelo global aplicado a todos los componentes.

Este módulo NO modifica ninguna de estas características. Las columnas
adicionales de salida (PANEL, NIVEL_CONFIANZA) son puramente informativas
para el reporte y no participan en el cálculo de PROBABILIDAD_FALLA.
"""
from __future__ import annotations

import logging
import re
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.base_predictor import BasePredictor, ReportSchema

logger = logging.getLogger(__name__)

_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


# ─── MAPEO_FALLAS — copiado verbatim de src/preprocessing.py del repo original ─

MAPEO_FALLAS = {
    'PLC Connect Error(-1) P2:L5:34-P1:L1:11': 'COMUNICACIÓN',
    'Human Detect LC (Entrance)': 'SEGURIDAD',
    'C3 Heavy Error': 'HARDWARE_ROBOT',
    'R/B-3 FAULT': 'HARDWARE_ROBOT',
    'C1 Heavy Error': 'HARDWARE_ROBOT',
    'R/B-1 FAULT': 'HARDWARE_ROBOT',
    'Safety Door Switch (Entrance)': 'SEGURIDAD',
    'SafetyHandle ENT-R(Detail)': 'SEGURIDAD',
    'Base Robot PROTECTIVE STOP': 'HARDWARE_ROBOT',
    'OP2 safety door SW batch': 'SEGURIDAD',
    'Pinching LS': 'SEGURIDAD',
    'Pinching LS Left (Detail)': 'SEGURIDAD',
    'R/B-2 FAULT': 'HARDWARE_ROBOT',
    'PLC Connect Error(-3) P2:L5:34-P1:L1:11': 'COMUNICACIÓN',
    '1st Alarm (Main)': 'CONTROL_LOGICA',
    'Job Not Complete Alarm Home2': 'CONTROL_LOGICA',
    'RB1 Job Complete Alarm Home2': 'CONTROL_LOGICA',
    'OP1 safety door SW batch': 'SEGURIDAD',
    'Primer Robot PROTECTIVE STOP': 'HARDWARE_ROBOT',
    'C5 Heavy Error': 'HARDWARE_ROBOT',
    'DCL TANK PRESSURE LOW': 'SUMINISTROS',
    'R/B-5 FAULT': 'HARDWARE_ROBOT',
    'OP3 emergency stop batch': 'SEGURIDAD',
    'Clear Robot PROTECTIVE STOP': 'HARDWARE_ROBOT',
    'OP3 safety door SW batch': 'SEGURIDAD',
    'Emergency Stop (Entrance)': 'SEGURIDAD',
    'E-Stop Entrance Right (Detail)': 'SEGURIDAD',
    'Human Detect LC (Exit)': 'SEGURIDAD',
    'R/B-4 FAULT': 'HARDWARE_ROBOT',
    'OP1 emergency stop batch': 'SEGURIDAD',
    '1ST Teach Pendant Emergency Stop': 'SEGURIDAD',
    'RB5-8 E-Stop (Detail)': 'SEGURIDAD',
    'Emergency Stop': 'SEGURIDAD',
    'RB1-4 E-Stop (Detail)': 'SEGURIDAD',
    'COMMUNICATION ERROR(MAIN)': 'COMUNICACIÓN',
    'P1-L3 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'R/B-6 FAULT': 'HARDWARE_ROBOT',
    'C4 Heavy Error': 'HARDWARE_ROBOT',
    'Auto on Job Completion Alarm': 'CONTROL_LOGICA',
    'OP1 Transfer fault': 'PROCESO_PINTURA',
    'Cartridge OFF work goes in': 'PROCESO_PINTURA',
    'Painter OFF SS work goes in': 'PROCESO_PINTURA',
    'CLEAR EXHAUST FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH WASHER PUMP - PANEL MODE OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH WASHER PUMP - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH WASHER PUMP - DSW FAULT': 'SISTEMA_ELECTRICO',
    'C2 Heavy Error': 'HARDWARE_ROBOT',
    'CLEAR ASH WASHER PUMP - PANEL MODE OFF': 'SISTEMA_ELECTRICO',
    'AntiChip Robot PROTECTIVE STOP': 'HARDWARE_ROBOT',
    'CLEAR ASH WASHER PUMP - DSW FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH WASHER PUMP - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'OP2 emergency stop batch': 'SEGURIDAD',
    'LSO-111 FAULT': 'PROCESO_PINTURA',
    'LSO-112 FAULT': 'PROCESO_PINTURA',
    'LSO-113 FAULT': 'PROCESO_PINTURA',
    'BASE ASH PREHEATER - #1 PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH PREHEATER - #1 PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH PREHEATER - GAS PRESSURE LOW': 'SUMINISTROS',
    'BASE PREHEAT HAB BURNER - PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH PREHEATER - #2 PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    '2st Alarm (Main)': 'CONTROL_LOGICA',
    'CYCLE TIME-OVER FAULT': 'CONTROL_LOGICA',
    'OP1 cycle over batch': 'CONTROL_LOGICA',
    'OP2 Transfer fault': 'PROCESO_PINTURA',
    'No Auto mode work goes in': 'CONTROL_LOGICA',
    'START LS ON ERROR': 'CONTROL_LOGICA',
    'OP2 Thermal trip': 'SISTEMA_ELECTRICO',
    'OP1 air pressure decrease batch': 'SUMINISTROS',
    'OP1 air pressure fault': 'SUMINISTROS',
    'BASE ASH WASHER TANK - FILTER BLOCKAGE': 'SUMINISTROS',
    'CLEAR ASH WASHER TANK - L LEVEL': 'SUMINISTROS',
    'CLEAR ASH - SUPPLY AIR HUM. FAULT': 'SUMINISTROS',
    '1ST Auto Cleaning Start Alarm': 'PROCESO_PINTURA',
    'OP3 Transfer fault': 'PROCESO_PINTURA',
    'RB2 Job Complete Alarm Home2': 'CONTROL_LOGICA',
    'RB3 Job Complete Alarm Home2': 'CONTROL_LOGICA',
    'RB4 Job Complete Alarm Home2': 'CONTROL_LOGICA',
    'RB5 Job Complete Alarm Home2': 'CONTROL_LOGICA',
    'RB6 Job Complete Alarm Home2': 'CONTROL_LOGICA',
    'BCR Reading Rate Low Alarm': 'COMUNICACIÓN',
    'PRIMER ASH WASHER TANK - FILTER BLOCKAGE': 'SUMINISTROS',
    'BASE ASH - SUPPLY AIR HUM. FAULT': 'SUMINISTROS',
    'OP1 light curtain batch': 'SEGURIDAD',
    'PBIS Fault': 'PROCESO_PINTURA',
    'Manual Write Alarm': 'CONTROL_LOGICA',
    'OP2 light curtain batch': 'SEGURIDAD',
    'OP3 light curtain batch': 'SEGURIDAD',
    'R/B-2 ALARM': 'HARDWARE_ROBOT',
    'Color Date Seach Fault': 'PROCESO_PINTURA',
    'BASE PREHEAT - CAB TEMP. FAULT': 'SUMINISTROS',
    'Cleaning mode work goes in': 'PROCESO_PINTURA',
    'Docking Error': 'CONTROL_LOGICA',
    'RB1 Docking Wait Fault': 'CONTROL_LOGICA',
    'PBIS NO Spec': 'PROCESO_PINTURA',
    'Pinching LS Right (Detail)': 'SEGURIDAD',
    'PCS FAULT(MAIN)': 'CONTROL_LOGICA',
    'PCS RMT RackI/O Fault': 'SISTEMA_ELECTRICO',
    'PCS TOTAL FAULT': 'CONTROL_LOGICA',
    'Safty PLC(PROCESS)': 'SEGURIDAD',
    'PCS STOP': 'CONTROL_LOGICA',
    'S/N Communication Fault': 'COMUNICACIÓN',
    'BCR Read Fault': 'COMUNICACIÓN',
    'PRIMER ASH - SUPPLY AIR HUM. FAULT': 'SUMINISTROS',
    'PRIMER ASH - SUPPLY AIR TEMP. FAULT': 'SUMINISTROS',
    'Color Confirm Error': 'PROCESO_PINTURA',
    'RB1 Cartridge Color Comparison Fault': 'PROCESO_PINTURA',
    'RB2 Cartridge Color Comparison Fault': 'PROCESO_PINTURA',
    'BASE ASH - SUPPLY AIR TEMP. FAULT': 'SUMINISTROS',
    'BOOTH SHUTDOWN': 'CONTROL_LOGICA',
    'CLEAR ASH C/P - EMERGENCY STOP': 'SEGURIDAD',
    'OP1 MCCB off batch': 'SISTEMA_ELECTRICO',
    'AVR1 exchange time': 'SISTEMA_ELECTRICO',
    'AVR1 fault interception': 'SISTEMA_ELECTRICO',
    'AVR1 voltage decrease': 'SISTEMA_ELECTRICO',
    'AVR2 exchange time': 'SISTEMA_ELECTRICO',
    'AVR2 fault interception': 'SISTEMA_ELECTRICO',
    'AVR2 voltage decrease': 'SISTEMA_ELECTRICO',
    'Process control panel and power panel fault': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - EMERGENCY STOP': 'SEGURIDAD',
    'PLC Connect Error(-1) P2:L5:13': 'COMUNICACIÓN',
    'PRIMER #1 EXHAUST FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER #2 EXHAUST FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER PREHEAT CAB CIRCULATION FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER PREHEAT HAB BURNER - GAS PRESSURE HIGH': 'SUMINISTROS',
    'PRIMER PREHEAT HAB BURNER - GAS PRESSURE LOW': 'SUMINISTROS',
    'PRIMER PREHEAT HAB BURNER BLOWER - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER PREHEAT HAB CIRCULATION FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER PREHEAT HAB SUPPLY AIR - TEMP. HIGH': 'SUMINISTROS',
    'PRIMER #1 EXHAUST FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'PRIMER #2 EXHAUST FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'PRIMER EXHAUST & PREHEAT C/P - BURNER POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER EXHAUST & PREHEAT C/P - CONTROL POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER EXHAUST & PREHEAT C/P - DC POWER (AVR) FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER EXHAUST & PREHEAT C/P - DC POWER (AVR) LFE': 'SISTEMA_ELECTRICO',
    'PRIMER EXHAUST & PREHEAT C/P - DC POWER (AVR) LOW': 'SISTEMA_ELECTRICO',
    'PRIMER EXHAUST & PREHEAT C/P - EMERGENCY STOP': 'SEGURIDAD',
    'PRIMER EXHAUST & PREHEAT C/P - INSTRUMENT POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER EXHAUST & PREHEAT C/P FRMT FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'PRIMER PREHEAT CAB CIRCULATION FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'PRIMER PREHEAT HAB BURNER - CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER PREHEAT HAB CIRCULATION FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'PRIMER BOOTH DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER #1 EXHAUST DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER #2 EXHAUST DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER PREHEAT HAB BURNER - CONTROL MOTOR FAULT': 'SISTEMA_ELECTRICO',
    'SAFETY PLC FAULT PL': 'SEGURIDAD',
    'FIRE SIGNAL': 'SEGURIDAD',
    'CLEAR EXHAUST C/P - CONTROL POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR EXHAUST C/P - DC POWER (AVR) FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR EXHAUST C/P - DC POWER (AVR) LFE': 'SISTEMA_ELECTRICO',
    'CLEAR EXHAUST C/P - DC POWER (AVR) LOW': 'SISTEMA_ELECTRICO',
    'CLEAR EXHAUST C/P - EMERGENCY STOP': 'SEGURIDAD',
    'CLEAR EXHAUST C/P - INSTRUMENT POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR EXHAUST C/P FRMT FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'CLEAR EXHAUST FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'CLEAR EXHAUST DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'LC E-Stop': 'SEGURIDAD',
    'C6 Heavy Error': 'HARDWARE_ROBOT',
    'PRIMER EXHAUST & PREHEAT C/P - TEH15311 CONTROLLER FAULT': 'SISTEMA_ELECTRICO',
    'DCL tank LOW': 'SUMINISTROS',
    'R/B-1 ALARM': 'HARDWARE_ROBOT',
    'PRIMER ASH REHEATER - PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH REHEATER - PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'R/B-4 ALARM': 'HARDWARE_ROBOT',
    'Paint OFF sw work goes in': 'PROCESO_PINTURA',
    'PLC Connect Error(-4) P2:L5:13': 'COMUNICACIÓN',
    'CLEAR ASH REHEATER - PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'OP2 cycle over': 'CONTROL_LOGICA',
    'PBIS Read Time Over': 'PROCESO_PINTURA',
    'BASE #1 EXHAUST FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'BASE #1 EXHAUST FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'BASE #2 EXHAUST FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'BASE #2 EXHAUST FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'BASE EXHAUST & PREHEAT C/P - BURNER POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE EXHAUST & PREHEAT C/P - CONTROL POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE EXHAUST & PREHEAT C/P - DC POWER (AVR) FAULT': 'SISTEMA_ELECTRICO',
    'BASE EXHAUST & PREHEAT C/P - DC POWER (AVR) LFE': 'SISTEMA_ELECTRICO',
    'BASE EXHAUST & PREHEAT C/P - DC POWER (AVR) LOW': 'SISTEMA_ELECTRICO',
    'BASE EXHAUST & PREHEAT C/P - EMERGENCY STOP': 'SEGURIDAD',
    'BASE EXHAUST & PREHEAT C/P - INSTRUMENT POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE EXHAUST & PREHEAT C/P FRMT FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'BASE PREHEAT CAB CIRCULATION FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT CAB CIRCULATION FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'BASE PREHEAT HAB BURNER - CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT HAB BURNER - GAS PRESSURE HIGH': 'SUMINISTROS',
    'BASE PREHEAT HAB BURNER - GAS PRESSURE LOW': 'SUMINISTROS',
    'BASE PREHEAT HAB BURNER BLOWER - DSW FAULT': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT HAB CIRCULATION FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT HAB CIRCULATION FAN - FL-REMOTE COMMUNICATION FAULT': 'COMUNICACIÓN',
    'BASE PREHEAT HAB SUPPLY AIR - TEMP. HIGH': 'SUMINISTROS',
    'BASE BOOTH DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'BASE #1 EXHAUST DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'BASE #2 EXHAUST DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT HAB BURNER - CONTROL MOTOR FAULT': 'SISTEMA_ELECTRICO',
    'BOOTH MASTER PANEL - EMERGENCY STOP': 'SEGURIDAD',
    'OP2 DISCONNECT OFF': 'SISTEMA_ELECTRICO',
    'EARTHQUAKE': 'SEGURIDAD',
    'FIRE': 'SEGURIDAD',
    'BASE ASH C/P - ETHERNET/IP COMMUNICATION FAULT': 'COMUNICACIÓN',
    'CLEAR ASH C/P - ETHERNET/IP COMMUNICATION FAULT': 'COMUNICACIÓN',
    'PRIMER ASH C/P - ETHERNET/IP COMMUNICATION FAULT': 'COMUNICACIÓN',
    'PLC Connect Error(-1) P2:L5:34-P1:L1:7': 'COMUNICACIÓN',
    'PLC Connect Error(-1) P2:L5:34-P1:L1:9': 'COMUNICACIÓN',
    'OP2 air pressure decrease batch': 'SUMINISTROS',
    'OP2 air pressure fault': 'SUMINISTROS',
    'DISCONNECT OFF': 'SISTEMA_ELECTRICO',
    'OP1 Thermal trip': 'SISTEMA_ELECTRICO',
    'BASE EXHAUST & PREHEAT C/P - TEH15311 CONTROLLER FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - EMERGENCY STOP': 'SEGURIDAD',
    'BASE ASH PREHEATER - GAS PRESSURE LOW': 'SUMINISTROS',
    'C1 Job Delay': 'CONTROL_LOGICA',
    'Cartridge Job Dealy': 'CONTROL_LOGICA',
    'OP3 Thermal trip': 'SISTEMA_ELECTRICO',
    'PG Seach Fault': 'PROCESO_PINTURA',
    'RB2 Docking Wait Fault': 'CONTROL_LOGICA',
    'C2 Job Delay': 'CONTROL_LOGICA',
    'C3 Job Delay': 'CONTROL_LOGICA',
    'C4 Job Delay': 'CONTROL_LOGICA',
    'Job Not Complete Alarm': 'CONTROL_LOGICA',
    'RB1 Job Not Complete Alarm': 'CONTROL_LOGICA',
    'RB2 Job Not Complete Alarm': 'CONTROL_LOGICA',
    '2nd Start ON Fault': 'CONTROL_LOGICA',
    'RB1 2nd Start ON Fault': 'CONTROL_LOGICA',
    'RB2 2nd Start ON Fault': 'CONTROL_LOGICA',
    'RB5 2nd Start ON Fault': 'CONTROL_LOGICA',
    'RB5 Docking Wait Fault': 'CONTROL_LOGICA',
    'RB6 Docking Wait Fault': 'CONTROL_LOGICA',
    'RB6 2nd Start ON Fault': 'CONTROL_LOGICA',
    'RB3 Cartridge Color Comparison Fault': 'PROCESO_PINTURA',
    'RB4 Cartridge Color Comparison Fault': 'PROCESO_PINTURA',
    'RB5 Cartridge Color Comparison Fault': 'PROCESO_PINTURA',
    'RB6 Cartridge Color Comparison Fault': 'PROCESO_PINTURA',
    'PRIMER PREHEAT - CAB TEMP. FAULT': 'SUMINISTROS',
    'RB3 Job Not Complete Alarm': 'CONTROL_LOGICA',
    'RB4 Job Not Complete Alarm': 'CONTROL_LOGICA',
    'RB5 Job Not Complete Alarm': 'CONTROL_LOGICA',
    'RB6 Job Not Complete Alarm': 'CONTROL_LOGICA',
    'RB3 2nd Start ON Fault': 'CONTROL_LOGICA',
    'RB4 2nd Start ON Fault': 'CONTROL_LOGICA',
    'OP1 Over run': 'PROCESO_PINTURA',
    'Disable Charge work goes in': 'PROCESO_PINTURA',
    'SafetyHandle ENT-L(Detail)': 'SEGURIDAD',
    'R/B-3 ALARM': 'HARDWARE_ROBOT',
    'R/B-5 ALARM': 'HARDWARE_ROBOT',
    'BASE ASH REHEATER CIRCULATION FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH REHEATER CIRCULATION FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH #1 AIR SUPPLY FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH #2 AIR SUPPLY FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - CONTROL POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - DC POWER (AVR) FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - DC POWER (AVR) LFE': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - DC POWER (AVR) LOW': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - INSTRUMENT POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - PREHEATER POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - REHEATER POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER ASH C/P - SOLENOID POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER ASH PREHEATER - #1 CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER ASH PREHEATER - #2 CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER ASH PREHEATER - GAS PRESSURE HIGH': 'SUMINISTROS',
    'PRIMER ASH PREHEATER - TEMP. HIGH': 'SUMINISTROS',
    'PRIMER ASH REHEATER - CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'PRIMER ASH REHEATER - GAS PRESSURE HIGH': 'SUMINISTROS',
    'PRIMER ASH REHEATER - GAS PRESSURE LOW': 'SUMINISTROS',
    'PRIMER ASH REHEATER BURNER BLOWER - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH REHEATER CIRCULATION FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH WASHER PUMP - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH WASHER PUMP - PANEL MODE OFF': 'SISTEMA_ELECTRICO',
    'AVR1 Change warning': 'SISTEMA_ELECTRICO',
    'DC POWER FAULT PL1(MAIN)': 'SISTEMA_ELECTRICO',
    'DC Power Shut OFF PL1(MAIN)': 'SISTEMA_ELECTRICO',
    'E-Stop Entrance Left (Detail)': 'SEGURIDAD',
    'P1-L1 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'PCS PLC FAN FAULT': 'SISTEMA_ELECTRICO',
    'PCS RMT1 FAN Fault': 'SISTEMA_ELECTRICO',
    'DC POWER FAULT PL2(MAIN)': 'SISTEMA_ELECTRICO',
    'PS1 Change warning': 'SISTEMA_ELECTRICO',
    'BASE ASH #1 AIR SUPPLY FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH #2 AIR SUPPLY FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - CONTROL POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - DC POWER (AVR) FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - DC POWER (AVR) LFE': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - DC POWER (AVR) LOW': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - INSTRUMENT POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - PREHEATER POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - REHEATER POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH C/P - SOLENOID POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH PREHEATER - #1 CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH PREHEATER - #2 CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH PREHEATER - GAS PRESSURE HIGH': 'SUMINISTROS',
    'BASE ASH PREHEATER - TEMP. HIGH': 'SUMINISTROS',
    'BASE ASH REHEATER - CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'BASE ASH REHEATER - GAS PRESSURE HIGH': 'SUMINISTROS',
    'BASE ASH REHEATER - GAS PRESSURE LOW': 'SUMINISTROS',
    'BASE ASH REHEATER BURNER BLOWER - DSW FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH AIR SUPPLY FAN - DSW FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH C/P - CONTROL POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR ASH C/P - DC POWER (AVR) FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH C/P - DC POWER (AVR) LFE': 'SISTEMA_ELECTRICO',
    'CLEAR ASH C/P - DC POWER (AVR) LOW': 'SISTEMA_ELECTRICO',
    'CLEAR ASH C/P - INSTRUMENT POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR ASH C/P - PREHEATER POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR ASH C/P - REHEATER POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR ASH C/P - SOLENOID POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR ASH PREHEATER - #1 CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR ASH PREHEATER - #2 CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR ASH PREHEATER - GAS PRESSURE HIGH': 'SUMINISTROS',
    'CLEAR ASH PREHEATER - GAS PRESSURE LOW': 'SUMINISTROS',
    'CLEAR ASH PREHEATER - TEMP. HIGH': 'SUMINISTROS',
    'CLEAR ASH REHEATER - CONTROL MOTOR POWER OFF': 'SISTEMA_ELECTRICO',
    'CLEAR ASH REHEATER - GAS PRESSURE HIGH': 'SUMINISTROS',
    'CLEAR ASH REHEATER - GAS PRESSURE LOW': 'SUMINISTROS',
    'CLEAR ASH REHEATER BURNER BLOWER - DSW FAULT': 'SISTEMA_ELECTRICO',
    'PLC Connect Error(-1) P2:L7:1': 'COMUNICACIÓN',
    'PLC Connect Error(-1) P2:L7:2': 'COMUNICACIÓN',
    'PLC Connect Error(-1) P2:L7:3': 'COMUNICACIÓN',
    'AVR fault interception batch(OP1)': 'SISTEMA_ELECTRICO',
    'AVR voltage decrease batch(OP1)': 'SISTEMA_ELECTRICO',
    'FL-remote communication fault batch': 'COMUNICACIÓN',
    'FB Setting Error': 'CONTROL_LOGICA',
    'BCR I/F Unit Fault': 'COMUNICACIÓN',
    'P1-L2 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'BASE EXHAUST & PREHEAT C/P - TEC15401 CONTROLLER FAULT': 'SISTEMA_ELECTRICO',
    'Auto On Error': 'CONTROL_LOGICA',
    'RB1 Auto ON Fault': 'CONTROL_LOGICA',
    'AVR fault interception batch(OP3)': 'SISTEMA_ELECTRICO',
    'AVR voltage decrease batch(OP3)': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT - HAB NON HEATUP': 'SUMINISTROS',
    'CLEAR ASH - CONDUCTIVITY HIGH': 'SUMINISTROS',
    'RB3 Auto ON Fault': 'CONTROL_LOGICA',
    'RB4 Docking Wait Fault': 'CONTROL_LOGICA',
    'PRIMER PREHEAT HAB BURNER - BURNER STOP': 'SUMINISTROS',
    'AVR exchange time batch(OP1)': 'SISTEMA_ELECTRICO',
    'OP1 warning circuit off': 'SISTEMA_ELECTRICO',
    'CLEAR ASH WASHER TANK - H LEVEL': 'SUMINISTROS',
    'OP2 Motion bad': 'PROCESO_PINTURA',
    'OP2 transportation defect': 'PROCESO_PINTURA',
    'AVR fault interception batch(OP2)': 'SISTEMA_ELECTRICO',
    'AVR voltage decrease batch(OP2)': 'SISTEMA_ELECTRICO',
    'OP2 MCCB off batch': 'SISTEMA_ELECTRICO',
    'OP2 warning circuit off': 'SISTEMA_ELECTRICO',
    'Detaching Error': 'CONTROL_LOGICA',
    'RB1 Detaching Wait Fault': 'CONTROL_LOGICA',
    'OP1 Motion bad': 'PROCESO_PINTURA',
    'OP1 transportation defect': 'PROCESO_PINTURA',
    'PRIMER PREHEAT CAB CIRCULATION FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH AIR SUPPLY FAN - ROTATION LOW': 'SISTEMA_ELECTRICO',
    'CLEAR ASH AIR SUPPLY FAN - PRESSURE LOW': 'SUMINISTROS',
    'OP1 INV fault': 'SISTEMA_ELECTRICO',
    'PCS TOTAL COMMUNICATION FAULT': 'COMUNICACIÓN',
    'R/B-6 ALARM': 'HARDWARE_ROBOT',
    'RB3 Docking Wait Fault': 'CONTROL_LOGICA',
    'PLC Connect Error(-3) P2:L5:13': 'COMUNICACIÓN',
    'P1-L7 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'BCR Reading Cycle Time Over': 'COMUNICACIÓN',
    'PCS-RMT5 I/O module fault': 'SISTEMA_ELECTRICO',
    'Safety PLC synthesis fault': 'SEGURIDAD',
    'OP3 MCCB off batch': 'SISTEMA_ELECTRICO',
    'OP2 INV fault': 'SISTEMA_ELECTRICO',
    'P1-L5 COMMUNICATION ALARM': 'COMUNICACIÓN',
    'PRIMER ASH #1 AIR SUPPLY FAN - PRESSURE LOW': 'SUMINISTROS',
    'BASE ASH #2 AIR SUPPLY FAN - PRESSURE LOW': 'SUMINISTROS',
    'PLC Connect Error(-4) P2:L5:34-P1:L1:9': 'COMUNICACIÓN',
    'SHIFT PULSE ERROR': 'CONTROL_LOGICA',
    'PCS-CPU I/O module fault': 'SISTEMA_ELECTRICO',
    'C5 Job Delay': 'CONTROL_LOGICA',
    'C6 Job Delay': 'CONTROL_LOGICA',
    'RB3 Detaching Wait Fault': 'CONTROL_LOGICA',
    'RB4 Detaching Wait Fault': 'CONTROL_LOGICA',
    'PLC Connect Error(-3) P2:L5:34-P1:L1:9': 'COMUNICACIÓN',
    'PRIMER ASH PREHEATER - BURNER STOP': 'SUMINISTROS',
    'PLC Connect Error(-4) P2:L5:34-P1:L1:7': 'COMUNICACIÓN',
    'PLC Connect Error(-3) P2:L5:34-P1:L1:7': 'COMUNICACIÓN',
    'Conveyor Pulse Fault': 'CONTROL_LOGICA',
    'START PH ERROR': 'CONTROL_LOGICA',
    'BASE ASH WASHER TANK - H LEVEL': 'SUMINISTROS',
    'PRIMER PREHEAT HAB BURNER - PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'No Paint Fault': 'PROCESO_PINTURA',
    'RB1 No Paint Fault': 'PROCESO_PINTURA',
    'PCS-RMT3 I/O module fault': 'SISTEMA_ELECTRICO',
    'PRIMER ASH WASHER TANK - H LEVEL': 'SUMINISTROS',
    'OP3 INV fault': 'SISTEMA_ELECTRICO',
    'PC10P Clock Fault': 'CONTROL_LOGICA',
    'RB4 PC10P Clock Fault': 'CONTROL_LOGICA',
    'CONVEYOR PULSE ERROR': 'CONTROL_LOGICA',
    'Safety PLC synthesis communication fault': 'SEGURIDAD',
    'AVR exchange time batch(OP3)': 'SISTEMA_ELECTRICO',
    'PCS-RMT1 I/O module fault': 'SISTEMA_ELECTRICO',
    'PRIMER ASH REHEATER - BURNER STOP': 'SUMINISTROS',
    'SHORT ALARM': 'SISTEMA_ELECTRICO',
    'OP3 lead back fault': 'PROCESO_PINTURA',
    'Safety PLC synthesis lead back fault': 'SEGURIDAD',
    'BASE #1 EXHAUST FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'BASE #2 EXHAUST FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH PREHEATER - PROVING SYSTEM VALVE FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH WASHER PUMP - INV FAULT': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT CAB CIRCULATION FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT HAB CIRCULATION FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH AIR SUPPLY FAN - INV FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH REHEATER - PROVING SYSTEM VALVE FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH WASHER PUMP - INV FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR EXHAUST FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER #1 EXHAUST FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER #2 EXHAUST FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH #1 AIR SUPPLY FAN - INV FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH #2 AIR SUPPLY FAN - INV FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH WASHER PUMP - INV FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER PREHEAT HAB CIRCULATION FAN - INV. FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH - SUPPLY AIR TEMP. FAULT': 'SUMINISTROS',
    'PRIMER ASH WASHER TANK - L LEVEL': 'SUMINISTROS',
    'PRIMER ASH REHEATER CIRCULATION FAN - PRESSURE LOW': 'SUMINISTROS',
    'PRIMER ASH REHEATER CIRCULATION FAN - ROTATION LOW': 'SISTEMA_ELECTRICO',
    'RB5 PC10P Clock Fault': 'CONTROL_LOGICA',
    'RB2 Auto ON Fault': 'CONTROL_LOGICA',
    'BASE ASH #1 AIR SUPPLY FAN - PRESSURE LOW': 'SUMINISTROS',
    'PLC Connect Error(-4) P2:L5:34-P1:L1:11': 'COMUNICACIÓN',
    'BASE ASH PREHEATER - BURNER STOP': 'SUMINISTROS',
    'BASE ASH REHEATER - BURNER STOP': 'SUMINISTROS',
    'CLEAR ASH PREHEATER - BURNER STOP': 'SUMINISTROS',
    'CLEAR ASH REHEATER - BURNER STOP': 'SUMINISTROS',
    'OP2 Motion fault': 'PROCESO_PINTURA',
    'BASE ASH #1 AIR SUPPLY FAN - INV FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH #2 AIR SUPPLY FAN - INV FAULT': 'SISTEMA_ELECTRICO',
    'START PH PHOTO AXIS ERROR': 'CONTROL_LOGICA',
    'RB3 PC10P Clock Fault': 'CONTROL_LOGICA',
    'RB4 Auto ON Fault': 'CONTROL_LOGICA',
    'CLEAR ASH REHEATER - TEMP. HIGH': 'SUMINISTROS',
    'PRIMER ASH REHEATER - GAS TRAIN GAS LEAKAGE': 'SUMINISTROS',
    'BASE ASH REHEATER - GAS TRAIN GAS LEAKAGE': 'SUMINISTROS',
    'CLEAR ASH REHEATER - GAS TRAIN GAS LEAKAGE': 'SUMINISTROS',
    'PRIMER PREHEAT HAB BURNER - GAS TRAIN GAS LEAK': 'SUMINISTROS',
    'CLEAR ASH WASHER TANK - FILTER BLOCKAGE': 'SUMINISTROS',
    'PRIMER PREHEAT HAB CIRCULATION FAN - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH WASHER PUMP - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'RB5 Auto ON Fault': 'CONTROL_LOGICA',
    'PRIMER PREHEAT - HAB TEMP. LOW': 'SUMINISTROS',
    'PCS-RMT2 I/O module fault': 'SISTEMA_ELECTRICO',
    'CLEAR ASH REHEATER - CUTOFF VALVE FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH REHEATER - MAIN VALVE FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH PREHEATER - #2 MAIN VALVE FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH AIR INTAKE DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'OP3 warning circuit off': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT HAB BURNER - BURNER STOP': 'SUMINISTROS',
    'PRIMER ASH AIR INTAKE DAMPER - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'CLEAR ASH PREHEATER - #1 PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'OP2 Sensor fault': 'PROCESO_PINTURA',
    'BASE ASH REHEATER - TEMP. HIGH': 'SUMINISTROS',
    'P3-L3 COMMUNICATION ALARM': 'COMUNICACIÓN',
    '1ST Read Back Fault': 'CONTROL_LOGICA',
    'PCS CPU Read Back Fault': 'CONTROL_LOGICA',
    'PCS Read Back FAULT': 'CONTROL_LOGICA',
    'OP1 Motion fault': 'PROCESO_PINTURA',
    'RB1 PC10P Clock Fault': 'CONTROL_LOGICA',
    'PRIMER ASH PREHEATER - #2 PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH PREHEATER - PROVING SYSTEM VALVE FAULT': 'SISTEMA_ELECTRICO',
    'BASE PREHEAT - HAB TEMP. LOW': 'SUMINISTROS',
    'BASE ASH #2 AIR SUPPLY FAN - ROTATION LOW': 'SISTEMA_ELECTRICO',
    'CLEAR ASH PREHEATER - #2 PROTECT RELAY FAULT': 'SISTEMA_ELECTRICO',
    'BASE ASH REHEATER CIRCULATION FAN - ROTATION LOW': 'SISTEMA_ELECTRICO',
    'BASE ASH REHEATER CIRCULATION FAN - PRESSURE LOW': 'SUMINISTROS',
    'CLEAR ASH PREHEATER - DIFFERENTIAL PRESSURE LOW': 'SUMINISTROS',
    'CLEAR ASH REHEATER CIRCULATION FAN - OVERLOAD': 'SISTEMA_ELECTRICO',
    'RB2 Detaching Wait Fault': 'CONTROL_LOGICA',
    'RB5 Detaching Wait Fault': 'CONTROL_LOGICA',
    'RB6 Detaching Wait Fault': 'CONTROL_LOGICA',
    'RB2 PC10P Clock Fault': 'CONTROL_LOGICA',
    'CLEAR EXHAUST FAN - MOVED FAULT': 'SISTEMA_ELECTRICO',
    'PRIMER ASH #2 AIR SUPPLY FAN - PRESSURE LOW': 'SUMINISTROS',
    'PRIMER ASH PREHEATER - DIFFERENTIAL PRESSURE LOW': 'SUMINISTROS',
    'RB6 Auto ON Fault': 'CONTROL_LOGICA',
    'OP1 Sensor fault': 'PROCESO_PINTURA',
    'CLEAR ASH REHEATER CIRCULATION FAN - ROTATION LOW': 'SISTEMA_ELECTRICO',
    'CLEAR ASH REHEATER CIRCULATION FAN - PRESSURE LOW': 'SUMINISTROS',
    'BASE ASH #1 AIR SUPPLY FAN - ROTATION LOW': 'SISTEMA_ELECTRICO',
    'PRIMER PREHEAT - HAB NON HEATUP': 'SUMINISTROS',
    'RB5 2nd Start ON Fault': 'CONTROL_LOGICA',
}


# ─── preprocess_plc_log — copiado verbatim ────────────────────────────────────

def preprocess_plc_log(text: str) -> str:
    """Transforma logs técnicos de PLC en lenguaje natural descriptivo.
    (Idéntico a src/preprocessing.py del repositorio original)."""
    text = text.upper()

    translations = {
        r"PLC CONNECT ERROR\(-1\)": "Communication failure in PLC system",
        r"HUMAN DETECT LC": "Safety violation: Human detected by light curtain",
        r"R/B-(\d+)": r"Painting Robot unit \1",
        r"C(\d+) HEAVY ERROR": r"Critical controller error in unit C\1",
        r"SAFETYHANDLE": "Safety handle sensor",
        r"ENT-R": "Right entrance",
        r"ENTRANCE": "Main entrance",
        r"FAULT": "Operational fault",
        r"DETAIL": "Detailed diagnostic",
    }

    for pattern, replacement in translations.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"P\d:L\d:\d+[-]*[P\d:L\d:\d+]*", "(Network Location)", text)
    text = text.replace("(", " ").replace(")", " ").replace("-", " ")
    text = " ".join(text.split())

    return text.capitalize()


def _calcular_target_lookahead(grupo: pd.DataFrame, ventana_dias: int) -> pd.DataFrame:
    """Idéntico a src/preprocessing.py del repositorio original."""
    ventana = pd.Timedelta(days=ventana_dias)
    targets = []

    for i in range(len(grupo)):
        t_actual = grupo["START TIME"].iloc[i]
        mascara = (grupo["START TIME"] > t_actual) & (
            grupo["START TIME"] <= t_actual + ventana
        )
        eventos_futuros = grupo[mascara]
        target = int(eventos_futuros["ES_CRITICA"].sum() > 0)
        targets.append(target)

    grupo_copy = grupo.copy()
    grupo_copy["TARGET"] = targets
    return grupo_copy


def _asignar_turno(hora: int) -> int:
    """Idéntico a src/preprocessing.py del repositorio original."""
    if 6 <= hora < 14:
        return 0
    elif 14 <= hora < 22:
        return 1
    else:
        return 2


def _preprocesamiento(df: pd.DataFrame, umbral_critico: int, ventana_dias: int) -> pd.DataFrame:
    """
    Equivalente a src/preprocessing.py::preprocesamiento(), adaptado para
    recibir un DataFrame ya cargado (en vez de leer un Excel directamente),
    ya que en el framework unificado el Agente 1 se encarga de la carga y
    normalización de columnas. La lógica de transformación es idéntica.
    """
    df = df.copy()
    df["START TIME"] = pd.to_datetime(df["START TIME"])
    df["END TIME"] = pd.to_datetime(df["END TIME"])

    df["DURACION_MIN"] = (df["END TIME"] - df["START TIME"]).dt.total_seconds() / 60
    df = df[df["DURACION_MIN"] > 0].copy()
    df = df.sort_values("START TIME").reset_index(drop=True)

    df["FAIL COMMENT_PROCESADO"] = df["FAIL COMMENT"].apply(preprocess_plc_log)
    df["CATEGORIA"] = df["FAIL COMMENT"].map(MAPEO_FALLAS).fillna("OTRO")
    df["ES_CRITICA"] = (df["DURACION_MIN"] >= umbral_critico).astype(int)

    # NOTA DE COMPATIBILIDAD: en pandas >= 3.0, groupby(...).apply() ya no
    # conserva la columna de agrupación en el resultado (comportamiento
    # distinto a pandas 2.x, con el que el repositorio original fue escrito).
    # Para no perder ADDRESS, se calcula TARGET por grupo y se reasigna por
    # índice — el valor de TARGET calculado es IDÉNTICO al original, esto
    # es solo un cambio mecánico de compatibilidad, no de lógica.
    df = df.sort_values(["ADDRESS", "START TIME"]).reset_index(drop=True)
    df["TARGET"] = 0
    for _, idx in df.groupby("ADDRESS").groups.items():
        grupo_con_target = _calcular_target_lookahead(df.loc[idx], ventana_dias)
        df.loc[idx, "TARGET"] = grupo_con_target["TARGET"].values

    df["hora_del_dia"] = df["START TIME"].dt.hour
    df["dia_semana"] = df["START TIME"].dt.dayofweek
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["mes"] = df["START TIME"].dt.month
    df["turno_cod"] = df["hora_del_dia"].apply(_asignar_turno)

    return df


# ─── generate_embeddings — copiado verbatim (misma lógica, mismo nombre de columnas) ─

def _generate_embeddings(df: pd.DataFrame, n_components: int, embedding_model) -> pd.DataFrame:
    """Idéntico a src/embeddings.py::generate_embeddings() del repo original.
    NOTA: el PCA se reajusta (fit) con los datos que se le pasen en cada
    llamada — igual que el original. No reutiliza un PCA de entrenamiento
    previo."""
    from sklearn.decomposition import PCA

    textos_unicos = df["FAIL COMMENT_PROCESADO"].unique()
    vectores_unicos = embedding_model.encode(textos_unicos, show_progress_bar=False)

    diccionario_embeddings = dict(zip(textos_unicos, vectores_unicos))
    embeddings_finales = np.stack(
        df["FAIL COMMENT_PROCESADO"].map(diccionario_embeddings).values
    )

    pca = PCA(n_components=min(n_components, embeddings_finales.shape[0] - 1, embeddings_finales.shape[1]))
    embeddings_reducidos = pca.fit_transform(embeddings_finales)

    columnas_pca = [f"pca_emb_{i}" for i in range(embeddings_reducidos.shape[1])]
    # Rellenar hasta n_components si el PCA produjo menos (dataset muy chico)
    df_embeddings = pd.DataFrame(embeddings_reducidos, columns=columnas_pca, index=df.index)
    for i in range(embeddings_reducidos.shape[1], n_components):
        df_embeddings[f"pca_emb_{i}"] = 0.0

    return pd.concat([df, df_embeddings], axis=1)


# ─── create_features — copiado verbatim ────────────────────────────────────────

def _create_features(df: pd.DataFrame, n_eventos: int, n_componentes_pca: int) -> tuple[pd.DataFrame, list[str]]:
    """Idéntico a src/features.py::create_features() del repositorio original,
    incluyendo el LabelEncoder que se reajusta en cada llamada (bug conocido,
    preservado a propósito)."""
    df = df.sort_values(["ADDRESS", "START TIME"]).reset_index(drop=True)

    df["tiempo_desde_ultima_falla"] = df.groupby("ADDRESS")["START TIME"].transform(
        lambda x: x.diff().dt.total_seconds() / 3600
    )

    df["media_duracion_3_previas"] = df.groupby("ADDRESS")["DURACION_MIN"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["max_duracion_5_previas"] = df.groupby("ADDRESS")["DURACION_MIN"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).max()
    )
    df["min_duracion_5_previas"] = df.groupby("ADDRESS")["DURACION_MIN"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).min()
    )
    df["tendencia_duracion"] = df.groupby("ADDRESS")["DURACION_MIN"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) >= 2 else 0,
            raw=True,
        )
    )

    df["conteo_criticas_previas"] = df.groupby("ADDRESS")["ES_CRITICA"].transform(
        lambda x: x.shift(1).rolling(n_eventos, min_periods=1).sum()
    )
    df["proporcion_criticas_previas"] = df.groupby("ADDRESS")["ES_CRITICA"].transform(
        lambda x: x.shift(1).rolling(n_eventos, min_periods=1).mean()
    )
    df["ultima_fue_critica"] = (
        df.groupby("ADDRESS")["ES_CRITICA"].transform(lambda x: x.shift(1))
    ).fillna(0).astype(int)

    df["misma_categoria_anterior"] = df.groupby("ADDRESS")["CATEGORIA"].transform(
        lambda x: (x == x.shift(1)).astype(int)
    )

    df["temp_unique_time"] = df["START TIME"] + df.groupby(
        ["ADDRESS", "START TIME"]
    ).cumcount().astype("timedelta64[ns]")

    df["fallas_componente_7d"] = df.groupby("ADDRESS")["DURACION_MIN"].transform(
        lambda x: x.shift(1)
        .set_axis(df.loc[x.index, "temp_unique_time"])
        .rolling("7D")
        .count()
        .reindex(x.index)
        .values
    )
    df = df.drop(columns=["temp_unique_time"])

    df["n_falla_componente"] = df.groupby("ADDRESS").cumcount()

    # ── LabelEncoder — se reajusta cada vez que corre esta función.
    #    Comportamiento IDÉNTICO al repositorio original, bug incluido. ──
    le = LabelEncoder()
    df["categoria_cod"] = le.fit_transform(df["CATEGORIA"])

    feature_cols = [
        "hora_del_dia",
        "dia_semana",
        "es_fin_semana",
        "mes",
        "turno_cod",
        "tiempo_desde_ultima_falla",
        "media_duracion_3_previas",
        "max_duracion_5_previas",
        "min_duracion_5_previas",
        "tendencia_duracion",
        "conteo_criticas_previas",
        "proporcion_criticas_previas",
        "ultima_fue_critica",
        "misma_categoria_anterior",
        "fallas_componente_7d",
        "n_falla_componente",
        "categoria_cod",
    ]

    embedding_feature_cols = [f"pca_emb_{i}" for i in range(n_componentes_pca)]
    feature_cols.extend(embedding_feature_cols)

    cols_presentes = [c for c in feature_cols if c in df.columns]
    extra_cols = [c for c in ("PANEL",) if c in df.columns]
    df_model = df[cols_presentes + ["TARGET", "START TIME", "ADDRESS"] + extra_cols].copy()
    df_model = df_model.dropna(subset=["TARGET"])
    df_model[cols_presentes] = df_model[cols_presentes].fillna(0)
    df_model = df_model.sort_values("START TIME").reset_index(drop=True)

    return df_model, feature_cols


# ─── Predictor ─────────────────────────────────────────────────────────────────

class LegacyPLCPredictor(BasePredictor):
    """
    Envoltorio BasePredictor sobre el repositorio original
    luisroberto-maker/PLC-failure-prediction-pipeline. Preserva su
    comportamiento exacto, incluyendo:
      - MAPEO_FALLAS idéntico
      - preprocess_plc_log() idéntico
      - create_features() idéntico (17 features + PCA, LabelEncoder
        reajustado en cada llamada)
      - Salida idéntica: COMPONENTE, ULTIMO_EVENTO, PROBABILIDAD_FALLA,
        NIVEL_RIESGO (mismos bins y emojis: 🟢 BAJO / 🟡 MEDIO / 🔴 ALTO)

    Uso típico — cargar el .pkl YA entrenado con el repo original:
        predictor = LegacyPLCPredictor.from_pretrained(
            "models/modelo.pkl",
            umbral_critico=30, ventana_dias=7,
            n_eventos=10, n_componentes_pca=15,
        )
        ranking = predictor.predict_all(df)   # df con columnas canónicas
    """

    def __init__(
        self,
        umbral_critico: int = 30,
        ventana_dias: int = 7,
        n_eventos: int = 10,
        n_componentes_pca: int = 15,
    ):
        self.umbral_critico = umbral_critico
        self.ventana_dias = ventana_dias
        self.n_eventos = n_eventos
        self.n_componentes_pca = n_componentes_pca
        self.modelo = None
        self._embedding_model = None
        self.feature_cols: list[str] = []

    # ── Carga del modelo ya entrenado (camino principal de uso) ────────────────

    @classmethod
    def from_pretrained(
        cls,
        modelo_path: str,
        umbral_critico: int = 30,
        ventana_dias: int = 7,
        n_eventos: int = 10,
        n_componentes_pca: int = 15,
    ) -> "LegacyPLCPredictor":
        """
        Carga el modelo .pkl EXACTAMENTE como lo hace
        src/predictions.py::generate_and_rank_predictions() del repo
        original: joblib.load(modelo_path) sobre un estimador de
        scikit-learn ya entrenado (no un objeto empaquetado).
        """
        predictor = cls(umbral_critico, ventana_dias, n_eventos, n_componentes_pca)
        logger.info("Cargando modelo legado desde '%s'...", modelo_path)
        predictor.modelo = joblib.load(modelo_path)
        if not hasattr(predictor.modelo, "predict_proba"):
            raise ValueError(
                f"El objeto cargado desde {modelo_path} no tiene predict_proba(). "
                f"Se esperaba un clasificador de scikit-learn entrenado "
                f"(tal como lo produce/consume el repositorio original)."
            )
        return predictor

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(_SENTENCE_TRANSFORMER_MODEL)
        return self._embedding_model

    # ── Entrenamiento (opcional — solo si NO tienes ya un .pkl entrenado) ──────

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Réplica de un entrenamiento equivalente al que produjo el .pkl
        original: mismas features, mismo preprocesamiento, un
        RandomForestClassifier con hiperparámetros razonables por defecto.

        El repositorio original no incluye un script de entrenamiento —
        solo consume un .pkl ya entrenado. Este método existe para cumplir
        la interfaz BasePredictor y por si necesitas reentrenar desde cero
        con esta misma lógica de features, pero el camino recomendado si ya
        tienes tu .pkl es from_pretrained().
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, average_precision_score

        df_prep = _preprocesamiento(df, self.umbral_critico, self.ventana_dias)
        df_prep = _generate_embeddings(df_prep, self.n_componentes_pca, self._get_embedding_model())
        df_model, feature_cols = _create_features(df_prep, self.n_eventos, self.n_componentes_pca)
        self.feature_cols = feature_cols

        split_idx = int(len(df_model) * 0.8)
        train, test = df_model.iloc[:split_idx], df_model.iloc[split_idx:]
        X_train, y_train = train[feature_cols], train["TARGET"].astype(int)
        X_test, y_test = test[feature_cols], test["TARGET"].astype(int)

        self.modelo = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.modelo.fit(X_train, y_train)

        y_prob = self.modelo.predict_proba(X_test)[:, 1]
        return {
            "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
            "auc_pr": round(average_precision_score(y_test, y_prob), 4),
            "n_test": len(y_test),
        }

    # ── Predicción — idéntica a generate_and_rank_predictions() ────────────────

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.modelo is None:
            raise RuntimeError(
                "El predictor no tiene modelo cargado. Usa "
                "LegacyPLCPredictor.from_pretrained(ruta) o llama a fit() primero."
            )

        warnings.warn(
            "LegacyPLCPredictor reajusta el PCA de embeddings y el LabelEncoder "
            "de categorías en cada llamada a predict_all(), igual que el "
            "repositorio original. Si los datos de inferencia difieren mucho "
            "de los de entrenamiento, esto puede desalinear las features "
            "respecto a lo que el modelo aprendió.",
            UserWarning,
        )

        df_prep = _preprocesamiento(df, self.umbral_critico, self.ventana_dias)
        df_prep = _generate_embeddings(df_prep, self.n_componentes_pca, self._get_embedding_model())
        df_model, feature_cols = _create_features(df_prep, self.n_eventos, self.n_componentes_pca)
        self.feature_cols = feature_cols

        ultimo_evento = (
            df_model.sort_values("START TIME").groupby("ADDRESS").last().reset_index()
        )

        cols_presentes = [c for c in feature_cols if c in ultimo_evento.columns]
        X_ultimo = ultimo_evento[cols_presentes].fillna(0)
        prob_falla = self.modelo.predict_proba(X_ultimo)[:, 1]

        ranking = pd.DataFrame(
            {
                "COMPONENTE": ultimo_evento["ADDRESS"],
                "ULTIMO_EVENTO": ultimo_evento["START TIME"].dt.date,
                "PROBABILIDAD_FALLA": prob_falla,
                "NIVEL_RIESGO": pd.cut(
                    prob_falla,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["🟢 BAJO", "🟡 MEDIO", "🔴 ALTO"],
                ),
            }
        )

        # Columnas puramente informativas para el reporte — NO alteran
        # PROBABILIDAD_FALLA ni la lógica de predicción original.
        ranking["PANEL"] = (
            ultimo_evento["PANEL"].values if "PANEL" in ultimo_evento.columns
            else "PANEL_DESCONOCIDO"
        )
        ranking["NIVEL_CONFIANZA"] = self._confianza_best_effort(X_ultimo)

        return ranking.sort_values("PROBABILIDAD_FALLA", ascending=False).reset_index(drop=True)

    def _confianza_best_effort(self, X) -> list[str]:
        """
        Intenta estimar confianza vía dispersión entre árboles SOLO si el
        modelo cargado es un ensamble tipo Random Forest (tiene
        .estimators_). Es puramente informativo para el reporte — el
        repositorio original no calcula esto, así que si el modelo no lo
        soporta, se muestra 'N/D' sin afectar nada más.
        """
        if not hasattr(self.modelo, "estimators_"):
            return ["N/D"] * len(X)
        try:
            preds = np.array([t.predict_proba(X)[:, 1] for t in self.modelo.estimators_])
            std = preds.std(axis=0)
            return [
                "BAJO" if s > 0.25 else ("MEDIO" if s > 0.15 else "ALTO")
                for s in std
            ]
        except Exception:
            return ["N/D"] * len(X)

    # ── Interfaz compartida ──────────────────────────────────────────────────

    def report_schema(self) -> ReportSchema:
        return ReportSchema(
            domain_label="Predicción de fallas PLC — modelo legado (repositorio original)",
            id_col="COMPONENTE", id_label="Componente",
            group_col="PANEL", group_label="Panel / línea",
            primary_metric_col="PROBABILIDAD_FALLA", primary_metric_label="Probabilidad de falla",
            status_col="NIVEL_RIESGO",
            status_levels=[("🔴 ALTO", "#dc2626"), ("🟡 MEDIO", "#d97706"), ("🟢 BAJO", "#059669")],
            confidence_col="NIVEL_CONFIANZA",
            extra_cols=[("ULTIMO_EVENTO", "Último evento registrado")],
            higher_is_worse=True,
        )

    def feature_importance(self) -> dict:
        if self.modelo is None or not hasattr(self.modelo, "feature_importances_"):
            return {}
        return dict(
            sorted(zip(self.feature_cols, self.modelo.feature_importances_),
                  key=lambda x: x[1], reverse=True)
        )
