import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import t as t_student
from scipy.optimize import minimize
from datetime import datetime
import warnings

# Silenciar advertencias de optimización numérica
warnings.filterwarnings('ignore')

# ==========================================================================
# 1. CONFIGURACIÓN E INTERFAZ CORPORATIVA (SUITE THE ONE RING)
# ==========================================================================
st.set_page_config(page_title="The One Ring: Bot 4 GARCH-Montecarlo", layout="wide", page_icon="🔮")

st.title("🔮 Bot 4 — GARCH + Montecarlo Breakout Cones")
st.caption("Ecosistema Cuantitativo 'The One Ring' | Enfoque: Optimización MLE t-Student y Caminatas Aleatorias de Frontera")
st.markdown("---")

# RESTRICCIONES DE CONTROL INSTITUCIONAL
MIN_USD_PER_ORDER = 10.0
N_SIMULACIONES = 10000  # Número de caminos aleatorios vectorizados
HORIZONTE_DIAS = 5      # Ventana cerrada de arbitraje estocástico

# BARRA LATERAL DE PARAMETRIZACIÓN - OPTIMIZADA PARA FLEXIBILIDAD TÁCTICA
st.sidebar.header("⚙️ Control de Tesorería y Filtros")
presupuesto_diario_bot4 = st.sidebar.number_input("Presupuesto de Despliegue Hoy (USD)", value=550.0, step=50.0)
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Calibración de Filtros de Frontera")
# Modificación: Bajamos los percentiles para capturar el inicio de la ruptura y evitar el agotamiento
percentil_entrada = st.sidebar.slider("Percentil de Entrada (Techo Estocástico)", 50, 75, 60) / 100

# Modificación: Relajamos la expansión de volatilidad (por defecto 15% en lugar de 40%) para activarse con flujos institucionales normales
umbral_expansion_vol = st.sidebar.slider("Umbral de Expansión de Volatilidad Rápida (%)", 5, 35, 15) / 100

# ==========================================================================
# 2. UNIVERSO TOTAL DE HIPER-VOLATILIDAD SINCRO (150 TICKERS)
# ==========================================================================
UNIVERSO_MOM_VOL = [
    "IBIT", "ETHA", "MSTR", "GLXY", "COIN", "BMNR", "BTBT", "HUT", "CIFR", "WULF", "SOL", "MARA", "RIOT", "CLSK", "SOFI", "NU", "SQ", "PYPL", "MELI", "DLO",
    "IONQ", "RGTI", "QBTS", "HON", "SYM", "PATH", "ISRG", "TER", "RKLB", "AVAV", "ASTS", "KTOS", "OKLO", "SMR", "BWXT", "CCJ", "LEU", "NXE", "UUUU", "GEV", 
    "VST", "CEG", "XOM", "CVX", "TPL", "NFE", "OKE", "ET", "FANG", "NVDA", "AMD", "AVGO", "ARM", "TSM", "ASML", "MU", "AMAT", "LRCX", "KLAC", "QCOM", 
    "ADI", "NXPI", "TXN", "ON", "MPWR", "CRWD", "PANW", "FTNT", "ZS", "DDOG", "NET", "SNOW", "NOW", "TEAM", "WDAY", "SHOP", "MDB", "TTD", "OKTA", "S",
    "VKTX", "HIMS", "CRSP", "MRNA", "AMGN", "REGN", "VRTX", "LLY", "NVO", "GILD", "PFE", "TSLA", "CELH", "NFLX", "BABA", "PDD", "JD", "BIDU", "NIO", "LI", 
    "FUTU", "DKNG", "RBLX", "AFRM", "UPST", "U", "TWLO", "LMT", "RTX", "GD", "NOC", "TDG", "AAPL", "MSFT", "GOOGL", "COST", "WMT", "JPM", "V", "MA", 
    "MCO", "SPGI", "ADP", "BAC", "MS", "GS", "BLK"
]
TICKERS = list(set
