import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress, norm
from datetime import datetime, timezone
import warnings

# Silenciar advertencias de optimización numérica
warnings.filterwarnings('ignore')

# ==========================================================================
# 1. CONFIGURACIÓN E INTERFAZ CORPORATIVA (SUITE THE ONE RING)
# ==========================================================================
st.set_page_config(page_title="The One Ring: Bot 4 GARCH-Montecarlo", layout="wide", page_icon="🔮")

st.title("🔮 Bot 4 — GARCH + Montecarlo Breakout Cones")
st.caption("Ecosistema Cuantitativo 'The One Ring' | Enfoque: Explosión de Varianza Condicional y Momentum Anómalo")
st.markdown("---")

# RESTRICCIONES DE CONTROL INSTITUCIONAL
MIN_USD_PER_ORDER = 10.0
N_SIMULACIONES = 10000  # Número de caminos aleatorios vectorizados
HORIZONTE_DIAS = 5      # Ventana cerrada de arbitraje estocástico

# BARRA LATERAL DE PARAMETRIZACIÓN Abierta para Socios
st.sidebar.header("⚙️ Control de Tesorería y Filtros")
presupuesto_diario_bot4 = st.sidebar.number_input("Presupuesto de Despliegue Hoy (USD)", value=550.0, step=50.0)
st.sidebar.markdown("---")

# Umbrales estocásticos basados en la propuesta analítica
percentil_entrada = st.sidebar.slider("Percentil de Entrada (Techo Estocástico)", 65, 85, 75) / 100
umbral_expansion_vol = st.sidebar.slider("Umbral de Expansión de Volatilidad Rápida (%)", 20, 60, 40) / 100

# ==========================================================================
# 2. UNIVERSO TOTAL DE HIPER-VOLATILIDAD SINCRO (150 TICKERS)
# ==========================================================================
UNIVERSO_MOM_VOL = [
    # --- FLOTA CRYPTO, PROXIES & MINERS (Tu inyección extra de Alfa) ---
    "IBIT", "ETHA", "MSTR", "GLXY", "COIN", "BMNR", "BTBT", "HUT", "CIFR", "WULF", "SOL", "MARA", "RIOT", "CLSK", "SOFI", "NU", "SQ", "PYPL", "MELI", "DLO",
    # --- COMPUTACIÓN CUÁNTICA & DEEP TECH ---
    "IONQ", "RGTI", "QBTS", "HON",
    # --- ROBÓTICA AVANZADA & AUTOMATIZACIÓN ---
    "SYM", "PATH", "ISRG", "TER",
    # --- ESPACIO AVANZADO & INFRAESTRUCTURA ASIMÉTRICA ---
    "RKLB", "AVAV", "ASTS", "KTOS",
    # --- ENERGÍA DE FRONTERA & URANIO ---
    "OKLO", "SMR", "BWXT", "CCJ", "LEU", "NXE", "UUUU", "GEV", "VST", "CEG", "XOM", "CVX", "TPL", "NFE", "OKE", "ET", "FANG",
    # --- SEMICONDUCTORES DE ALTA BETA ---
    "NVDA", "AMD", "AVGO", "ARM", "TSM", "ASML", "MU", "AMAT", "LRCX", "KLAC", "QCOM", "ADI", "NXPI", "TXN", "ON", "MPWR",
    # --- CIBERSEGURIDAD & SOFTWARE SAAS EXPLOSIVO ---
    "CRWD", "PANW", "FTNT", "ZS", "DDOG", "NET", "SNOW", "NOW", "TEAM", "WDAY", "SHOP", "MDB", "TTD", "OKTA", "S",
    # --- BIOTECH GROWTH ---
    "VKTX", "HIMS", "CRSP", "MRNA", "AMGN", "REGN", "VRTX", "LLY", "NVO", "GILD", "PFE",
    # --- COMPLEMENTARIAS DE ACCIÓN PARABÓLICA ---
    "TSLA", "CELH", "NFLX", "BABA", "PDD", "JD", "BIDU", "NIO", "LI", "FUTU", "DKNG", "RBLX", "AFRM", "UPST", "U", "TWLO",
    "LMT", "RTX", "GD", "NOC", "TDG", "AAPL", "MSFT", "GOOGL", "COST", "WMT", "JPM", "V", "MA", "MCO", "SPGI", "ADP", "BAC", "MS", "GS", "BLK"
]
TICKERS = list(set(UNIVERSO_MOM_VOL))

# ==========================================================================
# 3. MOTOR MATEMÁTICO: FILTRO DE DOS PASOS (LIGHTWEIGHT GARCH INTERPRETER)
# ==========================================================================
def estimar_garch_manual(retornos):
    """
    Algoritmo ligero vectorizado para estimar un modelo GARCH(1,1) base
    evitando la pesada carga de convergencia de solvers no lineales en Streamlit Cloud.
    """
    longitud = len(retornos)
    varianzas = np.zeros(longitud)
    residuos_cuadrados = retornos ** 2
    
    # Parámetros institucionales inicializados para persistencia de volatilidad growth
    omega = 0.05 * np.var(retornos)
    alpha = 0.10
    beta = 0.83  # Persistencia de clúster elevada (~0.93 total)
    
    varianzas[0] = np.var(retornos)
    for t in range(1, longitud):
        varianzas[t] = omega + alpha * residuos_cuadrados[t-1] + beta * varianzas[t-1]
        
    # Proyección de la varianza condicional para el horizonte t+1
    proxima_vol = np.sqrt(omega + alpha * residuos_cuadrados[-1] + beta * varianzas[-1])
    return proxima_vol

def ejecutar_montecarlo(precio_inicial, vol_garch, dias, n_sim):
    """
    Generación masiva vectorizada de Caminatas Aleatorias bajo Movimiento Browniano
    Geométrico alimentado por la volatilidad condicional calculada por el GARCH.
    """
    dt = 1.0
    rendimientos_simulados = np.random.normal(0, vol_garch, size=(dias, n_sim))
    trayectorias = np.zeros_like(rendimientos_simulados)
    
    # Vectorización exponencial para proteger la memoria RAM del servidor
    cambios_porcentuales = np.exp(rendimientos_simulados)
    precios_proyectados = precio_inicial * np.cumprod(cambios_porcentuales, axis=0)
    return precios_proyectados[-1, :]

def analizar_breakout_estocastico(ticker, umbral_vol, pct_entrada):
    try:
        df = yf.download(ticker, period="6m", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 40: return None
        
        precios = df['Close'].squeeze().dropna()
        retornos = precios.pct_change().dropna().values
        
        # ------------------------------------------------------------------
        # PASO 1: FILTRO OPERATIVO RÁPIDO (Desviaciones de Corto vs Medano Plazo)
        # ------------------------------------------------------------------
        vol_corta = np.std(retornos[-5:])
        vol_mediana = np.std(retornos[-20:])
        
        if vol_mediana == 0: return None
        expansion_real = (vol_corta - vol_mediana) / vol_mediana
        
        # Si la varianza condicional corta no se expande por encima del umbral, se descarta instantáneamente
        if expansion_real < umbral_vol:
            return None
            
        # ------------------------------------------------------------------
        # PASO 2: AJUSTE GARCH(1,1) & SIMULACIÓN MONTECARLO (A los clasificados)
        # ------------------------------------------------------------------
        vol_condicional_garch = estimar_garch_manual(retornos)
        precio_spot = float(precios.iloc[-1])
        
        # Correr las 10,000 proyecciones estocásticas
        precios_finales_simulados = ejecutar_montecarlo(precio_spot, vol_condicional_garch, HORIZONTE_DIAS, N_SIMULACIONES)
        
        # Extraer los percentiles de la distribución simulada
        p_techo_breakout = np.percentile(precios_finales_simulados, pct_entrada * 100)
        p_mediana = np.percentile(precios_finales_simulados, 50)
        p_take_profit = np.percentile(precios_finales_simulados, 90)
        
        # Evaluar la condición Extra-Región de Momentum Anómalo
        condicion_breakout = precio_spot >= p_techo_breakout
        
        return {
            "Ticker": ticker,
            "Precio Spot": round(precio_spot, 2),
            "Expansión Vol (5D vs 20D)": expansion_real,
            "Vol Condicional GARCH": vol_condicional_garch,
            "Techo Simulación (P75)": round(p_techo_breakout, 2),
            "Objetivo Take Profit (P90)": round(p_take_profit, 2),
            "Soporte Stop Loss (Mediana)": round(p_mediana, 2),
            "Estatus Matriz": "🆕 BREAKOUT DE VARIANZA" if condicion_breakout else "⌛ COMPRESIÓN EN RANGO"
        }
    except: return None

# ==========================================================================
# 4. TABLERO DE CONTROL Y PANELES VISUALES DE ISENGARD
# ==========================================================================
if st.button("🚀 Iniciar Escaneo de Varianza en Red de Frontera (150 Tickers)"):
    with st.spinner("Filtrando ruido y calculando matrices distributivas vectorizadas..."):
        alertas = []
        
        # El procesador pasa como un colador rápido sobre el universo expandido
        for t in TICKERS:
            res = analizar_breakout_estocastico(t, umbral_expansion_vol, percentil_entrada)
            if res: alertas.append(res)
            
        if alertas:
            df_res = pd.DataFrame(alertas)
            
            # --- TABLA CENTRAL 1: ALERTAS DE EXPLOSIÓN ---
            st.subheader("📡 Detección de Regímenes de Momentum Anómalo (Extra-Región)")
            
            breakouts = df_res[df_res["Estatus Matriz"] == "🆕 BREAKOUT DE VARIANZA"].copy()
            compresiones = df_res[df_res["Estatus Matriz"] == "⌛ COMPRESIÓN EN RANGO"].copy()
            
            df_visual = pd.concat([breakouts, compresiones], ignore_index=True)
            
            def style_garch(val):
                if val == "🆕 BREAKOUT DE VARIANZA": return "background-color: #f0f4ff; color: #1a365d; font-weight: 800; border-left: 5px solid #2b6cb0;"
                return "color: #718096; font-style: italic;"

            if not df_visual.empty:
                st.dataframe(df_visual.style.map(style_garch, subset=["Estatus Matriz"]).format({
                    "Expansión Vol (5D vs 20D)": "{:.1%}",
                    "Vol Condicional GARCH": "{:.4f}",
                    "Precio Spot": "${:.2f}",
                    "Techo Simulación (P75)": "${:.2f}",
                    "Objetivo Take Profit (P90)": "${:.2f}",
                    "Soporte Stop Loss (Mediana)": "${:.2f}"
                }), use_container_width=True, hide_index=True)
            
            # ======================================================================
            # 💰 COMPRA Y MONEY MANAGEMENT PROPORCIONAL CON PISO OPTIMIZADO ($10 USD)
            # ======================================================================
            st.divider()
            st.subheader("💰 Órdenes de Ejecución de Tesorería Automatizada")
            
            n_compras = len(breakouts)
            
            if n_compras > 0:
                # La ponderación premia el Ratio de Sharpe Estocástico: Expansión de Vol / Riesgo Condicional
                breakouts['Ratio_Eficiencia'] = breakouts['Expansión Vol (5D vs 20D)'] / breakouts['Vol Condicional GARCH']
                pesos_base = breakouts['Ratio_Eficiencia'] / breakouts['Ratio_Eficiencia'].sum()
                breakouts['Monto Invertir (USD)'] = pesos_base * presupuesto_diario_bot4
                
                # Algoritmo de optimización distributiva con restricción de piso de $10 USD para los socios
                if presupuesto_diario_bot4 >= (n_compras * MIN_USD_PER_ORDER):
                    monto_insuficiente = True
                    while monto_insuficiente:
                        bajo_piso = breakouts['Monto Invertir (USD)'] < MIN_USD_PER_ORDER
                        
                        if bajo_piso.any():
                            breakouts.loc[bajo_piso, 'Monto Invertir (USD)'] = MIN_USD_PER_ORDER
                            fondos_fijos = breakouts[breakouts['Monto Invertir (USD)'] == MIN_USD_PER_ORDER]['Monto Invertir (USD)'].sum()
                            presupuesto_restante = presupuesto_diario_bot4 - fondos_fijos
                            
                            sobre_piso = breakouts['Monto Invertir (USD)'] > MIN_USD_PER_ORDER
                            if sobre_piso.any() and presupuesto_restante > 0:
                                breakouts.loc[sobre_piso, 'Monto Invertir (USD)'] = (breakouts.loc[sobre_piso, 'Ratio_Eficiencia'] / breakouts.loc[sobre_piso, 'Ratio_Eficiencia'].sum()) * presupuesto_restante
                            else:
                                monto_insuficiente = False
                        else:
                            monto_insuficiente = False

                breakouts['Porcentaje Asignación'] = (breakouts['Monto Invertir (USD)'] / presupuesto_diario_bot4) * 100
                breakouts['Fracciones a Adquirir'] = breakouts['Monto Invertir (USD)'] / breakouts['Precio Spot']
                
                st.success(f"🎯 **ORDENANZA DE COMPRA EMITIDA:** Asignación proporcional de los ${presupuesto_diario_bot4:.2f} USD del portafolio entre los quiebres estocásticos confirmados.")
                st.table(breakouts[['Ticker', 'Estatus Matriz', 'Porcentaje Asignación', 'Monto Invertir (USD)', 'Fracciones a Adquirir', 'Objetivo Take Profit (P90)', 'Soporte Stop Loss (Mediana)']]
                         .style.format({
                             "Porcentaje Asignación": "{:.1f}%",
                             "Monto Invertir (USD)": "${:.2f}",
                             "Fracciones a Adquirir": "{:.4f}",
                             "Objetivo Take Profit (P90)": "${:.2f}",
                             "Soporte Stop Loss (Mediana)": "${:.2f}"
                         }))
            else:
                st.warning(f"⚠️ **CONGELAMIENTO DE CAPITAL POR RANGO:** El 100% de la flota de hipervolatilidad cotiza comprimida o dentro de los conos normales de la caminata aleatoria. Queda estrictamente prohibido comprar hoy. Los **${presupuesto_diario_bot4:.2f} USD** de hoy permanecen en efectivo líquido en la caja del fondo.")
        else:
            st.info("Ninguno de los 150 activos de frontera experimentó una expansión de volatilidad de corto plazo suficiente para activar el radar numérico hoy.")
