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
st.set_page_config(page_title="The One Ring: Bot 4 GARCH-Montecarlo Intradía", layout="wide", page_icon="🔮")

st.title("🔮 Bot 4 — GARCH + Montecarlo Breakout Cones (INTRADÍA 1H)")
st.caption("Ecosistema Cuantitativo 'The One Ring' | Enfoque: Micro-Clústeres de Volatilidad y Arbitraje Estocástico en Barras de 60 Minutos")
st.markdown("---")

# RESTRICCIONES DE CONTROL INSTITUCIONAL ADJUSTADAS A INTRADÍA
MIN_USD_PER_ORDER = 10.0
N_SIMULACIONES = 10000  # Número de caminos aleatorios vectorizados
HORIZONTE_VELAS = 7     # Ventana cerrada de arbitraje (7 velas de 1H equivalen a 1 día completo de trading)

# BARRA LATERAL DE PARAMETRIZACIÓN - OPTIMIZADA PARA OPERACIONES RÁPIDAS
st.sidebar.header("⚙️ Control de Tesorería y Filtros")
presupuesto_diario_bot4 = st.sidebar.number_input("Presupuesto de Despliegue Hoy (USD)", value=550.0, step=50.0)
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Calibración de Filtros Intradía")
# Parámetros calibrados para la estructura de micro-rupturas intradiarias
percentil_entrada = st.sidebar.slider("Percentil de Entrada (Techo Estocástico)", 50, 75, 55) / 100
umbral_expansion_vol = st.sidebar.slider("Umbral de Expansión de Volatilidad Rápida (%)", 5, 35, 10) / 100

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
TICKERS = list(set(UNIVERSO_MOM_VOL))

# ==========================================================================
# 3. MOTOR MATEMÁTICO AVANZADO: OPTIMIZACIÓN MLE GARCH(1,1) t-STUDENT
# ==========================================================================
def loss_garch_t_student(parametros, retornos, df_t):
    omega, alpha, beta = parametros
    longitud = len(retornos)
    varianzas = np.zeros(longitud)
    varianzas[0] = np.var(retornos)
    
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e10
        
    for t in range(1, longitud):
        varianzas[t] = omega + alpha * (retornos[t-1]**2) + beta * varianzas[t-1]
    
    varianzas = np.where(varianzas <= 0, 1e-6, varianzas)
    residuos_estandarizados = retornos / np.sqrt(varianzas)
    log_verosimilitul = np.sum(t_student.logpdf(residuos_estandarizados, df=df_t, loc=0, scale=1) - 0.5 * np.log(varianzas))
    return -log_verosimilitul

def estimar_garch_mle(retornos):
    var_inicial = np.var(retornos)
    x0 = [0.05 * var_inicial, 0.08, 0.85]
    bounds = ((1e-8, None), (0.0, 0.3), (0.5, 0.98))
    df_t = 5.0 
    
    res = minimize(loss_garch_t_student, x0, args=(retornos, df_t), method='SLSQP', bounds=bounds)
    
    if res.success:
        omega, alpha, beta = res.x
    else:
        omega, alpha, beta = 0.05 * var_inicial, 0.10, 0.88
        
    varianzas = np.zeros(len(retornos))
    varianzas[0] = var_inicial
    for t in range(1, len(retornos)):
        varianzas[t] = omega + alpha * (retornos[t-1]**2) + beta * varianzas[t-1]
        
    proxima_vol = np.sqrt(omega + alpha * (retornos[-1]**2) + beta * varianzas[-1])
    return proxima_vol, alpha, beta

def ejecutar_montecarlo(precio_inicial, vol_garch, velas, n_sim):
    rendimientos_simulados = t_student.rvs(df=5, loc=0, scale=vol_garch, size=(velas, n_sim))
    cambios_porcentuales = np.exp(rendimientos_simulados)
    precios_proyectados = precio_inicial * np.cumprod(cambios_porcentuales, axis=0)
    return precios_proyectados[-1, :]

def analizar_breakout_estocastico(ticker, umbral_vol, pct_entrada):
    try:
        # CAMBIO CLAVE: Solicitamos 1 mes de histórico con resolución de 60 minutos (Intradía)
        df = yf.download(ticker, period="1mo", interval="60m", progress=False, auto_adjust=True)
        if df.empty or len(df) < 50: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        precios = df['Close'].dropna()
        retornos = precios.pct_change().dropna().values
        
        # FILTRO OPERATIVO INTRADÍA: 7 velas (1 día de mercado) vs 35 velas (5 días de mercado)
        vol_corta = np.std(retornos[-7:])
        vol_mediana = np.std(retornos[-35:])
        
        if vol_mediana == 0: return None
        expansion_real = (vol_corta - vol_mediana) / vol_mediana
        
        if expansion_real < umbral_vol:
            return None
            
        vol_condicional_garch, alpha_opt, beta_opt = estimar_garch_mle(retornos)
        precio_spot = float(precios.iloc[-1])
        
        # Montecarlo proyecta las próximas 7 velas horarias
        precios_finales_simulados = ejecutar_montecarlo(precio_spot, vol_condicional_garch, HORIZONTE_VELAS, N_SIMULACIONES)
        
        p_techo_breakout = np.percentile(precios_finales_simulados, pct_entrada * 100)
        p_mediana = np.percentile(precios_finales_simulados, 50)
        p_take_profit = np.percentile(precios_finales_simulados, 90)
        
        # EVALUACIÓN DE RUPTURA: Rendimiento de las últimas 7 horas vs Volatilidad esperada por el GARCH
        precio_anterior = float(precios.iloc[-7])
        rendimiento_reciente = (precio_spot - precio_anterior) / precio_anterior
        condicion_breakout = rendimiento_reciente >= (vol_condicional_garch * np.sqrt(HORIZONTE_VELAS))
        
        return {
            "Ticker": ticker,
            "Precio Spot": round(precio_spot, 2),
            "Expansión Vol (1D vs 5D)": expansion_real,
            "Vol Condicional GARCH (1H)": vol_condicional_garch,
            "Alpha (ARCH)": alpha_opt,
            "Beta (GARCH)": beta_opt,
            "Techo Simulación (1D)": round(p_techo_breakout, 2),
            "Objetivo Take Profit (P90)": round(p_take_profit, 2),
            "Soporte Stop Loss (Mediana)": round(p_mediana, 2),
            "Estatus Matriz": "🆕 BREAKOUT DE VARIANZA" if condicion_breakout else "⌛ COMPRESIÓN EN RANGO"
        }
    except Exception as e:
        st.sidebar.error(f"Error en {ticker}: {str(e)}")
        return None

# ==========================================================================
# 4. TABLERO DE CONTROL Y PANELES VISUALES DE ISENGARD
# ==========================================================================
if st.button("🚀 Iniciar Escaneo de Varianza Intradía (150 Tickers)"):
    with st.spinner("Descargando flujos de 60 min y optimizando funciones MLE numéricas..."):
        alertas = []
        
        for t in TICKERS:
            res = analizar_breakout_estocastico(t, umbral_expansion_vol, percentil_entrada)
            if res: alertas.append(res)
            
        if alertas:
            df_res = pd.DataFrame(alertas)
            st.subheader("📡 Detección de Regímenes de Momentum Anómalo Intradía")
            
            breakouts = df_res[df_res["Estatus Matriz"] == "🆕 BREAKOUT DE VARIANZA"].copy()
            compresiones = df_res[df_res["Estatus Matriz"] == "⌛ COMPRESIÓN EN RANGO"].copy()
            df_visual = pd.concat([breakouts, compresiones], ignore_index=True)
            
            def style_garch(val):
                if val == "🆕 BREAKOUT DE VARIANZA": return "background-color: #f0f4ff; color: #1a365d; font-weight: 800; border-left: 5px solid #2b6cb0;"
                return "color: #718096; font-style: italic;"

            if not df_visual.empty:
                st.dataframe(df_visual.style.map(style_garch, subset=["Estatus Matriz"]).format({
                    "Expansión Vol (1D vs 5D)": "{:.1%}",
                    "Vol Condicional GARCH (1H)": "{:.4f}",
                    "Alpha (ARCH)": "{:.4f}",
                    "Beta (GARCH)": "{:.4f}",
                    "Precio Spot": "${:.2f}",
                    "Techo Simulación (1D)": "${:.2f}",
                    "Objetivo Take Profit (P90)": "${:.2f}",
                    "Soporte Stop Loss (Mediana)": "${:.2f}"
                }), use_container_width=True, hide_index=True)
            
            st.divider()
            st.subheader("💰 Órdenes de Ejecución de Tesorería Automatizada")
            n_compras = len(breakouts)
            
            if n_compras > 0:
                breakouts['Ratio_Eficiencia'] = breakouts['Expansión Vol (1D vs 5D)'] / breakouts['Vol Condicional GARCH (1H)']
                pesos_base = breakouts['Ratio_Eficiencia'] / breakouts['Ratio_Eficiencia'].sum()
                breakouts['Monto Invertir (USD)'] = pesos_base * presupuesto_diario_bot4
                
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
                
                st.success(f"🎯 **ORDENANZA DE COMPRA EMITIDA:** Asignación de los ${presupuesto_diario_bot4:.2f} USD basada en la calibración GARCH Intradía.")
                st.table(breakouts[['Ticker', 'Estatus Matriz', 'Alpha (ARCH)', 'Beta (GARCH)', 'Porcentaje Asignación', 'Monto Invertir (USD)', 'Fracciones a Adquirir', 'Objetivo Take Profit (P90)']]
                           .style.format({
                               "Porcentaje Asignación": "{:.1f}%",
                               "Monto Invertir (USD)": "${:.2f}",
                               "Fracciones a Adquirir": "{:.4f}",
                               "Alpha (ARCH)": "{:.4f}",
                               "Beta (GARCH)": "{:.4f}",
                               "Objetivo Take Profit (P90)": "${:.2f}"
                           }))
            else:
                st.warning(f"⚠️ **CONGELAMIENTO DE CAPITAL POR RANGO:** El 100% de la flota de hipervolatilidad cotiza comprimida en rango intradiario. Queda prohibido comprar en esta barra de 60 minutos. Los **${presupuesto_diario_bot4:.2f} USD** permanecen en efectivo líquido.")
        else:
            st.info("Ninguno de los activos superó el filtro primario de expansión de varianza en la sesión actual.")
