def analizar_breakout_estocastico(ticker, umbral_vol, pct_entrada):
    try:
        # Descargamos asegurando un formato plano de columnas
        df = yf.download(ticker, period="6m", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 40: return None
        
        # CORRECCIÓN DE MULTI-INDEX: Forzar a que encuentre la columna de cierre limpio
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        precios = df['Close'].dropna()
        retornos = precios.pct_change().dropna().values
        
        # ------------------------------------------------------------------
        # PASO 1: FILTRO OPERATIVO RÁPIDO (Desviaciones de Corto vs Mediano Plazo)
        # ------------------------------------------------------------------
        vol_corta = np.std(retornos[-5:])
        vol_mediana = np.std(retornos[-20:])
        
        if vol_mediana == 0: return None
        expansion_real = (vol_corta - vol_mediana) / vol_mediana
        
        if expansion_real < umbral_vol:
            return None
            
        # ------------------------------------------------------------------
        # PASO 2: AJUSTE GARCH(1,1) MLE & SIMULACIÓN MONTECARLO
        # ------------------------------------------------------------------
        vol_condicional_garch, alpha_opt, beta_opt = estimar_garch_mle(retornos)
        precio_spot = float(precios.iloc[-1])
        
        # Correr las 10,000 proyecciones estocásticas con el riesgo ajustado
        precios_finales_simulados = ejecutar_montecarlo(precio_spot, vol_condicional_garch, HORIZONTE_DIAS, N_SIMULACIONES)
        
        # Extraer los percentiles reales de la simulación
        p_techo_breakout = np.percentile(precios_finales_simulados, pct_entrada * 100)
        p_mediana = np.percentile(precios_finales_simulados, 50)
        p_take_profit = np.percentile(precios_finales_simulados, 90)
        
        # CORRECCIÓN DE LA PARADOJA GEOMÉTRICA: 
        # El breakout se confirma si el precio spot actual es superior al precio spot de hace 5 días 
        # más un umbral derivado de la volatilidad condicional calculada por el modelo GARCH.
        precio_anterior = float(precios.iloc[-5])
        rendimiento_reciente = (precio_spot - precio_anterior) / precio_anterior
        
        # Si el rendimiento de los últimos 5 días supera la volatilidad condicional esperada por el percentil
        condicion_breakout = rendimiento_reciente >= (vol_condicional_garch * np.sqrt(HORIZONTE_DIAS))
        
        return {
            "Ticker": ticker,
            "Precio Spot": round(precio_spot, 2),
            "Expansión Vol (5D vs 20D)": expansion_real,
            "Vol Condicional GARCH": vol_condicional_garch,
            "Alpha (ARCH)": alpha_opt,
            "Beta (GARCH)": beta_opt,
            "Techo Simulación (P60)": round(p_techo_breakout, 2),
            "Objetivo Take Profit (P90)": round(p_take_profit, 2),
            "Soporte Stop Loss (Mediana)": round(p_mediana, 2),
            "Estatus Matriz": "🆕 BREAKOUT DE VARIANZA" if condicion_breakout else "⌛ COMPRESIÓN EN RANGO"
        }
    except Exception as e:
        # Imprime el error real en la consola de Streamlit para saber si algo más truena
        st.sidebar.error(f"Error en {ticker}: {str(e)}")
        return None
