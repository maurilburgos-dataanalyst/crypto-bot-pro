import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import requests

# --- CONFIGURACIÓN DE PÁGINA (DEBE SER LO PRIMERO) ---
st.set_page_config(page_title="Crypto Bot Pro", layout="wide")

# --- FUNCIÓN TELEGRAM (SEGURA CON SECRETS) ---
def enviar_telegram(mensaje):
    # Busca las claves en la configuración segura de Streamlit Cloud
    try:
        bot_token = st.secrets["TELEGRAM_TOKEN"] 
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        params = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error de configuración Telegram: {e}")
        return False

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧠 Bot Full Stack: IA + Sentimiento (Fear & Greed)")
st.markdown("""
Estrategia Multinivel:
1. **Técnico:** XGBoost (Tendencia) + ADX (Fuerza).
2. **Psicológico:** Fear & Greed Index (Ajuste de apuesta).
3. **Gestión:** Criterio de Kelly (Tamaño) + Trailing Stop (Salida).
""")

# --- SIDEBAR: CONFIGURACIÓN (OPTIMIZADA) ---
st.sidebar.header("Configuración")
ticker = st.sidebar.text_input("Ticker", value="ETH-USD")
capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=1000)

st.sidebar.subheader("Parámetros IA (Tus Ganadores)")
# Valores por defecto actualizados a tu "Sweet Spot"
sma_s = st.sidebar.slider("SMA Corta", 10, 50, 28)
sma_l = st.sidebar.slider("SMA Larga", 50, 200, 69)
trailing_stop_pct = st.sidebar.slider("Trailing Stop (%)", 0.01, 0.15, 0.07)
adx_threshold = st.sidebar.slider("Filtro ADX", 10, 50, 33)

st.sidebar.subheader("Psicología de Mercado")
usar_sentimiento = st.sidebar.checkbox("Activar Análisis de Sentimiento", value=True)
umbral_panico = st.sidebar.slider("Comprar más en Pánico (< X)", 10, 40, 30)
umbral_codicia = st.sidebar.slider("Cuidarse en Codicia (> X)", 60, 90, 80)

# --- 1. FUNCIÓN DE PRECIOS (YAHOO) ---
@st.cache_data
def descargar_precios(ticker):
    data = yf.download(ticker, start="2018-01-01", interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# --- 2. FUNCIÓN DE SENTIMIENTO (API EXTERNA) ---
@st.cache_data
def descargar_fng():
    # Bajamos la historia del Fear & Greed Index
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        response = requests.get(url)
        data = response.json()['data']
        
        df_fng = pd.DataFrame(data)
        df_fng['value'] = df_fng['value'].astype(int)
        df_fng['timestamp'] = pd.to_datetime(df_fng['timestamp'], unit='s')
        
        # Ajustamos el índice para poder cruzarlo con Yahoo Finance
        df_fng = df_fng.set_index('timestamp')
        df_fng = df_fng.sort_index()
        return df_fng['value'] 
    except Exception as e:
        st.warning("No se pudo conectar a la API de Sentimiento.")
        return pd.Series()

# --- PROCESAMIENTO ---
data = descargar_precios(ticker)

if usar_sentimiento:
    try:
        fng_series = descargar_fng()
        if not fng_series.empty:
            # Cruzamos los datos (Merge) por fecha. 
            data = pd.merge_asof(data.sort_index(), fng_series.sort_index(), left_index=True, right_index=True, direction='backward')
            data = data.rename(columns={'value': 'FNG'})
            # Rellenamos huecos
            data['FNG'] = data['FNG'].fillna(method='ffill').fillna(50)
            st.success("✅ Datos de Sentimiento Integrados")
        else:
            data['FNG'] = 50
    except Exception as e:
        data['FNG'] = 50

# --- INDICADORES ---
data["SMA_S"] = data["Close"].rolling(window=sma_s).mean()
data["SMA_L"] = data["Close"].rolling(window=sma_l).mean()
data["Signal_Cruce"] = np.where(data["SMA_S"] > data["SMA_L"], 1, 0)

# RSI
delta = data["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# ADX
def calcular_adx(df):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame(df['High'] - df['Low'])
    tr2 = pd.DataFrame(abs(df['High'] - df['Close'].shift(1)))
    tr3 = pd.DataFrame(abs(df['Low'] - df['Close'].shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    return dx.rolling(14).mean()

data["ADX"] = calcular_adx(data)
data = data.dropna()

# --- ENTRENAMIENTO IA ---
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
features = ["SMA_S", "SMA_L", "Signal_Cruce", "RSI", "ADX"] 
X = data[features]
y = data["Target"]

split = int(len(data) * 0.75)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# --- BACKTEST (SIMULACIÓN) ---
test_data = data.iloc[split:].copy()
probs = model.predict_proba(X_test)[:, 1]
test_data['Probabilidad_Suba'] = probs

capital = capital_inicial
posicion = 0
cantidad_btc = 0
max_precio = 0
historial_capital = []
fng_hist = []

precios = test_data['Close'].values
probs_arr = test_data['Probabilidad_Suba'].values
adx_vals = test_data['ADX'].values
fng_vals = test_data['FNG'].values if 'FNG' in test_data.columns else [50]*len(precios)

for i in range(len(precios) - 1):
    precio = precios[i]
    prob = probs_arr[i]
    adx = adx_vals[i]
    fng = fng_vals[i]
    
    if posicion == 0:
        if prob > 0.5 and adx > adx_threshold:
            # KELLY
            b = 2.0
            p = prob
            q = 1 - p
            f = (b * p - q) / b
            if f < 0: f = 0
            
            # AJUSTE POR SENTIMIENTO
            factor_psicologico = 1.0
            if usar_sentimiento:
                if fng < umbral_panico:
                    factor_psicologico = 1.3 
                elif fng > umbral_codicia:
                    factor_psicologico = 0.5 
            
            f = f * 0.5 * factor_psicologico 
            f = min(f, 0.8) 
            
            dinero = capital * f
            if dinero > 10:
                cantidad_btc = dinero / precio
                capital -= dinero
                posicion = 1
                max_precio = precio
    
    elif posicion == 1:
        if precio > max_precio:
            max_precio = precio
        
        stop_actual = trailing_stop_pct
        if usar_sentimiento and fng < 20:
             stop_actual = stop_actual * 1.5 
             
        precio_salida = max_precio * (1 - stop_actual)
        
        if precio < precio_salida:
            capital += cantidad_btc * precio
            posicion = 0
            cantidad_btc = 0
            
    val = capital if posicion == 0 else (cantidad_btc * precio)
    historial_capital.append(val)
    fng_hist.append(fng)

# --- VISUALIZACIÓN ---
capital_final = historial_capital[-1]
retorno = ((capital_final - capital_inicial) / capital_inicial) * 100

col1, col2 = st.columns(2)
col1.metric("Capital Final", f"${capital_final:.2f}", f"{retorno:.2f}%")
ultimo_fng = fng_hist[-1]
estado_mercado = "😨 Miedo" if ultimo_fng < 40 else "😎 Codicia" if ultimo_fng > 60 else "😐 Neutral"
col2.metric("Sentimiento Mercado Hoy", f"{ultimo_fng:.0f}/100", estado_mercado)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(historial_capital, color='#00ff00', label='Estrategia IA + FNG')
ax1.set_ylabel("Capital", color='white')
ax1.set_facecolor('black')
fig.patch.set_facecolor('black')
ax1.tick_params(colors='white')
ax1.grid(alpha=0.2)

ax2 = ax1.twinx()
ax2.fill_between(range(len(fng_hist)), fng_hist, color='purple', alpha=0.1, label='Nivel de Codicia')
ax2.set_ylim(0, 100)
ax2.axhline(umbral_panico, color='cyan', linestyle='--', alpha=0.3)
ax2.axhline(umbral_codicia, color='red', linestyle='--', alpha=0.3)
st.pyplot(fig)

# --- NOTIFICACIONES TELEGRAM ---
st.sidebar.markdown("---")
st.sidebar.header("📱 Notificaciones")

if st.sidebar.button("📢 Chequear y Avisar por Telegram"):
    ultima_fila = data.iloc[-1]
    precio_hoy = ultima_fila['Close']
    rsi_hoy = ultima_fila['RSI']
    adx_hoy = ultima_fila['ADX']
    fng_hoy = ultima_fila['FNG'] if 'FNG' in ultima_fila else 50
    
    # Predecir HOY
    features_hoy = pd.DataFrame([ultima_fila[features]]) 
    probabilidad = model.predict_proba(features_hoy)[0][1]
    senal = 1 if probabilidad > 0.5 else 0
    
    icono = "🟢 COMPRA" if senal == 1 else "🔴 ESPERAR/VENTA"
    mensaje = (
        f"🤖 *REPORTE CRYPTO BOT*\n"
        f"Ticker: {ticker}\n"
        f"Precio: ${precio_hoy:.2f}\n\n"
        f"{icono} (Confianza IA: {probabilidad*100:.1f}%)\n\n"
        f"📊 *Indicadores:*\n"
        f"- RSI: {rsi_hoy:.1f}\n"
        f"- ADX: {adx_hoy:.1f}\n"
        f"- Sentimiento: {fng_hoy:.0f}/100"
    )
    
    with st.spinner("Enviando mensaje a Telegram..."):
        exito = enviar_telegram(mensaje)
    
    if exito:
        st.sidebar.success("✅ ¡Mensaje enviado!")
    else:
        st.sidebar.error("❌ Error al enviar. Verificá los Secrets.")