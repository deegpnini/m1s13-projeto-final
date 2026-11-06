import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Monitoramento de Máquina", layout="wide")

@st.cache_data
def treinar_modelo():
    df = pd.read_csv('manutencao_preditiva.csv')
    df = df.dropna()
    le = LabelEncoder()
    df['tipo'] = le.fit_transform(df['tipo'])
    df['potencia_w'] = df['torque_nm'] * df['velocidade_rotacao_rpm']
    colunas_X = ['tipo', 'temperatura_ar_k', 'temperatura_processo_k', 'velocidade_rotacao_rpm', 'torque_nm', 'desgaste_ferramenta_min', 'potencia_w']
    X = df[colunas_X]
    y = df['falha_maquina']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo.fit(X_train_res, y_train_res)
    return modelo, colunas_X

modelo, colunas_X = treinar_modelo()

st.title("🏭 Monitoramento de Máquina — Previsão de Falha")
st.markdown("Ajuste os sensores e veja o modelo prever se a máquina vai falhar.")

col1, col2 = st.columns(2)

with col1:
    torque = st.slider("Torque (Nm)", 0.0, 80.0, 40.0)
    rpm = st.slider("Velocidade de Rotação (rpm)", 1000.0, 3000.0, 1500.0)
    desgaste = st.slider("Desgaste da Ferramenta (min)", 0.0, 250.0, 100.0)
    temperatura_ar = st.slider("Temperatura do Ar (K)", 295.0, 305.0, 300.0)
    temperatura_processo = st.slider("Temperatura do Processo (K)", 305.0, 314.0, 310.0)
    tipo = st.selectbox("Tipo da Máquina", ["L", "M", "H"])

with col2:
    potencia = torque * rpm
    st.metric("Potência Calculada (W)", f"{potencia:.1f}")
    tipo_num = 0 if tipo == "L" else (1 if tipo == "M" else 2)
    entrada = pd.DataFrame([[tipo_num, temperatura_ar, temperatura_processo, rpm, torque, desgaste, potencia]], columns=colunas_X)
    previsao = modelo.predict(entrada)[0]
    if previsao == 1:
        st.error("⚠️ ALERTA: A máquina vai falhar! Pare a linha e inspecione.")
        st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=150)
    else:
        st.success("✅ Máquina normal. Operação segura.")
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=150)

st.markdown("---")
st.markdown("**Modelo:** Árvore de Decisão (max_depth=5) | **Acurácia:** 0,902")
