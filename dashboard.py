# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Monitoramento de Máquina", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.alerta-ok { background: #0f3d2e; border: 2px solid #22c55e; border-radius: 12px; padding: 1.2rem 1.4rem; color: #bbf7d0; }
.alerta-falha { background: #3f1d1d; border: 2px solid #ef4444; border-radius: 12px; padding: 1.2rem 1.4rem; color: #fecaca; }
.alerta-titulo { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.4rem; }
.alerta-texto { font-size: 1.05rem; line-height: 1.45; }
.pill { display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; margin-top: 0.6rem; }
.pill-ok { background: #166534; color: #dcfce7; }
.pill-falha { background: #991b1b; color: #fee2e2; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Treinando modelo...")
def treinar_modelo():
    df = pd.read_csv("manutencao_preditiva.csv")
    vazamento = ["falha_twf", "falha_hdf", "falha_pwf", "falha_osf", "falha_rnf"]
    df = df.drop(columns=[c for c in vazamento if c in df.columns], errors="ignore")
    for col in ["temperatura_ar_k", "temperatura_processo_k", "velocidade_rotacao_rpm", "torque_nm", "desgaste_ferramenta_min"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=["falha_maquina"])
    le = LabelEncoder()
    df["tipo"] = le.fit_transform(df["tipo"].astype(str))
    df["potencia_w"] = df["torque_nm"] * df["velocidade_rotacao_rpm"]
    colunas_X = ["tipo", "temperatura_ar_k", "temperatura_processo_k", "velocidade_rotacao_rpm", "torque_nm", "desgaste_ferramenta_min", "potencia_w"]
    X = df[colunas_X]
    y = df["falha_maquina"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo.fit(X_train_res, y_train_res)
    return modelo, colunas_X

modelo, colunas_X = treinar_modelo()

st.title("Monitoramento de máquina")
st.caption("Projeto M1S13 · Árvore de decisão (max_depth=5) · Ajuste os sensores à esquerda e leia o painel à direita.")
st.info("Demonstração. O modelo foi treinado no dataset do módulo (falhas raras ~3,4%). Acurácia de teste: 0,902. Alarme pode errar.")

col_sensores, col_painel = st.columns([1.05, 1.2], gap="large")

with col_sensores:
    st.subheader("Sensores (entrada)")
    torque = st.slider("Torque (Nm)", 0.0, 80.0, 40.0, 0.5)
    rpm = st.slider("Rotação (rpm)", 1000.0, 3000.0, 1500.0, 10.0)
    desgaste = st.slider("Desgaste da ferramenta (min)", 0.0, 250.0, 100.0, 1.0)
    temperatura_ar = st.slider("Temperatura do ar (K)", 295.0, 305.0, 300.0, 0.1)
    temperatura_processo = st.slider("Temperatura do processo (K)", 305.0, 314.0, 310.0, 0.1)
    tipo = st.selectbox("Tipo da máquina", ["L", "M", "H"], index=0)
    st.caption("L / M / H = categorias do dataset.")

with col_painel:
    st.subheader("Painel de status")
    potencia = torque * rpm
    tipo_num = {"L": 0, "M": 1, "H": 2}[tipo]
    entrada = pd.DataFrame([[tipo_num, temperatura_ar, temperatura_processo, rpm, torque, desgaste, potencia]], columns=colunas_X)
    previsao = int(modelo.predict(entrada)[0])
    proba_falha = None
    if hasattr(modelo, "predict_proba"):
        proba_falha = float(modelo.predict_proba(entrada)[0][1]) * 100
    k1, k2, k3 = st.columns(3)
    k1.metric("Potência (W)", f"{potencia:,.0f}".replace(",", "."))
    k2.metric("Torque × RPM", f"{torque:.0f} × {rpm:.0f}")
    k3.metric("Chance de falha", f"{proba_falha:.0f}%" if proba_falha is not None else "—")
    st.write("")
    if previsao == 1:
        st.markdown(f'<div class="alerta-falha"><div class="alerta-titulo">ALERTA DE FALHA</div><div class="alerta-texto">O modelo classifica esta combinação como <strong>risco de falha</strong>.<br><br><strong>O que fazer:</strong> reduzir carga (torque/rotação), checar desgaste e temperatura do processo.</div><span class="pill pill-falha">Status: FALHA PREVISTA</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alerta-ok"><div class="alerta-titulo">OPERAÇÃO NORMAL</div><div class="alerta-texto">Com os valores atuais, o modelo não indica falha. Continua monitorando torque, rotação e desgaste.</div><span class="pill pill-ok">Status: NORMAL</span></div>', unsafe_allow_html=True)
    st.write("")
    with st.expander("Leitura dos sensores neste momento"):
        st.table(pd.DataFrame({"Sensor": ["Tipo", "Torque (Nm)", "Rotação (rpm)", "Potência (W)", "Desgaste (min)", "Temp. ar (K)", "Temp. processo (K)"], "Valor": [tipo, f"{torque:.1f}", f"{rpm:.0f}", f"{potencia:.0f}", f"{desgaste:.0f}", f"{temperatura_ar:.1f}", f"{temperatura_processo:.1f}"]}))

st.divider()
st.markdown("**Modelo:** Árvore de Decisão (`max_depth=5`) · **Acurácia (teste):** 0,902 · **Script oficial:** `pipeline_m1s13.py`")
