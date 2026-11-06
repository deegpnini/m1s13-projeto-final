# -*- coding: utf-8 -*-
"""
Projeto Avaliativo M1 - Desenvolvimento de IA para Análise Preditiva (SCTEC/SENAI)
Autor: Helyton Ronchi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# ==========================================
# FASE 1: ANÁLISE EXPLORATÓRIA (EDA)
# ==========================================
df = pd.read_csv('manutencao_preditiva.csv')
print("Dimensões:", df.shape)
print(df.describe())
print(df['falha_maquina'].value_counts(normalize=True) * 100)

# Gráfico 1: Histograma (distribuição das preditoras)
plt.figure(figsize=(10, 6))
sns.histplot(df['torque_nm'], bins=30, kde=True)
plt.title('Distribuição do Torque')
plt.savefig('grafico_histograma.png')
plt.close()

# Gráfico 2: Barras (desbalanceamento)
plt.figure(figsize=(6, 4))
sns.countplot(x='falha_maquina', data=df)
plt.title('Desbalanceamento (0=Normal, 1=Falha)')
plt.savefig('grafico_barras.png')
plt.close()

# Gráfico 3: Heatmap de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlação de Pearson')
plt.savefig('grafico_heatmap.png')
plt.close()

# ==========================================
# FASE 2: LIMPEZA E TRATAMENTO
# ==========================================
# Remover duplicatas
df = df.drop_duplicates()
print("Após remover duplicatas:", df.shape)

# Tratar nulos: imputação por MEDIANA (justificativa: variáveis com outliers)
for col in ['temperatura_ar_k', 'temperatura_processo_k', 'velocidade_rotacao_rpm', 'torque_nm']:
    df[col] = df[col].fillna(df[col].median())

# Boxplots para outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['torque_nm', 'desgaste_ferramenta_min']])
plt.title('Boxplot: Outliers')
plt.savefig('grafico_boxplot.png')
plt.close()

# Remover colunas de vazamento
colunas_vazamento = ['falha_twf', 'falha_hdf', 'falha_pwf', 'falha_osf', 'falha_rnf']
df = df.drop(columns=colunas_vazamento)

# ==========================================
# FASE 3: FEATURE ENGINEERING
# ==========================================
# Criar coluna de potência (torque * rpm), tratando nulos antes
df['potencia_w'] = df['torque_nm'] * df['velocidade_rotacao_rpm']
print("Potência criada. Nulos restantes:", df['potencia_w'].isna().sum())

# ==========================================
# FASE 4: DIVISÃO E BALANCEAMENTO
# ==========================================
le = LabelEncoder()
df['tipo'] = le.fit_transform(df['tipo'])

X = df.drop(columns=['udi', 'id_produto', 'falha_maquina'])
y = df['falha_maquina']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Após SMOTE (treino):", X_train_res.shape)

# ==========================================
# FASE 5: ESCALONAMENTO (APENAS KNN)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
# A Árvore não é escalonada porque é imune à escala (divide por limiares, não por distância).

# ==========================================
# FASE 6: AJUSTE DE PARÂMETROS E OVERFITTING
# ==========================================
# KNN: testar K = 3, 5, 7
print("\n--- KNN ---")
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_res)
    acc_train = accuracy_score(y_train_res, knn.predict(X_train_scaled))
    acc_test = accuracy_score(y_test, knn.predict(X_test_scaled))
    print(f"K={k}: Acurácia Treino={acc_train:.3f}, Teste={acc_test:.3f}")

# Árvore: testar max_depth = 3, 5, None
print("\n--- Árvore de Decisão ---")
for depth in [3, 5, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train_res, y_train_res)
    acc_train = accuracy_score(y_train_res, tree.predict(X_train_res))
    acc_test = accuracy_score(y_test, tree.predict(X_test))
    print(f"max_depth={depth}: Acurácia Treino={acc_train:.3f}, Teste={acc_test:.3f}")

# ==========================================
# FASE 7: AVALIAÇÃO FINAL E VEREDITO
# ==========================================
# Melhor KNN (K=5) e melhor Árvore (max_depth=5)
knn_final = KNeighborsClassifier(n_neighbors=5)
knn_final.fit(X_train_scaled, y_train_res)
acc_knn = accuracy_score(y_test, knn_final.predict(X_test_scaled))

tree_final = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_final.fit(X_train_res, y_train_res)
acc_tree = accuracy_score(y_test, tree_final.predict(X_test))

print(f"\n=== RESULTADO FINAL ===")
print(f"Acurácia KNN (K=5): {acc_knn:.3f}")
print(f"Acurácia Árvore (max_depth=5): {acc_tree:.3f}")
print(f"Modelo escolhido: {'KNN' if acc_knn > acc_tree else 'Árvore de Decisão'} (maior acurácia no teste)")
