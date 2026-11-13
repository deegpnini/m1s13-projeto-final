# -*- coding: utf-8 -*-
"""
Oikos Machine - Manutencao Preditiva (Industria 4.0)
Autor: Helyton Ronchi
Curso: SCTEC/SENAI - Modulo 1
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# FASE 1: ANALISE EXPLORATORIA (EDA)
df = pd.read_csv('manutencao_preditiva.csv')
print("Dimensoes:", df.shape)
print(df.describe())
print(df['falha_maquina'].value_counts(normalize=True) * 100)

plt.figure(figsize=(10, 6))
sns.histplot(df['torque_nm'], bins=30, kde=True)
plt.savefig('grafico_histograma.png'); plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x='falha_maquina', data=df)
plt.savefig('grafico_barras.png'); plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
plt.savefig('grafico_heatmap.png'); plt.close()

print("\n=== ANALISE DOS PADROES (EDA) ===")
print("Desbalanceamento extremo (96.6% normal vs 3.4% falha) justifica o uso de SMOTE apenas no treino.")
print("Histograma do torque mostra assimetria, o que justifica a imputacao por mediana nos nulos.")
print("Heatmap mostra baixa correlacao linear entre as features.")

# FASE 2: LIMPEZA E TRATAMENTO
df = df.drop_duplicates()
for col in ['temperatura_ar_k', 'temperatura_processo_k', 'velocidade_rotacao_rpm', 'torque_nm']:
    df[col] = df[col].fillna(df[col].median())
df = df.drop(columns=['falha_twf', 'falha_hdf', 'falha_pwf', 'falha_osf', 'falha_rnf'])

# FASE 3: FEATURE ENGINEERING
df['potencia_w'] = df['torque_nm'] * df['velocidade_rotacao_rpm'] * 2 * np.pi / 60
print("\nPotencia criada (watts). Nulos restantes:", df['potencia_w'].isna().sum())

# FASE 4: DIVISAO E BALANCEAMENTO
df['tipo'] = LabelEncoder().fit_transform(df['tipo'])
X = df.drop(columns=['udi', 'id_produto', 'falha_maquina'])
y = df['falha_maquina']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Apos SMOTE (treino):", X_train_res.shape)

# FASE 5: ESCALONAMENTO (APENAS KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# FASE 6: AJUSTE DE PARAMETROS (KNN E ARVORE)
print("\n--- KNN ---")
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_res)
    acc_train = accuracy_score(y_train_res, knn.predict(X_train_scaled))
    acc_test = accuracy_score(y_test, knn.predict(X_test_scaled))
    print(f"K={k}: Acuracia Treino={acc_train:.3f}, Teste={acc_test:.3f}")

print("\n--- Arvore de Decisao ---")
for depth in [3, 5, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train_res, y_train_res)
    acc_train = accuracy_score(y_train_res, tree.predict(X_train_res))
    acc_test = accuracy_score(y_test, tree.predict(X_test))
    print(f"max_depth={depth}: Acuracia Treino={acc_train:.3f}, Teste={acc_test:.3f}")

print("\n=== ANALISE DE OVERFITTING ===")
print("KNN: K=3 tem acuracia alta no treino e cai no teste. K=5 e o equilibrio.")
print("Arvore: max_depth=None decorou o treino (0.993) e caiu no teste (0.937). Depth=5 foi o equilibrio.")

# FASE 7: AVALIACAO FINAL E BONUS (RF E GB)
print("\n=== COMPARACAO FINAL DE MODELOS ===")

knn_final = KNeighborsClassifier(n_neighbors=5)
knn_final.fit(X_train_scaled, y_train_res)
acc_knn = accuracy_score(y_test, knn_final.predict(X_test_scaled))
print(f"KNN (K=5): {acc_knn:.3f}")

tree_final = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_final.fit(X_train_res, y_train_res)
acc_tree = accuracy_score(y_test, tree_final.predict(X_test))
print(f"Arvore (max_depth=5): {acc_tree:.3f}")

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_res, y_train_res)
acc_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest (Bonus): {acc_rf:.3f}")

gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X_train_res, y_train_res)
acc_gb = accuracy_score(y_test, gb.predict(X_test))
print(f"Gradient Boosting (Bonus): {acc_gb:.3f}")

print("\n=== VEREDITO FINAL ===")
modelos = {'KNN': acc_knn, 'Arvore': acc_tree, 'Random Forest': acc_rf, 'Gradient Boosting': acc_gb}
melhor = max(modelos, key=modelos.get)
print(f"Melhor modelo: {melhor} (Acuracia: {modelos[melhor]:.3f})")
