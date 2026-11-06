# Manutenção Preditiva - Projeto do Módulo 1 (SCTEC/SENAI)
# Autor: Helyton Ronchi
# Ideia: escolher o modelo pelo CUSTO de errar, não só pela métrica.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------
# 0. Quanto custa cada erro (valores que EU defini, é uma suposição)
# Máquina que quebra sem aviso custa 10x mais que uma inspeção à toa
# ---------------------------------------------------------
CUSTO_FALHA_NAO_PREVISTA = 10   # falso negativo
CUSTO_ALARME_FALSO = 1          # falso positivo


def calcular_custo(y_real, y_previsto):
    tn, fp, fn, tp = confusion_matrix(y_real, y_previsto).ravel()
    return fn * CUSTO_FALHA_NAO_PREVISTA + fp * CUSTO_ALARME_FALSO


# ---------------------------------------------------------
# 1. Carregar e limpar os dados
# ---------------------------------------------------------
df = pd.read_csv('manutencao_preditiva.csv')
print("Nulos por coluna:")
print(df.isna().sum())

# Essas colunas já dizem o tipo da falha, então entregam a resposta (vazamento)
colunas_vazamento = ['falha_twf', 'falha_hdf', 'falha_pwf', 'falha_osf', 'falha_rnf']
df = df.drop(columns=colunas_vazamento).dropna()

# ---------------------------------------------------------
# 2. Transformar 'tipo' (L, M, H) em número
# ---------------------------------------------------------
le = LabelEncoder()
df['tipo'] = le.fit_transform(df['tipo'])

# ---------------------------------------------------------
# 3. Separar X (entradas) e y (o que quero prever)
# ---------------------------------------------------------
X = df.drop(columns=['udi', 'id_produto', 'falha_maquina'])
y = df['falha_maquina']

# ---------------------------------------------------------
# 4. Dividir em 3 partes: treino, validação e teste
# Validação = onde eu escolho o modelo. Teste = só no final, uma vez.
# ---------------------------------------------------------
X_resto, X_teste, y_resto, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_treino, X_valid, y_treino, y_valid = train_test_split(
    X_resto, y_resto, test_size=0.25, random_state=42, stratify=y_resto)

# ---------------------------------------------------------
# 5. SMOTE só no treino (valid e teste ficam com a vida real)
# ---------------------------------------------------------
smote = SMOTE(random_state=42)
X_treino_bal, y_treino_bal = smote.fit_resample(X_treino, y_treino)

# ---------------------------------------------------------
# 6. KNN (agora com escala, senão rpm "engole" as outras colunas)
# ---------------------------------------------------------
escala = StandardScaler()
X_treino_esc = escala.fit_transform(X_treino_bal)
X_valid_esc = escala.transform(X_valid)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_treino_esc, y_treino_bal)
pred_knn = knn.predict(X_valid_esc)
custo_knn = calcular_custo(y_valid, pred_knn)
print("\nKNN -> recall:", round(recall_score(y_valid, pred_knn), 3),
      "| custo:", custo_knn)

# ---------------------------------------------------------
# 7. Árvore: testar várias profundidades na VALIDAÇÃO
# ---------------------------------------------------------
resultados = []
for prof in [3, 5, 7, 10, 15, 20, None]:
    arvore = DecisionTreeClassifier(max_depth=prof, random_state=42)
    arvore.fit(X_treino_bal, y_treino_bal)
    pred = arvore.predict(X_valid)
    resultados.append({
        'profundidade': prof,
        'recall': round(recall_score(y_valid, pred), 3),
        'custo': calcular_custo(y_valid, pred),
    })

tabela = pd.DataFrame(resultados)
print("\n--- Profundidades (validação) ---")
print(tabela.to_string(index=False))

# Melhor = menor custo
melhor_prof = tabela.sort_values('custo').iloc[0]['profundidade']
if pd.isna(melhor_prof):
    melhor_prof = None
else:
    melhor_prof = int(melhor_prof)
print("\nProfundidade escolhida (menor custo):", melhor_prof)

# ---------------------------------------------------------
# 8. Modelo final, avaliado no TESTE uma única vez
# ---------------------------------------------------------
arvore_final = DecisionTreeClassifier(max_depth=melhor_prof, random_state=42)
arvore_final.fit(X_treino_bal, y_treino_bal)

pred_treino = arvore_final.predict(X_treino)
pred_teste = arvore_final.predict(X_teste)

print("\n=== RESULTADO FINAL NO TESTE ===")
print(classification_report(y_teste, pred_teste))
print("Recall no treino:", round(recall_score(y_treino, pred_treino), 3))
print("Recall no teste: ", round(recall_score(y_teste, pred_teste), 3))
print("Custo no teste:  ", calcular_custo(y_teste, pred_teste))

# ---------------------------------------------------------
# 9. Gráficos
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_teste, pred_teste), annot=True, fmt='d', cmap='Greens')
plt.title('Matriz de Confusão - Árvore (teste)')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.savefig('matriz_final.png')

importancias = pd.DataFrame({
    'Variavel': X.columns,
    'Importancia': arvore_final.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("\n--- Top 5 variáveis ---")
print(importancias.head(5).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variavel', data=importancias,
            hue='Variavel', legend=False, palette='viridis')
plt.title('Importância das Variáveis')
plt.tight_layout()
plt.savefig('importancia_variaveis.png')

print("\nPronto. Gráficos salvos.")
