
# 🏭 Manutenção Preditiva — Pipeline de Falhas Mecânicas (Indústria 4.0)

## 📌 Objetivo
Este projeto foi desenvolvido como avaliação final do Módulo 1 da formação **SCTEC/SENAI (Desenvolvimento de IA para Análise Preditiva)**. O objetivo é construir um pipeline preditivo capaz de identificar falhas mecânicas em máquinas industriais antes que elas aconteçam, reduzindo custos de manutenção e paradas não planejadas.

## 📊 O Dataset
- **Fonte:** Dataset de Manutenção Preditiva (10.000 amostras).
- **Variável Alvo:** `falha_maquina` (0 = Normal, 1 = Falha).
- **Features:** Temperatura do ar, temperatura do processo, velocidade de rotação, torque e desgaste da ferramenta.
- **Desafio:** O dataset era altamente desbalanceado (96,6% normal vs 3,4% falha).

## 🛠️ Metodologia (Pipeline)
1. **Limpeza:** Remoção de 500 linhas com valores nulos.
2. **Prevenção de Data Leakage:** Remoção das colunas `falha_twf`, `falha_hdf`, etc. (que revelam o motivo da quebra e não devem ser usadas no treino).
3. **Label Encoding:** Transformação da coluna `tipo` (L, M, H) em variáveis numéricas.
4. **Balanceamento:** Aplicação do **SMOTE** (Synthetic Minority Over-sampling Technique) para equilibrar o treino (50/50).
5. **Modelagem:** Treinamento de **KNN** e **Árvore de Decisão** (com ajuste de hiperparâmetros de profundidade).

## 🏆 Resultados Finais (Árvore de Decisão — Prof. 15)
- **Acurácia:** 95%
- **Recall (Falhas):** 83% (detecta 83% das falhas reais)
- **Precisão (Falhas):** 41%
- **F1-Score (Falhas):** 55%

## 🔍 Top 5 Variáveis Mais Importantes
1. `velocidade_rotacao_rpm` (35%)
2. `torque_nm` (28%)
3. `desgaste_ferramenta_min` (23%)
4. `temperatura_ar_k` (4,8%)
5. `temperatura_processo_k` (4,6%)

## 💡 Aprendizados (Honestidade Radical)
- O modelo **KNN** teve um desempenho inicial inferior, com muitos falsos positivos.
- A **Árvore de Decisão** com profundidade 15 foi o melhor modelo, equilibrando recall e precisão.
- O SMOTE foi essencial para que o modelo não ignorasse a classe minoritária (falha).

## 🧰 Tecnologias Utilizadas
- Python 3.14
- Pandas, NumPy
- Scikit-Learn (Árvore de Decisão, KNN, Train/Test Split)
- Imbalanced-Learn (SMOTE)
- Matplotlib, Seaborn

---
*Desenvolvido por Helyton Ronchi (Comandante Hebron) — 06/11/2026*
