# Oikos Machine: Manutenção Preditiva para a Indústria 4.0

**Autor:** Helyton Renato Gonçalves Ronchi (São Ludgero, SC)
**LinkedIn:** linkedin.com/in/helytonronchi
**Curso:** Desenvolvimento de IA para Análise Preditiva, T4 (SCTEC/SENAI)

**Arquivos principais:**
- `notebook_final.ipynb`: Notebook Jupyter completo com as 7 fases, as células de texto justificando as decisões e os gráficos.
- `pipeline_m1s13.py`: Versão em script Python com a mesma lógica, para rodar direto no terminal.

## 1. O Problema
Quando uma máquina quebra no meio do trabalho, a fábrica para e o prejuízo é grande. O objetivo deste projeto é construir um programa que leia os dados dos sensores (temperatura, velocidade, torque e desgaste) e avise antes que a máquina falhe.

A variável alvo é binária: `falha_maquina = 0` (normal) ou `1` (falha). O dataset tem 10.000 amostras, sendo que 96,6% são normais e apenas 3,4% são falhas. Esse desbalanceamento é o principal desafio do projeto.

## 2. Origem do Meu Olhar Técnico
Antes de migrar para a tecnologia, trabalhei com pintura, elétrica e serviços gerais. Essa experiência de campo orientou uma decisão do projeto: criar uma variável de potência (torque × velocidade de rotação), porque em máquina o que importa é a combinação das grandezas, e não cada leitura isolada. A potência foi calculada em watts de verdade (torque × rpm × 2π/60).

## 3. As 7 Fases do Pipeline

**Fase 1: Análise Exploratória (EDA)**
- Dimensões: 10.000 linhas × 14 colunas.
- Desbalanceamento: 96,61% normal vs 3,39% falha.
- Gráficos: histograma do torque, barras de desbalanceamento, heatmap de correlação.
- Análise: O gráfico de barras confirma o desbalanceamento extremo, o que me levou a usar SMOTE apenas no treino e a desconfiar da acurácia como métrica principal. O histograma do torque mostrou distribuição assimétrica, o que me levou a imputar nulos pela mediana.

**Fase 2: Limpeza e Tratamento**
- Duplicatas: removidas.
- Dados ausentes: imputação por mediana (a distribuição do torque tem outliers, e a mediana é mais robusta que a média).
- Vazamento: colunas falha_twf, falha_hdf, falha_pwf, falha_osf, falha_rnf removidas.

**Fase 3: Feature Engineering**
- Nova variável: potencia_w = torque_nm * velocidade_rotacao_rpm * 2 * π / 60 (watts de verdade).

**Fase 4: Divisão e Balanceamento**
- Split: 80/20 com stratify=y.
- SMOTE: apenas no treino (8.000 -> 15.458 amostras balanceadas).

**Fase 5: Escalonamento**
- KNN: StandardScaler aplicado apenas nas colunas contínuas (fit_transform no treino, transform no teste).
- Árvore: sem escalonamento (imune à escala).

**Fase 6: Ajuste de Parâmetros e Overfitting**
- KNN: testados K=3, 5 e 7.
- Árvore: testadas max_depth=3, 5 e None.
- Overfitting: max_depth=None decorou o treino (0.993) e caiu no teste (0.940). Depth=5 foi o equilíbrio.

**Fase 7: Avaliação Final e Veredito**
- Modelo obrigatório vencedor: **Árvore de Decisão (max_depth=5)**, com acurácia de **0.925** no teste, superando o KNN (0.910).

## 4. Comparação de Modelos (Bônus)
Além dos modelos obrigatórios (KNN e Árvore), testei dois modelos de ensemble (Random Forest e Gradient Boosting) para comparar o desempenho:

| Modelo | Acurácia (Teste) |
|---|---|
| KNN (K=5) | 0.910 |
| Árvore (max_depth=5) | 0.925 |
| Random Forest | 0.913 |
| **Gradient Boosting** | **0.928** |

**Veredito Final:** Entre os modelos obrigatórios, a Árvore de Decisão venceu o KNN. Como bônus, o Gradient Boosting apresentou o melhor desempenho geral (92,8%), combinando a robustez das árvores com o poder do boosting.

## 5. Como Executar o Projeto
Para rodar o pipeline principal:
git clone https://github.com/deegpnini/oikos-machine.git
cd oikos-machine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 pipeline_m1s13.py

Para rodar o notebook, basta abrir o arquivo notebook_final.ipynb no Jupyter ou no Google Colab.

## 6. Dashboard Interativo (Extra)
Criei um dashboard em Streamlit (dashboard.py) para mostrar o modelo operando uma maquina. O operador ajusta os sliders e o painel mostra a potencia calculada, o status e a chance de falha.

Para rodar localmente:
pip install streamlit
streamlit run dashboard.py

O codigo do dashboard esta versionado no repositorio e roda localmente.

## 7. Git
- Branch principal: main.
- Branch de desenvolvimento: develop.
- Branches feature: feature/corrige-pipeline, feature/atualiza-readme, etc.

## 8. Limitacoes (Honestidade Radical)
- Precisao baixa: de cada 100 alertas, so 23 sao falhas reais. O recall e 0,78.
- Dados estaticos: o dataset e um retrato fixo.
- Acuracia engana: com 96,6% normais, um modelo que chuta "tudo normal" acerta quase isso.

## 9. Tecnologias
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn e Streamlit.

Usei ferramentas de inteligencia artificial como apoio, mas conferi e validei os resultados.

## 10. Video
[COLE AQUI O LINK DO GOOGLE DRIVE QUANDO GRAVAR]

Helyton Renato Goncalves Ronchi, novembro de 2026.
