# Oikos Machine: Manutenção Preditiva para a Indústria (P/Final-M1)

**Autor:** Helyton Renato Gonçalves Ronchi (São Ludgero, SC)
**LinkedIn:** linkedin.com/in/helytonronchi
**Curso:** Desenvolvimento de IA para Análise Preditiva, T4 (SCTEC/SENAI)

**Arquivos principais:**
- `notebook_final.ipynb`: Notebook Jupyter completo com as 7 fases, as células de texto justificando as decisões e as saídas executadas.
- `pipeline_m1s13.py`: Versão em script Python com a mesma lógica.

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
- Boxplot de outliers gerado.

**Fase 3: Feature Engineering**
- Nova variável: potencia_w = torque_nm * velocidade_rotacao_rpm * 2 * π / 60 (watts de verdade).

**Fase 4: Divisão e Balanceamento**
- Split: 80/20 com stratify=y.
- SMOTE: apenas no treino (8.000 -> 15.458 amostras balanceadas).

**Fase 5: Escalonamento**
- KNN: StandardScaler aplicado apenas nas colunas contínuas (fit_transform no treino, transform no teste).
- Árvore: sem escalonamento (imune à escala).

**Fase 6: Ajuste de Parâmetros e Overfitting**
- KNN: testados K=3, 5 e 7. O melhor foi K=3, com acurácia de 0,918 no teste.
- Árvore: testadas max_depth=3, 5 e None. A árvore sem limite (None) teve a maior acurácia no teste (0,940), mas decorou o treino (0,993 no treino balanceado).
- A tabela completa, com acurácia no treino balanceado, no treino original e no teste, está no notebook.

**Fase 7: Avaliação Final e Veredito**
- Melhor KNN: K=3 (0,918). Melhor Árvore: max_depth=None (0,940). **Vence a Árvore de Decisão.**
- O chute "tudo normal" acerta 0,966, mais que os dois modelos. A acurácia sozinha não sustenta a escolha.
- A árvore vencedora tem precisão 0,31 e recall 0,65 na classe falha: pega cerca de 65% das falhas, mas muitos alertas são alarme falso.
- Recomendação: adotar a Árvore de Decisão como triagem, com ressalvas, porque parar a máquina à toa custa menos do que deixar quebrar.

## 4. Como Executar o Projeto
git clone https://github.com/deegpnini/oikos-machine.git
cd oikos-machine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook_final.ipynb

## 5. Dashboard Interativo (Extra)
Criei um dashboard em Streamlit (dashboard.py) para mostrar o modelo operando uma máquina. O operador ajusta os sliders e o painel mostra a potência calculada, o status e a chance de falha.

Para rodar localmente:
pip install streamlit
streamlit run dashboard.py

## 6. Git
- `main`: código final entregue.
- `develop`: integração das features.
- `feature/eda-e-limpeza`: fases 1 e 2.
- `feature/modelagem-e-metricas`: fases 3 a 7.
- `feature/readme-e-notebook`: documentação e notebook final.

## 7. Limitações (Honestidade Radical)
- Precisão baixa (0,31): menos de um terço dos alertas é falha real. Recall 0,65: cerca de 35% das falhas passam.
- Overfitting: a árvore sem limite memoriza parte do treino.
- Dados estáticos: o dataset é um retrato fixo, sem série temporal.
- A acurácia engana: o chute "tudo normal" acerta 96,6%.
- SMOTE sobre o tipo codificado gera valores fracionados.

## 8. Melhorias Futuras
- Validação cruzada.
- SMOTENC para a coluna categórica.
- Ajustar o limiar de decisão para trocar alarmes falsos por falhas detectadas.
- Testar Random Forest e Gradient Boosting com a mesma regra de escolha.

## 9. Tecnologias
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn e Streamlit.

Usei ferramentas de inteligência artificial como apoio, mas conferi e validei os resultados.

## 10. Vídeo
[COLE AQUI O LINK DO GOOGLE DRIVE QUANDO GRAVAR]

Helyton Renato Gonçalves Ronchi, novembro de 2026.
