# Prevendo Falhas em Máquinas: Projeto Final do Módulo 1

**Autor:** Helyton Renato Gonçalves Ronchi (São Ludgero, SC)

**LinkedIn:** linkedin.com/in/helytonronchi

**Curso:** Desenvolvimento de IA para Análise Preditiva (SCTEC/SENAI)

---

## 1. O Problema

Quando uma máquina quebra no meio do trabalho, a fábrica para e o prejuízo é grande. O objetivo deste projeto é construir um programa que leia os dados dos sensores (temperatura, velocidade, torque e desgaste) e avise antes que a máquina falhe.

A variável alvo é binária:
- `falha_maquina = 0`: funcionamento normal.
- `falha_maquina = 1`: falha detectada.

O dataset tem 10.000 amostras, sendo que **96,6% são normais e apenas 3,4% são falhas**. Esse desbalanceamento é o principal desafio do projeto.

## 2. Origem do Meu Olhar Técnico

Antes de migrar para a tecnologia, trabalhei com pintura, elétrica e serviços gerais. Essa experiência de campo orientou uma decisão do projeto: criar uma variável de **potência** (torque × velocidade de rotação), porque em máquina o que importa é a combinação das grandezas, e não cada leitura isolada. A potência foi calculada em **watts de verdade** (torque × rpm × 2π/60).

A análise de importância das variáveis apoiou esse caminho: a **potência** ficou em segundo lugar (0,289), e o torque sozinho quase não pesou (0,010). A variável que criei absorveu a informação do torque. A rotação liderou (0,412) e o desgaste veio em terceiro (0,195).

## 3. As 7 Fases do Pipeline

### Fase 1: Análise Exploratória (EDA)
- **Dimensões:** 10.000 linhas × 14 colunas.
- **Tipos de dados:** impressos no terminal.
- **.describe():** temperatura do ar de 295,3 K a 304,5 K; torque de 3,8 Nm a 76,6 Nm.
- **Desbalanceamento:** 96,61% normal vs 3,39% falha.
- **Gráficos:** histograma do torque, barras de desbalanceamento, heatmap de correlação.

**Análise dos gráficos:** O gráfico de barras confirma o desbalanceamento extremo, o que me levou a usar SMOTE apenas no treino e a desconfiar da acurácia como métrica principal. O heatmap mostrou que as variáveis são pouco correlacionadas entre si. O histograma do torque mostrou distribuição assimétrica, o que me levou a imputar nulos pela mediana.

### Fase 2: Limpeza e Tratamento
- **Duplicatas:** removidas (0 encontradas).
- **Dados ausentes:** imputação por **mediana** (a distribuição do torque tem outliers, e a mediana é mais robusta que a média).
- **Outliers:** boxplots gerados. Decidi manter os outliers, porque são leituras reais de máquina.
- **Vazamento:** colunas falha_twf, falha_hdf, falha_pwf, falha_osf, falha_rnf removidas.

### Fase 3: Feature Engineering
- **Nova variável:** `potencia_w = torque_nm * velocidade_rotacao_rpm * 2 * π / 60` (watts de verdade).
- **Nulos:** 0 após imputação.

### Fase 4: Divisão e Balanceamento
- **Split:** 80/20 com stratify=y (8.000 treino, 2.000 teste).
- **SMOTE:** apenas no treino (8.000 → 15.458 amostras balanceadas).

### Fase 5: Escalonamento
- **KNN:** StandardScaler aplicado apenas nas colunas contínuas (fit_transform no treino, transform no teste).
- **Árvore:** sem escalonamento (imune à escala).

### Fase 6: Ajuste de Parâmetros e Overfitting
- **KNN:** K=3 (0,964 treino / 0,905 teste), K=5 (0,953 / 0,896), K=7 (0,946 / 0,888).
- **Árvore:** max_depth=3 (0,857 / 0,847), max_depth=5 (0,902 / 0,902), max_depth=None (0,993 / 0,937).
- **Overfitting:** max_depth=None decorou o treino.

### Fase 7: Avaliação Final
| Modelo | Acurácia (teste) |
|---|---|
| KNN (K=5) | 0,896 |
| Árvore (max_depth=5) | 0,902 |

**Modelo escolhido:** Árvore de Decisão (max_depth=5), pela configuração mais estável.

**Sobre o KNN:** K=3 ganhou no teste (0,905), mas K=5 foi escolhido por ser um meio-termo.

**Pela regra da maior acurácia no teste:** a árvore sem limite (0,937) também vence o KNN K=3 (0,905).

## 4. Como Executar o Projeto

```bash
git clone https://github.com/deegpnini/m1s13-projeto-final.git
cd m1s13-projeto-final
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline_m1s13.py

5. Dashboard Interativo (Extra)
Criei um dashboard em Streamlit (dashboard.py) para mostrar o modelo "operando" uma máquina. O operador ajusta os sliders (torque, rpm, desgaste, temperaturas) e o painel mostra a potência calculada, o status (verde/vermelho), a chance de falha e a leitura dos sensores.

Para rodar:

bash
pip install streamlit
streamlit run dashboard.py
O link do dashboard é local (http://localhost:8503). O código está em: https://github.com/deegpnini/m1s13-projeto-final/blob/main/dashboard.py

6. Git
Branch principal: main.

Branch de desenvolvimento: develop.

Branches feature: feature/corrige-pipeline, feature/atualiza-readme, feature/ajusta-requirements, feature/link-do-video.

7. Limitações (Honestidade Radical)
Precisão baixa (0,23): de cada 100 alertas, só 23 são falhas reais. O recall é 0,78.

Overfitting: max_depth=None decorou o treino.

Dados estáticos: o dataset é um retrato fixo.

Acurácia engana: com 96,6% normais, um modelo que chuta "tudo normal" acerta quase isso.

Acurácia de treino: o valor de treino é medido no conjunto balanceado pelo SMOTE.

8. Melhorias Futuras
Validação cruzada.

Testar Random Forest e Gradient Boosting.

Ajustar o limiar de decisão para reduzir alarmes falsos.

9. Tecnologias
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn e Streamlit.

Usei ferramentas de inteligência artificial como apoio, mas conferi e validei os resultados.

10. Vídeo de Apresentação
O vídeo de até 7 minutos, com a demonstração do terminal e do dashboard, está disponível no link abaixo:
[COLE AQUI O LINK DO GOOGLE DRIVE QUANDO GRAVAR]

Helyton Renato Gonçalves Ronchi, novembro de 2026.
