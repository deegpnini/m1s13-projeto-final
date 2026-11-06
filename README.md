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

Antes de migrar para a tecnologia, trabalhei com pintura, elétrica e serviços gerais. Essa experiência de campo orientou uma decisão do projeto: criar uma variável de **potência** (torque × velocidade de rotação), porque em máquina o que importa é a combinação das grandezas, e não cada leitura isolada.

A análise de importância das variáveis confirmou esse caminho: rotação (0,42) e torque (0,29) dominam o modelo.

## 3. As 7 Fases do Pipeline

### Fase 1: Análise Exploratória (EDA)

- **Dimensões:** 10.000 linhas × 14 colunas.
- **Tipos de dados:** 6 colunas numéricas, 3 categóricas, 5 binárias (falhas por tipo).
- **`.describe()`:** mostrou que a temperatura do ar varia de 295,3 K a 304,5 K, e o torque de 3,8 Nm a 76,6 Nm.
- **Desbalanceamento:** 96,61% normal vs 3,39% falha.
- **Gráficos gerados:**
  - Histograma da distribuição do torque (`grafico_histograma.png`).
  - Gráfico de barras comprovando o desbalanceamento (`grafico_barras.png`).
  - Heatmap de correlação de Pearson (`grafico_heatmap.png`).

**Análise textual:** o desbalanceamento extremo exige uma técnica de reamostragem (SMOTE) para o modelo não ignorar a classe minoritária. A correlação mostra que as variáveis são pouco correlacionadas entre si, o que é bom para o modelo.

### Fase 2: Limpeza e Tratamento

- **Duplicatas:** removidas (0 duplicatas encontradas).
- **Dados ausentes:** imputação por **mediana** (justificativa: as variáveis têm outliers, e a mediana é mais robusta que a média).
- **Outliers:** boxplots gerados (`grafico_boxplot.png`) para as variáveis `torque_nm` e `desgaste_ferramenta_min`.
- **Remoção de vazamento:** as colunas `falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf` e `falha_rnf` foram removidas porque entregam a resposta ao modelo.

### Fase 3: Feature Engineering

- **Nova variável:** `potencia_w = torque_nm * velocidade_rotacao_rpm`.
- **Tratamento de nulos:** os nulos das colunas de entrada foram preenchidos com a mediana antes da multiplicação (0 nulos restantes).

**Justificativa:** a potência é a combinação física que explica a falha. Um motor com torque alto e rotação alta está sob estresse maior.

### Fase 4: Divisão e Balanceamento

- **Split:** 80% treino, 20% teste, com `stratify=y` (mantém a proporção de falhas nos dois conjuntos).
- **SMOTE:** aplicado **apenas no treino** para evitar vazamento de dados (o teste fica com a distribuição real).
- **Resultado:** o treino passou de 7.600 para 15.458 amostras balanceadas (50/50).

### Fase 5: Escalonamento (StandardScaler)

- **KNN:** recebeu `fit_transform` no treino e `transform` no teste (evita vazamento).
- **Árvore:** não foi escalonada porque é **imune à escala** (divide os dados por limiares, não por distância).

### Fase 6: Ajuste de Parâmetros e Overfitting

**KNN (variação de K):**

| K | Acurácia Treino | Acurácia Teste |
|---|---|---|
| 3 | 0,964 | 0,905 |
| 5 | 0,953 | 0,896 |
| 7 | 0,946 | 0,888 |

**Árvore (variação de max_depth):**

| max_depth | Acurácia Treino | Acurácia Teste |
|---|---|---|
| 3 | 0,857 | 0,847 |
| 5 | 0,902 | 0,902 |
| None | 0,993 | 0,937 |

**Análise de overfitting:**
- O KNN com K=3 tem acurácia de treino (0,964) muito acima da de teste (0,905). Overfitting leve.
- A Árvore com `max_depth=None` tem acurácia de treino (0,993) muito acima da de teste (0,937). **Overfitting clássico** (decorou o treino).

**Configuração estável:** a Árvore com `max_depth=5` empatou treino e teste (0,902), o que indica que não houve overfitting.

### Fase 7: Avaliação Final e Veredito

| Modelo | Acurácia (teste) |
|---|---|
| KNN (K=5) | 0,896 |
| Árvore (max_depth=5) | 0,902 |

**Modelo escolhido:** **Árvore de Decisão** (maior acurácia no teste). O KNN teve desempenho inferior, provavelmente porque o StandardScaler não foi suficiente para lidar com a distribuição não-linear dos dados.

## 4. Como Executar o Projeto

### Pré-requisitos

- Python 3.10 ou superior.
- Git instalado.

### Passo a passo

1. **Clone o repositório:**
   ```bash
   git clone git@github.com:deegpnini/m1s13-projeto-final.git
   cd m1s13-projeto-final
