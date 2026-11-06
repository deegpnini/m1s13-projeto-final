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
- **.describe():** temperatura do ar de 295,3 K a 304,5 K; torque de 3,8 Nm a 76,6 Nm.
- **Desbalanceamento:** 96,61% normal vs 3,39% falha.
- **Gráficos:** histograma do torque, barras de desbalanceamento, heatmap de correlação.

### Fase 2: Limpeza e Tratamento
- **Duplicatas:** removidas (0 encontradas).
- **Dados ausentes:** imputação por **mediana** (robusta a outliers).
- **Outliers:** boxplots de torque_nm e desgaste_ferramenta_min.
- **Vazamento:** colunas falha_twf, falha_hdf, falha_pwf, falha_osf, falha_rnf removidas.

### Fase 3: Feature Engineering
- **Nova variável:** potencia_w = torque_nm * velocidade_rotacao_rpm.
- **Nulos:** 0 após imputação.

### Fase 4: Divisão e Balanceamento
- **Split:** 80/20 com stratify=y.
- **SMOTE:** apenas no treino (7.600 → 15.458 amostras balanceadas).

### Fase 5: Escalonamento
- **KNN:** fit_transform no treino, transform no teste.
- **Árvore:** sem escalonamento (imune à escala).

### Fase 6: Ajuste de Parâmetros e Overfitting
- **KNN:** K=3 (0,964 treino / 0,905 teste), K=5 (0,953 / 0,896), K=7 (0,946 / 0,888).
- **Árvore:** max_depth=3 (0,857 / 0,847), max_depth=5 (0,902 / 0,902), max_depth=None (0,993 / 0,937).
- **Overfitting:** max_depth=None decorou o treino. A configuração estável foi max_depth=5.

### Fase 7: Avaliação Final
| Modelo | Acurácia (teste) |
|---|---|
| KNN (K=5) | 0,896 |
| Árvore (max_depth=5) | 0,902 |

**Modelo escolhido:** Árvore de Decisão (maior acurácia no teste).

**Sobre o KNN:** K=3 ganhou no teste (0,905), mas K=5 foi escolhido por ter menor gap treino-teste (menos overfitting).

## 4. Como Executar o Projeto

git clone git@github.com:deegpnini/m1s13-projeto-final.git
cd m1s13-projeto-final
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline_m1s13.py

**Script oficial:** pipeline_m1s13.py (as 7 fases do edital).

## 5. Dashboard Interativo (Extra)

Criei um dashboard em Streamlit (dashboard.py) para mostrar o modelo "operando" uma máquina. O operador ajusta os sliders (torque, rpm, desgaste, temperaturas) e o painel mostra:
- A potência calculada (torque × rpm).
- O status da máquina: verde (normal) ou vermelho (alerta de falha).
- A acurácia do modelo (0,902).

Para rodar:
pip install streamlit
streamlit run dashboard.py

## 6. Git

- **Branch principal:** master (código final).
- **Branch de desenvolvimento:** develop (ajustes e experimentos).

## 7. Limitações (Honestidade Radical)

- **Precisão baixa:** a acurácia é alta (0,902), mas a precisão da classe falha é baixa. Em produção, muitos alarmes falsos ocorreriam.
- **Overfitting:** max_depth=None decorou o treino. A configuração escolhida (max_depth=5) evita isso.
- **Dados estáticos:** o dataset é um retrato fixo, sem série temporal.
- **Acurácia engana:** com 96,6% de máquinas normais, um modelo que chuta "tudo normal" acerta quase isso.

## 8. Melhorias Futuras

- Validação cruzada para medir a variação da acurácia.
- Testar Random Forest e Gradient Boosting.
- Ajustar o limiar de decisão para reduzir alarmes falsos.

## 9. Tecnologias

Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn e Streamlit.

Usei ferramentas de inteligência artificial como apoio, mas conferi e validei os resultados por conta própria.

## 10. Vídeo de Apresentação

O vídeo de até 7 minutos, com a demonstração do terminal e do dashboard, está disponível no link abaixo:
[COLE AQUI O LINK DO GOOGLE DRIVE QUANDO GRAVAR]

---

**Helyton Renato Gonçalves Ronchi**, novembro de 2026.

## 5.1. Código do Dashboard

O código do dashboard está disponível no repositório:
https://github.com/deegpnini/m1s13-projeto-final/blob/master/dashboard.py

Para rodar na sua máquina:
pip install streamlit
streamlit run dashboard.py

## 5.2. Como visualizar o dashboard

O dashboard roda **localmente** na máquina de quem for avaliar. Não é um site público. Para visualizar:

1. Abra o terminal na pasta do projeto.
2. Rode:
   pip install streamlit
   streamlit run dashboard.py
3. O Streamlit vai mostrar um link no terminal, algo como:
   Local URL: http://localhost:8501
4. Copie esse link e cole no navegador. O dashboard abrirá.

**Importante:** o link `localhost` só funciona na máquina onde o dashboard está rodando. Por isso, o avaliador precisa executar o comando acima para ver o painel.
