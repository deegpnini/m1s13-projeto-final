# Manutenção Preditiva: Classificação de Falhas em Máquinas Industriais

Projeto avaliativo do Módulo 1 do curso **Desenvolvimento de IA para Análise Preditiva** (SCTEC/SENAI).

**Autor:** Helyton Renato Gonçalves Ronchi · São Ludgero, SC
**LinkedIn:** linkedin.com/in/helytonronchi

---

## 1. Objetivo

Construir um modelo de classificação que indique, a partir de leituras de sensores, se uma máquina vai falhar. O foco é reduzir paradas não planejadas: em manutenção, deixar a falha passar custa muito mais do que parar a máquina por precaução. Por isso o projeto é avaliado por **custo de erro**, e não só por acurácia.

## 2. Origem do olhar técnico

Antes de migrar para tecnologia, trabalhei com pintura, elétrica e serviços gerais. Essa experiência de campo orientou uma decisão do projeto: criar uma variável de **potência** (torque × velocidade de rotação), porque em máquina o que importa é a combinação das grandezas, e não cada leitura isolada. A análise de importância das variáveis confirmou esse caminho (seção 6).

## 3. Dados

| Item | Valor |
|---|---|
| Fonte | Dataset de manutenção preditiva fornecido no curso (arquivo `manutencao_preditiva.csv`) |
| Tamanho original | 10.000 amostras |
| Classe alvo | `falha_maquina` (0 = normal, 1 = falha) |
| Desbalanceamento | cerca de 96,6% normal e 3,4% falha |
| Variáveis usadas | tipo do produto, temperatura do ar, temperatura do processo, velocidade de rotação, torque e desgaste da ferramenta |

## 4. Metodologia

1. **Limpeza:** remoção de 500 linhas com valores nulos (restaram 9.500).
2. **Prevenção de vazamento de dados:** remoção das colunas `falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf` e `falha_rnf`. Elas indicam o tipo da falha e entregariam a resposta ao modelo.
3. **Codificação:** Label Encoding da coluna `tipo` (L, M, H).
4. **Divisão dos dados:** treino, validação e teste separados. O teste só é usado uma vez, no final.
5. **Balanceamento:** SMOTE aplicado **apenas no treino**, para não contaminar a validação nem o teste.
6. **Modelos comparados:** KNN e Árvore de Decisão.
7. **Escolha da profundidade da árvore por custo:** uma falha não prevista pesa 10 vezes mais que um alarme falso. A profundidade com menor custo na validação foi a escolhida.

## 5. Resultados (conjunto de teste, 1.900 amostras)

Árvore de Decisão, profundidade 10:

| Métrica (classe falha) | Valor |
|---|---|
| Recall | 0,78 |
| Precisão | 0,29 |
| F1-score | 0,43 |
| Acurácia geral | 0,93 |
| Custo no teste | 260 |

Comparação na validação:

| Modelo | Recall | Custo |
|---|---|---|
| KNN | 0,69 | 340 |
| Árvore, profundidade 10 | 0,71 | 287 |
| Árvore, profundidade 5 | 0,80 | 288 |

As profundidades 5 e 10 ficaram praticamente empatadas em custo. A 10 foi adotada pelo critério definido antes do teste (menor custo).

## 6. Variáveis mais importantes

| Posição | Variável | Importância |
|---|---|---|
| 1 | `velocidade_rotacao_rpm` | 0,42 |
| 2 | `torque_nm` | 0,29 |
| 3 | `desgaste_ferramenta_min` | 0,20 |
| 4 | `temperatura_ar_k` | 0,04 |
| 5 | `temperatura_processo_k` | 0,03 |

Rotação e torque dominam o modelo, o que é coerente com a física: potência é o produto dos dois.

Teste complementar com a potência (torque × rpm): das 322 falhas, 91 ficaram fora da faixa de 3.500 a 9.000 W, e nenhuma máquina normal ficou fora dessa faixa.

## 7. Validação do pipeline

Antes de confiar no resultado, rodei o mesmo código em um conjunto sintético com dados aleatórios (sem relação entre variáveis e falha). O recall no teste ficou em **0,17**, nível de chute. Com os dados reais, ficou em **0,78**. Isso indica que o ganho vem do sinal nos dados, e não de erro no código.

## 8. Limitações

- **Precisão baixa (0,29):** a maior parte dos alertas é alarme falso. A escolha por custo aceita esse efeito de forma deliberada.
- **Sobreajuste:** recall de 0,995 no treino contra 0,78 no teste. O modelo memoriza parte do treino.
- **Dados estáticos:** o dataset é um retrato fixo, sem série temporal nem dados de campo.
- **Uma única divisão de teste:** não foi feita validação cruzada completa.

## 9. Próximos passos

- Testar validação cruzada para medir a variação do recall.
- Testar outros modelos de árvores (Random Forest, Gradient Boosting) mantendo o critério de custo.
- Ajustar o limiar de decisão para reduzir alarmes falsos sem perder muito recall.

## 10. Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn
python3 manutencao_preditiva_final.py
```

O script lê `manutencao_preditiva.csv`, treina os modelos e salva os gráficos na pasta.

## 11. Tecnologias

Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib e Seaborn.

Desenvolvido com apoio de ferramentas de IA generativa. Os resultados foram conferidos e validados pelo autor.

---

Helyton Renato Gonçalves Ronchi · Novembro de 2026
