# Prevendo Falhas em Máquinas: Projeto Final do Módulo 1

Este é o meu projeto final do Módulo 1 do curso **Desenvolvimento de IA para Análise Preditiva**, no SCTEC/SENAI.

**Autor:** Helyton Renato Gonçalves Ronchi, de São Ludgero, SC
**LinkedIn:** linkedin.com/in/helytonronchi

---

## A ideia do projeto

Quando uma máquina quebra no meio do trabalho, a fábrica para, o prejuízo vem e ninguém estava esperando. Eu quis construir um programa que olhasse os dados da máquina (temperatura, velocidade, força e desgaste) e avisasse antes se ela está perto de falhar.

Uma regra guiou o projeto inteiro: **deixar uma falha passar sai muito mais caro do que parar a máquina à toa.** Por isso, nas contas, tratei uma falha não avisada como 10 vezes pior que um alarme falso.

## De onde vem o meu jeito de olhar

Antes de entrar na tecnologia, trabalhei com pintura, elétrica e serviços gerais. Quem trabalha com elétrica aprende cedo que a força de um motor depende de duas coisas juntas: o torque e a velocidade. Foi por isso que fui conferir também a **potência** (torque vezes rotação). Não bastava olhar cada número sozinho.

## Os dados

- Usei o arquivo de manutenção preditiva do curso, com **10.000 máquinas**.
- Só **3,4%** delas tinham falha. Ou seja, quase tudo é "normal", e isso atrapalha o programa a aprender o que é falha.
- Tirei 500 linhas que tinham informação faltando. Sobraram 9.500.

## O que eu fiz, passo a passo

1. **Limpei os dados**, tirando as linhas incompletas.
2. **Tirei as colunas que entregavam a resposta.** Existem colunas que dizem o tipo da falha (`falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf`, `falha_rnf`). Se eu deixasse, o programa estaria colando da prova. Isso se chama vazamento de dados.
3. **Transformei o tipo da máquina (L, M, H) em números**, porque o programa só entende números.
4. **Separei os dados em três partes:** uma para treinar, uma para ajustar e uma para o teste final. O teste só foi usado no fim, uma vez.
5. **Equilibrei o treino com SMOTE.** Como as falhas eram poucas, criei exemplos parecidos com as reais, só na parte do treino, para o teste continuar honesto.
6. **Comparei dois modelos:** o KNN e a Árvore de Decisão.
7. **Escolhi o tamanho da árvore pelo custo**, e não pelo palpite. Ficou com a profundidade 10, a de menor custo na validação.

## Resultado no teste final

O teste usou 1.900 máquinas que o programa nunca tinha visto.

| O que foi medido | Resultado | Em palavras simples |
|---|---|---|
| Recall | 0,78 | De cada 100 falhas reais, o programa avisou cerca de 78 |
| Precisão | 0,29 | Quando ele avisa que vai falhar, acerta perto de 3 vezes em cada 10 |
| Acurácia | 0,93 | Acerta 93% no geral, mas esse número engana, porque quase tudo é normal |
| Custo | 260 | Quanto menor, melhor. É a nossa conta de prejuízo |

Na etapa de ajuste, o KNN ficou com custo 340 e a árvore com custo 287. A árvore foi melhor para o que eu queria.

As profundidades 5 e 10 ficaram quase empatadas (custo 288 e 287). Fiquei com a 10 porque segui a regra que eu tinha definido antes de olhar o teste.

## O que mais pesou na decisão do programa

| Posição | Informação | Peso |
|---|---|---|
| 1 | Velocidade de rotação | 0,42 |
| 2 | Torque | 0,29 |
| 3 | Desgaste da ferramenta | 0,20 |
| 4 | Temperatura do ar | 0,04 |
| 5 | Temperatura do processo | 0,03 |

Rotação e torque foram os que mais pesaram, e isso faz sentido: eles formam a potência da máquina. Conferi a potência à parte. Das 322 falhas, 91 aconteceram fora da faixa de 3.500 a 9.000 W, e nenhuma máquina normal ficou fora dessa faixa.

## Como eu me certifiquei de que não foi sorte

Antes de confiar no resultado, rodei o mesmo programa com dados inventados, sem nenhuma ligação entre as informações e a falha. Ali o programa praticamente chutou: acertou só **17%** das falhas. Com os dados reais, ele acertou **78%**. Isso me mostrou que o ganho vem dos dados e não de algum erro meu no código.

## O que ainda não está bom

- **Muito alarme falso.** A precisão de 0,29 é baixa. Escolhi assim de propósito, porque prefiro um alarme à toa a uma máquina quebrada. Mas na prática isso daria trabalho para a equipe.
- **O programa decorou parte do treino.** No treino ele pegou quase todas as falhas (0,995), mas no teste pegou 0,78. É um sinal de que ele se apegou demais aos exemplos que viu.
- **Os dados são uma foto parada.** Não têm o passar do tempo nem informação de uma fábrica de verdade.
- **Testei uma vez só.** Não fiz a validação cruzada, que daria mais segurança no resultado.

## O que eu faria a seguir

- Fazer a validação cruzada, para ver se o resultado se mantém em várias divisões dos dados.
- Experimentar modelos mais fortes, como Random Forest e Gradient Boosting, mantendo a conta de custo.
- Ajustar o ponto de decisão do programa para diminuir os alarmes falsos sem perder muitas falhas.

## Como rodar no seu computador

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn
python3 manutencao_preditiva_final.py
```

O programa lê o arquivo `manutencao_preditiva.csv`, treina os modelos e salva os gráficos na própria pasta.

## Ferramentas usadas

Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib e Seaborn.

Usei ferramentas de inteligência artificial como apoio, mas conferi e validei os resultados por conta própria.

---

Helyton Renato Gonçalves Ronchi, novembro de 2026
