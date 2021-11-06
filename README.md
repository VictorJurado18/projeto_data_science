# Projeto Ciência de Dados - Previsão de Vendas

Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio.

## :memo: Pré-requisitos:

- Instalar biblioteca pandas:  ```pip install pandas```
- Instalar biblioteca matplotlib:  ```pip install matplotlib```
- Instalar biblioteca seaborn:  ```pip install seaborn```
- Utilizar a IDE Jupyter :ringed_planet:


## :snake: Executando os scripts:

#Passo 1: Importar base de dados e exibir 
(obs. neste caso, a base de dados está na mesma pasta do arquivo notebook).

```python
import pandas as pd

tabela = pd.read_csv("advertising.csv")
display(tabela)
```

<img src="https://github.com/VictorJurado18/projeto_data_science/blob/main/tabela.png?raw=true" width="180px">

#Passo 2: Análise Exploratória
-Vamos tentar visualizar como as informações de cada item estão distribuídas
-Vamos ver a correlação entre cada um dos itens

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(tabela.corr(), cmap="Blues", annot = True)
print("Grafio Correlação")
plt.show()

sns.pairplot(tabela)
print("Grafico de Ruido")
plt.show()

```
<img src="https://github.com/VictorJurado18/projeto_data_science/blob/main/grafico%20correlacao.jpg?raw=true" width="250px">
<img src="https://github.com/VictorJurado18/projeto_data_science/blob/main/pairplot.jpg?raw=true" width="250px">

#Passo 3: Preparação dos dados para treinarmos o Modelo de Machine Learning
- Separando em dados de treino e dados de teste

```python
from sklearn.model_selection import train_test_split

#separar dados de x e de y

#quem é o y
y = tabela["Vendas"]

#quem é o x
x = tabela.drop("Vendas", axis = 1) #0 para linha e 1 para coluna

#aplicar a train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, random_state = 1)#random_state = 1 separa sempre a mesma base de dados para treino e teste
#test_size = 0.3 ou seja 30% dos dados para teste...deixar por padrão vai ficar em media 20% para teste

```
#Passo 4: Vamos escolher os modelos que vamos usar:
- Regressão Linear
- RandomForest (Árvore de Decisão)

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

modelo_regressao_linear = LinearRegression()
modelo_random_forest = RandomForestRegressor()

modelo_regressao_linear.fit(x_treino,y_treino)
modelo_random_forest.fit(x_treino, y_treino)
```

#Passo 5: Teste da AI e Avaliação do Melhor Modelo
- Vamos usar o R²

```python
prev_regressao_linear = modelo_regressao_linear.predict(x_teste)
prev_random_forest = modelo_random_forest.predict(x_teste)

from sklearn import metrics

print(metrics.r2_score(y_teste, prev_regressao_linear))
print(metrics.r2_score(y_teste, prev_random_forest))
```
```
0.9048917241361681
0.967234317428181
```
## :chart_with_upwards_trend: Visualização Gráfica das Previsões: 

```python
#random_forest é o melhor modelo
tabela_aux = pd.DataFrame()
tabela_aux ["y_teste"] = y_teste
tabela_aux ["regressao linear"] = prev_regressao_linear
tabela_aux ["random forest"] = prev_random_forest

plt.figure(figsize = (15,5))
sns.lineplot(data = tabela_aux)
plt.show()
```
<img src="https://github.com/VictorJurado18/projeto_data_science/blob/main/compara%C3%A7%C3%A3o%20dos%20testes.jpg?raw=true" width="100%">

## Qual a importância de cada variável para as vendas?

```python
sns.barplot(x = x_treino.columns, y = modelo_random_forest.feature_importances_)
plt.show()
```
<img src="https://github.com/VictorJurado18/projeto_data_science/blob/main/Importancia%20para%20resultado.jpg?raw=true" width="250px">

- Investimento em TV tem trazido resultados expressivos diretamente nas vendas.


