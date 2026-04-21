import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# carregar dados
df = pd.read_csv("data/train.csv")

# selecionar colunas simples
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]

# transformar texto em número
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# remover linhas dados faltantes para evitar erros durante a análise
df = df.dropna()
# Caso tenha muitas linha com dados faltando, é recomendado utilizar:
# df['Age'] = df['Age'].fillna(df['Age'].mean())
# essa função serve para preencher as os campos vazios com a media das outras linhas, isso evita que
# muitos dados sejam perdidos com o uso do dropna

# separar entrada e saída
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]
y = df['Survived']

# dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# criar modelo
modelLR = LogisticRegression(random_state=42)
modelDT = DecisionTreeClassifier(random_state=42)
modelRF = RandomForestClassifier(random_state=42)

# treinar
modelLR.fit(X_train, y_train)
modelDT.fit(X_train, y_train)
modelRF.fit(X_train, y_train)

# avaliar
print("Acurácia Logistic Regretion: ", modelLR.score(X_test, y_test))
print("Acurácia Decision Tree: ", modelDT.score(X_test, y_test))
print("Acurácia Randon Forest: ", modelRF.score(X_test, y_test))

''' O que eu entendi até agora:
1 - foi criado uma variavel chamada df (dataframe) e ela vai receber um tipo de planilha com os dados lidos do arquivo train.csv lidos pela função pd.read_csv 
2 - a segunda linha que filtra apenas o que é necessario para a análise.
3 - Transformar conteudo da coluna sexo em numero, porque o modelo funciona melhor com numeros do que com nomes.
4 - Remove linhas que tem dados faltando dados faltantes para não gerar erro no codigo
5 - separa entrada e saída, x vai receber os dados para analise e y vai retornar a resposta
6 - divisão treino e teste, cria 4 variaveis (x e y para treino e x e y para  teste) para receber os 4 valores retornados pela função train_test_split() que recebe os valores de x e y
agora o que significa a linha model = LogisticRegression()
7 - cria o modelo de LogisticRegression() numa variavel chamada model 
8 - a função model.fit() recebe os valores de x train e y train para treinar o modelo
agora para finalizar o print(model.score(X_test e Y_test)) servem pra mostrar o quão esse modelo é preciso né? como essa função funciona? '''