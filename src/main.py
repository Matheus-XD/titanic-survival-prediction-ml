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

