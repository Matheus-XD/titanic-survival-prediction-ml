# 🚢 Titanic Survival Prediction (Machine Learning)

## 📌 Sobre o Projeto

Este projeto tem como objetivo prever a sobrevivência de passageiros do Titanic utilizando técnicas de Machine Learning.

O modelo foi treinado com dados reais contendo informações como:

* Classe do passageiro (`Pclass`)
* Sexo (`Sex`)
* Idade (`Age`)
* Tarifa (`Fare`)
* Número de familiares a bordo (`SibSp`)

---

## 🧠 Tecnologias Utilizadas

* Python
* pandas
* scikit-learn

---

## ⚙️ Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd ML-Titanic
```

---

### 2. Crie um ambiente virtual

```bash
python -m venv venv
```

---

### 3. Ative o ambiente virtual

#### Windows (PowerShell)

```powershell
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

---

### 5. Execute o projeto

```bash
python main.py
```

---

## 📊 Modelos Utilizados

O projeto compara diferentes algoritmos de Machine Learning:

* Regressão Logística
* Árvore de Decisão
* Random Forest

---

## 📈 Resultados

Os modelos foram avaliados utilizando acurácia.

Exemplo de resultados:

```text
Acurácia Logistic Regression: ~0.75 - 0.83
Acurácia Decision Tree:       ~0.71 - 0.76
Acurácia Random Forest:       ~0.78 - 0.85
```

👉 O modelo **Random Forest** apresentou melhor desempenho na maioria dos testes.

---

## 🔁 Processo de Machine Learning

O fluxo do projeto segue:

1. Leitura dos dados (pandas)
2. Limpeza e tratamento
3. Seleção de features
4. Treinamento do modelo
5. Avaliação de desempenho

---

## 🧩 Melhorias Futuras

* Feature engineering mais avançado
* Ajuste de hiperparâmetros
* Validação cruzada
* Uso de modelos mais avançados (Gradient Boosting)

---

## 📁 Estrutura do Projeto

```text
ML-Titanic/
│
├── main.py
├── train.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Autor

Projeto desenvolvido como prática de Machine Learning.

---

## 💡 Observações

* Não subir a pasta `venv/` para o repositório
* Garantir que o arquivo `requirements.txt` esteja atualizado

