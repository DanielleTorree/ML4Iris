"""
Machine Learning - Danielle Torres
Classificação do dataset Iris
"""

#%% Ambiente e Imports

# Oculta warnings
import warnings
warnings.filterwarnings("ignore")

# Bibliotecas essenciais
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Pré-processamento e modelos
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Modelos básicos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Ensembles
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
)

#%% Carga do Dataset

"""
O dataset Iris contém informações sobre diferentes espécies de flores,
incluindo medidas das sépalas e pétalas. O objetivo é prever a espécie da flor.
"""

# Caminho do arquivo (relativo à raiz do projeto)
dataset_path = "https://raw.githubusercontent.com/DanielleTorree/ML4Iris/main/api/MachineLearning/dataset/dataset_iris.csv"

# Leitura
dataset = pd.read_csv(dataset_path)

# Visualização inicial
print(dataset.head())
print(dataset.info())

#%% Separação dos Dados

# Definições
test_size = 0.20
seed = 7

# Separação entre atributos (X) e classe (y)
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Holdout com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed, stratify=y
)

# Validação cruzada estratificada
num_folds = 10
scoring = "accuracy"
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

#%% Modelagem - Linha Base e Ensembles

np.random.seed(seed)

# Modelos base
base_models = [
    ("LR", LogisticRegression(max_iter=200)),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC())
]

# Ensembles
base_estimator = DecisionTreeClassifier()
num_trees = 100
max_features = 3

voting_models = [
    ("logistic", LogisticRegression(max_iter=200)),
    ("cart", DecisionTreeClassifier()),
    ("svm", SVC())
]

ensemble_models = [
    ("Bagging", BaggingClassifier(estimator=base_estimator, n_estimators=num_trees)),
    ("RF", RandomForestClassifier(n_estimators=num_trees, max_features=max_features)),
    ("ET", ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)),
    ("Ada", AdaBoostClassifier(n_estimators=num_trees)),
    ("GB", GradientBoostingClassifier(n_estimators=num_trees)),
    ("Voting", VotingClassifier(estimators=voting_models))
]

# Junta todos os modelos
all_models = base_models + ensemble_models

#%% Avaliação dos Modelos

results = []
names = []

for name, model in all_models:
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_scores)
    names.append(name)
    print(f"{name}: {cv_scores.mean():.4f} ({cv_scores.std():.4f})")

#%% Visualização

plt.figure(figsize=(15, 8))
plt.title("Comparação dos Modelos - Acurácia")
plt.boxplot(results, labels=names)
plt.ylabel("Acurácia")
plt.grid(True)
plt.show()

#%% Modelagem com dados padronizados e normalizados usando Pipelines

standard_scaler = ("StandardScaler", StandardScaler())
minmax_scaler = ("MinMxScaler", MinMaxScaler())

pipelines = []

# Criando pipelines para cada modelo e cada tipo de pré-processamento
for name, model in all_models: 
    pipelines.append((f"{name}-orig", Pipeline(steps=[(name, model)])))
    pipelines.append((f"{name}-std", Pipeline(steps=[standard_scaler, (name, model)])))
    pipelines.append((f"{name}-minmax", Pipeline(steps=[minmax_scaler, (name, model)])))

results = [] 
names = []

for name, pipeline in pipelines:
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

plt.figure(figsize=(25, 6))
plt.title("Comparação de Modelos - Dataset Original, Padronizado e Normalizado")
plt.boxplot(results, labels=names)
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

#%% Otimização de hiperparâmetros com Grid Search
param_grids = {
    "LR": {
        "LR__C": [0.5, 1, 2],  # Faixa em torno do melhor valor encontrado (1)
        "LR__solver": ["saga"],  # Melhor solver encontrado
    },
    "KNN": {
        "KNN__n_neighbors": [9, 11, 13],  # Em torno do melhor (11)
        "KNN__metric": ["euclidean"],     # Melhor métrica encontrada
    },
    "RF": {
        "RF__n_estimators": [5, 10, 20],            # Faixa menor, centrada no melhor (10)
        "RF__max_features": ["sqrt"],               # Melhor resultado
        "RF__max_depth": [None, 10],                # Melhor resultado: None
        "RF__min_samples_split": [2, 5],            # Ambos deram bons resultados
        "RF__min_samples_leaf": [1, 2],             # Ambos foram usados nos melhores casos
    }
}

# Para modelos onde aplicaremos Grid Search (exemplo: LR, KNN, RF)
grid_models = ["LR", "KNN", "RF"]

kfold_gs = 5

for name, pipeline in pipelines:
    model_key = name.split("-")[0]
    if model_key in grid_models:
        param_grid = param_grids[model_key]
        grid = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, cv=kfold_gs, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Modelo: {name} - Melhor: {grid.best_score_:.4f} usando {grid.best_params_}")

#%% Treinamento final e avaliação no conjunto teste usando o melhor modelo encontrado (exemplo Random Forest + MinMaxScaler)

from sklearn.pipeline import make_pipeline

np.random.seed(seed)

final_model = RandomForestClassifier(
    n_estimators=50,
    max_features="sqrt",
    min_samples_split=2,
    max_depth=10,
    min_samples_leaf=1,
)

pipeline_final = make_pipeline(MinMaxScaler(), final_model)
pipeline_final.fit(X_train, y_train)

y_pred = pipeline_final.predict(X_test)
print(f"Acurácia no conjunto teste: {accuracy_score(y_test, y_pred):.4f}")

#%% Salvando modelo e scaler

import os

os.makedirs("api/MachineLearning/models", exist_ok=True)
os.makedirs("api/MachineLearning/scalers", exist_ok=True)
os.makedirs("api/MachineLearning/pipelines", exist_ok=True)
os.makedirs("api/MachineLearning/data", exist_ok=True)

with open("api/MachineLearning/models/rf_iris_classifier.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("api/MachineLearning/scalers/minmax_scaler_iris.pkl", "wb") as f:
    pickle.dump(pipeline_final.named_steps["minmaxscaler"], f)

with open("api/MachineLearning/pipelines/rf_iris_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline_final, f)

pd.DataFrame(X_test, columns=dataset.columns[:-1]).to_csv("api/MachineLearning/data/X_test_iris.csv", index=False)
pd.DataFrame(y_test, columns=[dataset.columns[-1]]).to_csv("api/MachineLearning/data/y_test_iris.csv", index=False)

#%% Simulação de predição em dados novos

new_data = pd.DataFrame({
    "SepalLengthCm": [5.1, 6.2, 5.9, 4.7],
    "SepalWidthCm": [3.5, 2.8, 3.0, 3.2],
    "PetalLengthCm": [1.4, 4.8, 4.2, 1.3],
    "PetalWidthCm": [0.2, 1.8, 1.3, 0.2]
})

X_new = new_data.values.astype(float)
X_new_scaled = pipeline_final.named_steps["minmaxscaler"].transform(X_new)
predictions = final_model.predict(X_new_scaled)
print("Previsões para novos dados:", predictions)