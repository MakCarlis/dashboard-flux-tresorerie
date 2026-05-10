# ================================
# 📊 Projet Machine Learning - Flux Trésorerie
# Dataset : dataset_financier.csv
# Objectif : Prédire la colonne "flux_tresorerie" avec un modèle Random Forest
# ================================

# 1. Importation des librairies nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. Chargement du dataset
df = pd.read_csv("dataset_financier.csv")

# 3. Inspection initiale
print("Dimensions du dataset :", df.shape)
print("Types de colonnes :\n", df.dtypes)
print("Aperçu des données :\n", df.head())


# 4. Vérification des valeurs manquantes
print("Valeurs manquantes par colonne :\n", df.isnull().sum())

# Traitement des NA : imputation par médiane pour les colonnes numériques
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# 5. Détection et traitement des outliers (méthode IQR)
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Winsorisation : remplacer les valeurs extrêmes par les bornes
    df[col] = np.where(df[col] < lower_bound, lower_bound,
                       np.where(df[col] > upper_bound, upper_bound, df[col]))
                       

# 6. Analyse exploratoire (EDA)
print("Statistiques descriptives :\n", df.describe())

# Histogrammes
df.hist(figsize=(12,8), bins=30)
plt.suptitle("Distribution des variables numériques")
plt.show()

# Boxplots
for col in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# Matrice de corrélation
corr_matrix = df.corr()
print("Matrice de corrélation :\n", corr_matrix)

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap des corrélations")
plt.show()

# 7. Préparation des données
X = df.drop("flux_tresorerie", axis=1)   # Features
y = df["flux_tresorerie"]                # Target

# Encodage des variables catégorielles
X = pd.get_dummies(X, drop_first=True)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Modélisation Random Forest
model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42
)
model.fit(X_train, y_train)

# 9. Évaluation du modèle
# Sur train
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Sur test
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("=== Évaluation du modèle ===")
print(f"Train MSE : {train_mse:.2f}, Train R² : {train_r2:.2f}")
print(f"Test MSE  : {test_mse:.2f}, Test R²  : {test_r2:.2f}")

# 10. Importance des variables
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind="bar", figsize=(12,6))
plt.title("Importance des variables dans le modèle")
plt.show()

# ================================
# 🧑‍💼 Interprétation métier :
# - Si R² train >> R² test → sur-apprentissage (overfitting).
# - Si R² train ≈ R² test et élevé → modèle fiable pour prédire le flux de trésorerie.
# - Les variables avec forte importance sont les principaux leviers financiers
#   (ex. revenu, dépenses, taux d’intérêt, capital).
# - Les résultats permettent d’anticiper la trésorerie et d’identifier
#   les facteurs qui l’impactent le plus.
# ================================
