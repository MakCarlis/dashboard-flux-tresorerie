
# Importation des bibliothèques
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time


# Chargement du modèle + colonnes
@st.cache_resource
def load_model():
    with open("reg.pkl", "rb") as file:
        model, model_columns = pickle.load(file)
    return model, model_columns

model, model_columns = load_model()


# Configuration de la page
st.set_page_config(page_title="Prévision du Flux de Trésorerie", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.title("💰 Dashboard de Prévision du Flux de Trésorerie Bancaire au Cameroun")
st.markdown("Simulation basée sur un modèle de Machine Learning")


# Ajout d'une animation simple
with st.spinner("Chargement du modèle...🌸🌸🌸"):
    time.sleep(1)


# Entrée utilisateur (SIDEBAR)
st.sidebar.header("📌Paramètres ⚙ d'entrée 🧩🕹")

revenu = st.sidebar.number_input("Revenu", min_value=0.0, value=10000.0)
depenses = st.sidebar.number_input("Dépenses", min_value=0.0, value=8000.0)
taux_interet = st.sidebar.slider("Taux d'intérêt (%)", 0.0, 20.0, 5.0)
capital = st.sidebar.number_input("Capital", min_value=0.0, value=50000.0)

banque = st.sidebar.selectbox("Banque", ["UBA", "Ecobank", "Société Générale"])
agence = st.sidebar.selectbox("Agence", ["Agence_Nord", "Agence_Sud"])
lieu = st.sidebar.selectbox("Ville", ["Douala", "Yaoundé", "Garoua"])


# Features Engineering
marge = (revenu - depenses) / (revenu + 1)
ratio_dep_rev = depenses / (revenu + 1)
charge_interet = taux_interet * capital


# Création du dataframe INPUT (rien que les variables numériques)
data = {
    "revenu": revenu,
    "depenses": depenses,
    "taux_interet": taux_interet,
    "capital": capital,
    "marge": marge,
    "ratio_dep_rev": ratio_dep_rev,
    "charge_interet": charge_interet
}

df_input = pd.DataFrame([data])


# Création des colonnes (encodées manuellement)

## Banque
df_input["banque_Ecobank"] = 1 if banque == "Ecobank" else 0
df_input["banque_BGFI"] = 1 if banque == "BGFI" else 0
df_input["banque_Société Générale"] = 1 if banque == "Société Générale" else 0
# UBA = référence, donc tout à 0

## Agence
df_input["agence_Agence_Nord"] = 1 if agence == "Agence_Nord" else 0
df_input["agence_Agence_Centre"] = 1 if agence == "Agence_Centre" else 0
# Agence_Sud = référence

## Lieu
df_input["lieu_Douala"] = 1 if lieu == "Douala" else 0
df_input["lieu_Yaoundé"] = 1 if lieu == "Yaoundé" else 0
df_input["lieu_Garoua"] = 1 if lieu == "Garoua" else 0
# Bafoussam = référence


# Important : L'alignement avec les colonnes du modèle (pour éviter des erreurs si colonnes manquantes)
## En principe : model_columns = model.feature_names_in_
## Mais model_columns a déjà été chargé depuis pickle.


for col in model_columns:
    if col not in df_input.columns:
        df_input[col] = 0

df_input = df_input[model_columns]


# BOUTON DE PREDICTION

if st.button("🚀 Lancer la prédiction ⚖"):

    # Animation chargement
    with st.spinner("Calcul en cours...🌸🌸🌸"):
        time.sleep(1.5)

    prediction = model.predict(df_input)[0]
   
    # Résultat
    st.success("✅ Prédiction réalisée !! 📡")

    st.metric(
        label="Flux de trésorerie estimé",
        value=f"{prediction:,.2f} FCFA"
    )

    # INTERPRETATION SIMPLE
    if prediction > 0:
        st.info("📈 Situation positive : flux entrant supérieur aux sorties 💸")
    else:
        st.warning("⚠️ Situation négative : attention à la trésorerie 💸")

    # VISUALISATION
    chart_data = pd.DataFrame({
        "Catégorie": ["Revenu", "Dépenses", "Flux prédit"],
        "Valeur": [revenu, depenses, prediction]
    })

    st.bar_chart(chart_data.set_index("Catégorie"))

    # ANIMATION BONUS
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    st.balloons()


# COMPARAISON MULTI-SCÉNARIOS

st.markdown("## 🔄 Comparaison de scénarios")

nb_scenarios = st.slider("Nombre de scénarios à comparer", 1, 5, 2)

scenarios = []

for i in range(nb_scenarios):
    st.markdown(f"### Scénario {i+1}")

    col1, col2 = st.columns(2)

    with col1:
        rev = st.number_input(f"Revenu {i}", value=10000.0, key=f"rev{i}")
        dep = st.number_input(f"Dépenses {i}", value=8000.0, key=f"dep{i}")
    
    with col2:
        taux = st.slider(f"Taux {i}", 0.0, 20.0, 5.0, key=f"taux{i}")
        cap = st.number_input(f"Capital {i}", value=50000.0, key=f"cap{i}")

    # Features
    marge_i = (rev - dep) / (rev + 1)
    ratio_i = dep / (rev + 1)
    charge_i = taux * cap

    df_temp = pd.DataFrame([{
        "revenu": rev,
        "depenses": dep,
        "taux_interet": taux,
        "capital": cap,
        "marge": marge_i,
        "ratio_dep_rev": ratio_i,
        "charge_interet": charge_i
    }])

  
# Encodage manuel

# Banque
df_temp["banque_Ecobank"] = 1 if banque == "Ecobank" else 0
df_temp["banque_BGFI"] = 1 if banque == "BGFI" else 0
df_temp["banque_Société Générale"] = 1 if banque == "Société Générale" else 0
# UBA = référence, donc tout à 0

# Agence
df_temp["agence_Agence_Nord"] = 1 if agence == "Agence_Nord" else 0
df_temp["agence_Agence_Centre"] = 1 if agence == "Agence_Centre" else 0
# Agence_Sud = référence

# Lieu
df_temp["lieu_Douala"] = 1 if lieu == "Douala" else 0
df_temp["lieu_Yaoundé"] = 1 if lieu == "Yaoundé" else 0
df_temp["lieu_Garoua"] = 1 if lieu == "Garoua" else 0
# Bafoussam = référence


# Alignement colonnes
for col in model_columns:
    if col not in df_temp.columns:
        df_temp[col] = 0
df_temp = df_temp[model_columns]

pred = model.predict(df_temp)[0]

scenarios.append({
     "Scénario": f"S{i+1}",
     "Flux prédit": pred
})


# Visualisation Comparative
df_scenarios = pd.DataFrame(scenarios)
st.bar_chart(df_scenarios.set_index("Scénario"))


# Interprêtation Autpmatique
best = df_scenarios.loc[df_scenarios["Flux prédit"].idxmax()]
st.success(f"🏆 Meilleur scénario : {best['Scénario']} avec {best['Flux prédit']:.2f}")


# Simulation Dynamique (SLIDER)

st.markdown("## 🎛️ Simulation dynamique")

revenu_sim = st.slider("Faire varier le revenu", 1000, 20000, 10000)

marge_sim = (revenu_sim - depenses) / (revenu_sim + 1)
ratio_sim = depenses / (revenu_sim + 1)
charge_sim = taux_interet * capital

df_sim = pd.DataFrame([{
    "revenu": revenu_sim,
    "depenses": depenses,
    "taux_interet": taux_interet,
    "capital": capital,
    "marge": marge_sim,
    "ratio_dep_rev": ratio_sim,
    "charge_interet": charge_sim
}])


# Encodage fixe

# Banque
df_sim["banque_Ecobank"] = 1 if banque == "Ecobank" else 0
df_sim["banque_BGFI"] = 1 if banque == "BGFI" else 0
df_sim["banque_Société Générale"] = 1 if banque == "Société Générale" else 0

# Agence
df_sim["agence_Agence_Nord"] = 1 if agence == "Agence_Nord" else 0
df_sim["agence_Agence_Centre"] = 1 if agence == "Agence_Centre" else 0

# Lieu
df_sim["lieu_Douala"] = 1 if lieu == "Douala" else 0
df_sim["lieu_Yaoundé"] = 1 if lieu == "Yaoundé" else 0
df_sim["lieu_Garoua"] = 1 if lieu == "Garoua" else 0


# Alignement des colonnes !

for col in model_columns:
    if col not in df_sim.columns:
        df_sim[col] = 0

df_sim = df_sim[model_columns]

# Prédiction simulation
pred_sim = model.predict(df_sim)[0]

# Calcul de la prédiction de base
pred_base = model.predict(df_input)[0]

# Graphique
st.line_chart(pd.DataFrame({
    "Revenu": [revenu, revenu_sim],
    "Flux": [pred_base, pred_sim]
}))


# Section INSIGHTS (Interprétation intelligente des résultats)

st.markdown("## 📊 Insights automatiques")

if pred_sim > pred_base:
    st.info("📈 Augmenter le revenu améliore le flux de trésorerie")
else:
    st.warning("⚠️ Augmenter le revenu seul ne suffit pas ici")


# Animation Bonus
st.markdown("## 🎉 Simulation terminée")

progress = st.progress(0)
for i in range(100):
    time.sleep(0.005)
    progress.progress(i + 1)

st.snow()


# Pied-de-page (FOOTER)
st.markdown("---")
st.markdown("📊 Projet de Machine Learning - Dashboard interactif by CARLIS - Prévisions des Flux de Trésorerie 💸")