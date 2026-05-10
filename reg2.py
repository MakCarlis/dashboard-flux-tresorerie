# Importation des bibliothèques
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

# Chargement du modèle (pipeline complet)
@st.cache_resource
def load_model():
    with open("reg.pkl", "rb") as file:
        model, model_columns = pickle.load(file)  ## On récupère les 2 objets
    return model, model_columns

model, model_columns = load_model()

# Configuration de la page
st.set_page_config(page_title="Prévision du Flux de Trésorerie", page_icon="📊", layout="wide")

st.markdown("<h1 style='text-align: center;'>💰 Dashboard de Prévision du Flux de Trésorerie</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Version production-ready 🚀</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📌 Paramètres")

revenu = st.sidebar.number_input("Revenu", 0.0, value=10000.0)
depenses = st.sidebar.number_input("Dépenses", 0.0, value=8000.0)
taux_interet = st.sidebar.slider("Taux (%)", 0.0, 20.0, 5.0)
capital = st.sidebar.number_input("Capital", 0.0, value=50000.0)

banque = st.sidebar.selectbox("Banque", ["UBA", "Ecobank", "BGFI", "Société Générale"])
agence = st.sidebar.selectbox("Agence", ["Agence_Sud", "Agence_Nord", "Agence_Centre"])
lieu = st.sidebar.selectbox("Ville", ["Bafoussam", "Douala", "Yaoundé", "Garoua"])

# DataFrame brut (PAS d'encodage manuel 🔥)
df_input = pd.DataFrame([{
    "revenu": revenu,
    "depenses": depenses,
    "taux_interet": taux_interet,
    "capital": capital,
    "banque": banque,
    "agence": agence,
    "lieu": lieu
}])

df_input = df_input.drop(columns=["Banque", "Agence", "lieu"], errors="ignore")


## Encodage manuel

# Banque
df_input["banque_Ecobank"] = 1 if banque == "Ecobank" else 0
df_input["banque_BGFI"] = 1 if banque == "BGFI" else 0
df_input["banque_Société Générale"] = 1 if banque == "Société Générale" else 0
# UBA = référence

# Agence
df_input["agence_Agence_Nord"] = 1 if agence == "Agence_Nord" else 0
df_input["agence_Agence_Sud"] = 1 if agence == "Agence_Sud" else 0
# Agence_Centre = référence

# Lieu
df_input["lieu_Douala"] = 1 if lieu == "Douala" else 0
df_input["lieu_Yaoundé"] = 1 if lieu == "Yaoundé" else 0
df_input["lieu_Garoua"] = 1 if lieu == "Garoua" else 0
# Bafoussam = référence


# Onglets
tab1, tab2, tab3 = st.tabs([
    "1. ⚙️ Prédiction",
    "2. 🔄 Scénarios",
    "3. 🎛️ Simulation + Insights"
])

# ================= TAB 1 =================
with tab1:

    if st.button("🚀 Lancer la prédiction"):

        with st.spinner("Calcul..."):
            time.sleep(1)

        pred = model.predict(df_input)[0]

        st.metric("Flux estimé", f"{pred:,.2f} FCFA")

        st.bar_chart(pd.DataFrame({
            "Valeur": [revenu, depenses, pred]
        }, index=["Revenu", "Dépenses", "Flux"]))

# ================= TAB 2 =================
with tab2:

    st.markdown("## 🔄 Comparaison")

    n = st.slider("Nb scénarios", 1, 5, 2)

    results = []

    for i in range(n):

        st.markdown(f"### Scénario {i+1}")

        rev = st.number_input(f"Revenu {i}", value=10000.0, key=f"r{i}")
        dep = st.number_input(f"Dépenses {i}", value=8000.0, key=f"d{i}")
        taux = st.slider(f"Taux {i}", 0.0, 20.0, 5.0, key=f"t{i}")
        cap = st.number_input(f"Capital {i}", value=50000.0, key=f"c{i}")

        df_temp = pd.DataFrame([{
            "revenu": rev,
            "depenses": dep,
            "taux_interet": taux,
            "capital": cap,
            "banque": banque,
            "agence": agence,
            "lieu": lieu
        }])
        

        # Encodage manuel (OBLIGATOIRE)

        # Banque
        df_temp["banque_Ecobank"] = 1 if banque == "Ecobank" else 0
        df_temp["banque_BGFI"] = 1 if banque == "BGFI" else 0
        df_temp["banque_Société Générale"] = 1 if banque == "Société Générale" else 0
        # UBA = référence

        # Agence
        df_temp["agence_Agence_Nord"] = 1 if agence == "Agence_Nord" else 0
        df_temp["agence_Agence_Sud"] = 1 if agence == "Agence_Sud" else 0
        # Centre = référence

        # Lieu
        df_temp["lieu_Douala"] = 1 if lieu == "Douala" else 0
        df_temp["lieu_Yaoundé"] = 1 if lieu == "Yaoundé" else 0
        df_temp["lieu_Garoua"] = 1 if lieu == "Garoua" else 0
        # Bafoussam = référence

        # ALIGNEMENT FINAL
        for col in model_columns:
            if col not in df_temp.columns:
                df_temp[col] = 0

        df_temp = df_temp[model_columns]


        pred = model.predict(df_temp)[0]

        results.append(pred)

    st.bar_chart(pd.DataFrame(results, columns=["Flux"]))

    best = np.argmax(results)
    st.success(f"🏆 Meilleur scénario : S{best+1}")

# ================= TAB 3 =================
with tab3:

    st.markdown("## 🎛️ Simulation")

    revenu_sim = st.slider("Revenu", 1000, 20000, 10000)

    df_sim = pd.DataFrame([{
        "revenu": revenu_sim,
        "depenses": depenses,
        "taux_interet": taux_interet,
        "capital": capital,
        "banque": banque,
        "agence": agence,
        "lieu": lieu
    }])

    pred_sim = model.predict(df_sim)[0]
    pred_base = model.predict(df_input)[0]

    st.line_chart(pd.DataFrame({
        "Revenu": [revenu, revenu_sim],
        "Flux": [pred_base, pred_sim]
    }))

    st.markdown("## 📊 Insights")

    if pred_sim > pred_base:
        st.info("📈 Augmenter le revenu améliore le flux")
    else:
        st.warning("⚠️ Effet limité du revenu")

    st.snow()

