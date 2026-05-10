# Importation des bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time


# Chargement du modèle + colonnes
@st.cache_resource
def load_model():
    with open("reg.pkl", "rb") as file:
        model, model_columns = pickle.load(file)
    return model, model_columns

model, model_columns = load_model()


# Configuration
st.set_page_config(page_title="Prévision Flux", layout="wide")

# CSS PRO (pour un meilleur visuel)
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #1E88E5;
}

.stButton>button {
    background-color: #1E88E5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #1565C0;
    color: white;
}

.css-1d391kg {
    background-color: #ffffff;
}

.metric-box {
    background-color: #e3f2fd;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>💰 Dashboard Flux de Trésorerie</h1>", unsafe_allow_html=True)


# Navigation
page = st.sidebar.radio("Navigation", [
    "📊 Prédiction",
    "🔄 Scénarios",
    "🎛️ Simulation"
])


# =========================
# 📊 PAGE 1 : PREDICTION
# =========================
if page == "📊 Prédiction":

    st.caption("⚠️ Prédiction illustrative basée sur un modèle à très faibles performances")

    revenu = st.number_input("Revenu", 0.0, 100000.0, 10000.0)
    depenses = st.number_input("Dépenses", 0.0, 100000.0, 8000.0)
    taux_interet = st.slider("Taux (%)", 0.0, 20.0, 5.0)
    capital = st.number_input("Capital", 0.0, 100000.0, 50000.0)

    banque = st.selectbox("Banque", ["UBA", "Ecobank", "BGFI", "Société Générale"])
    agence = st.selectbox("Agence", ["Agence_Sud", "Agence_Nord", "Agence_Centre"])
    lieu = st.selectbox("Ville", ["Bafoussam", "Douala", "Yaoundé", "Garoua"])

    # Features
    marge = (revenu - depenses) / (revenu + 1)
    ratio = depenses / (revenu + 1)
    charge = taux_interet * capital

    df = pd.DataFrame([{
        "revenu": revenu,
        "depenses": depenses,
        "taux_interet": taux_interet,
        "capital": capital,
        "marge": marge,
        "ratio_dep_rev": ratio,
        "charge_interet": charge
    }])

    # Encodage
    df["banque_Ecobank"] = 1 if banque == "Ecobank" else 0
    df["banque_BGFI"] = 1 if banque == "BGFI" else 0
    df["banque_Société Générale"] = 1 if banque == "Société Générale" else 0

    df["agence_Agence_Nord"] = 1 if agence == "Agence_Nord" else 0
    df["agence_Agence_Centre"] = 1 if agence == "Agence_Centre" else 0

    df["lieu_Douala"] = 1 if lieu == "Douala" else 0
    df["lieu_Yaoundé"] = 1 if lieu == "Yaoundé" else 0
    df["lieu_Garoua"] = 1 if lieu == "Garoua" else 0

    # Alignement
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    if st.button("🚀 Prédire"):

        with st.spinner("Analyse en cours... 🔍"):
            time.sleep(1)

        pred = model.predict(df)[0]

        st.markdown(f"""
        <div style="
            background-color: rgba(30,136,229,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: inherit;
        ">
            💰 Flux estimé : {pred:,.2f} FCFA
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Revenu", f"{revenu:,.0f}")
        col2.metric("Dépenses", f"{depenses:,.0f}")
        col3.metric("Flux", f"{pred:,.0f}")

        st.bar_chart(pd.DataFrame({
            "Valeur": [revenu, depenses, pred]
        }, index=["Revenu", "Dépenses", "Flux"]))

        st.balloons()


# =========================
# 🔄 PAGE 2 : SCENARIOS
# =========================
elif page == "🔄 Scénarios":

    nb = st.slider("Nombre de scénarios", 1, 5, 2)
    results = []

    for i in range(nb):
        st.subheader(f"Scénario {i+1}")

        rev = st.number_input(f"Revenu {i}", 0.0, 100000.0, 10000.0, key=i)
        dep = st.number_input(f"Dépenses {i}", 0.0, 100000.0, 8000.0, key=i+10)

        marge = (rev - dep) / (rev + 1)
        ratio = dep / (rev + 1)

        df = pd.DataFrame([{
            "revenu": rev,
            "depenses": dep,
            "taux_interet": 5,
            "capital": 50000,
            "marge": marge,
            "ratio_dep_rev": ratio,
            "charge_interet": 5 * 50000
        }])

        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]

        pred = model.predict(df)[0]
        results.append(pred)

    ## Graphique des résultats
    best_index = np.argmax(results)
    st.success(f"🏆 Meilleur scénario : S{best_index+1}")
    st.progress(int((max(results)/ (sum(results)+1)) * 100))


# =========================
# 🎛️ PAGE 3 : SIMULATION
# =========================
elif page == "🎛️ Simulation":

    st.caption("⚠️ Simulation illustrative basée sur un modèle à très faibles performances")

    # Variables nécessaires
    depenses = st.number_input("Dépenses", value=8000.0)
    taux_interet = st.slider("Taux d'intérêt", 0.0, 20.0, 5.0)
    capital = st.number_input("Capital", value=50000.0)

    # Slider pour point dynamique
    revenu = st.slider("Revenu (point sélectionné)", 1000, 20000, 10000)

    # Génération de la courbe
    revenus = np.linspace(1000, 20000, 50)
    flux = []

    for r in revenus:
        marge = (r - depenses) / (r + 1)
        ratio = depenses / (r + 1)

        df_temp = pd.DataFrame([{
            "revenu": r,
            "depenses": depenses,
            "taux_interet": taux_interet,
            "capital": capital,
            "marge": marge,
            "ratio_dep_rev": ratio,
            "charge_interet": taux_interet * capital
        }])

        for col in model_columns:
            if col not in df_temp.columns:
                df_temp[col] = 0

        df_temp = df_temp[model_columns]

        flux.append(model.predict(df_temp)[0])

    # Graphique principal
    df_plot = pd.DataFrame({
        "Revenu": revenus,
        "Flux": flux
    })

    st.line_chart(df_plot.set_index("Revenu"))

    # 👉 POINT DYNAMIQUE (IMPORTANT)
    marge = (revenu - depenses) / (revenu + 1)
    ratio = depenses / (revenu + 1)

    df_single = pd.DataFrame([{
        "revenu": revenu,
        "depenses": depenses,
        "taux_interet": taux_interet,
        "capital": capital,
        "marge": marge,
        "ratio_dep_rev": ratio,
        "charge_interet": taux_interet * capital
    }])

    for col in model_columns:
        if col not in df_single.columns:
            df_single[col] = 0

    df_single = df_single[model_columns]

    pred = model.predict(df_single)[0]

    # Affichage stylé (dark/light automatique)
    st.markdown(f"""
    <div style="
        background-color: rgba(30,136,229,0.1);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: inherit;
    ">
        📊 Flux simulé : {pred:,.2f} FCFA
    </div>
    """, unsafe_allow_html=True)

    # Insight simple
    if pred > 0:
        st.success("📈 Bonne performance financière")
    else:
        st.error("⚠️ Risque de trésorerie")

    # Animation progressive
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    st.snow()