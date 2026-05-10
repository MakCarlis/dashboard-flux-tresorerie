### Page d'accueil avec navigation multi-pages

# Importation des bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO
import pickle
import time


# Chargement du modèle + colonnes
@st.cache_resource
def load_model():
    with open("reg.pkl", "rb") as file:
        model, model_columns = pickle.load(file)
    return model, model_columns

model, model_columns = load_model()


# Pour utiliser notre fichier .css
with open('C:/Users/PC/OneDrive/Documents/IA_et_ML/Projet_Final-TP2/TP2_machine-learning/app_streamlit/assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Configuration de la page
st.set_page_config(page_title="Prévision du Flux de Trésorerie", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.title("💰 Dashboard Prévisionnel des Flux de Trésorerie Bancaire au Cameroun")
# En-tête
st.markdown('<p class="sub-header">Simulation basée sur un modèle de Machine Learning</p>', unsafe_allow_html=True)

# Ajout d'une animation simple
with st.spinner("Chargement du modèle...🌸🌸🌸"):
    time.sleep(1)


# CSS personnalisé pour un meilleur rendu
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .card-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 10px;
    }
    .card-desc {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 15px;
    }
    .metric-highlight {
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
local_css()


# Barre latérale pour la navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)  ## ????????????????
st.sidebar.title("Navigation")
st.sidebar.markdown("---")


# Menu de navigation
page = st.sidebar.radio(
    "Choisissez une section :",
    ["Accueil", "📊 Prévision du Flux de Trésorerie", 
     "🔄 Comparaison de scénarios", 
     "🎛️ Simulation dynamique"],
)


# Affichage conditionnel des pages

if page == "Accueil":
    # Affichage des cartes de présentation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Section 1</div>
            <div class="card-desc"><strong>Prévisions du Flux de Trésorerie</strong></div>
            <p>Prédiction du flux de trésorerie dans quelques banques du Cameroun à partir des variables indiquée.</p>
            <p><strong>Modèle :</strong> Random Forest Regressor</p>
            <p><strong>Meilleur modèle :</strong> Bien que très mal prédictif, faute de variables imposées non explicatives (TEST R² = 0.0048)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔄 Section 2</div>
            <div class="card-desc"><strong>Comparaison de scénarios</strong></div>
            <p>Identifier le meilleur scénario parmi 2 à 5 autres.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">🎛️ Section 3</div>
            <div class="card-desc"><strong>Simulation dynamique</strong></div>
            <p>Simulation dynamique du flux de trésorerie en fonction du revenu.</p>
            <p>Et Insights automatiques.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistiques globales
    st.subheader("📈 Aperçu du dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombres d'instances", "1000")
    with col2:
        st.metric("Nombre de banques", "4")
    with col3:
        st.metric("Nombre de villes", "4")
    
    # Instructions d'utilisation
    with st.expander("ℹ️ Comment utiliser ce dashboard ?"):
        st.markdown("""
        **Navigation :**
        - Utilisez le menu dans la barre latérale pour accéder aux différentes parties
        - Chaque partie propose une interface interactive de prédiction
        
        **Pour la Section 1 (Prédictions) :**
        - Ajustez les caractéristiques
        - Obtenez la prédiction du Flux de Trésorerie recherchée (bien que peu de chance qu'elle soit correcte)
        
        **Pour la Section 2 (Comparaison de scénarios) :**
        - Définissez le nombre de scénarios à comparer
        - Obtenez le score du meilleur scénario
        
        **Pour la Section 3 (Simulation dynamique) :**
        - Faire varier le revenu
        - Estimez le Flux de Trésorerie potentiel
        - Obtenir des insights (bien que décoratifs -pas fiables métier- car le modèle n'est pas bon)
        """)
    
    # Pied-de-page (FOOTER)
    st.markdown("---")
    st.markdown("📊 Projet de Machine Learning - Dashboard interactif by CARLIS - Prévisions des Flux de Trésorerie 💸")

else:
    # Redirection vers les pages correspondantes
    if page == "📊 Prévision du Flux de Trésorerie":
        
        st.markdown("## 📊 Prévision du Flux de Trésorerie")

        # Entrées utilisateur
        revenu = st.number_input("Revenu", min_value=0.0)
        depenses = st.number_input("Dépenses", min_value=0.0)
        taux_interet = st.slider("Taux d'intérêt (%)", 0.0, 20.0)
        capital = st.number_input("Capital", min_value=0.0)

        # Features
        marge = (revenu - depenses) / (revenu + 1)
        ratio_dep_rev = depenses / (revenu + 1)
        charge_interet = taux_interet * capital

        df_input = pd.DataFrame([{
            "revenu": revenu,
            "depenses": depenses,
            "taux_interet": taux_interet,
            "capital": capital,
            "marge": marge,
            "ratio_dep_rev": ratio_dep_rev,
            "charge_interet": charge_interet
        }])

        df_input = df_input[model_columns]

        if st.button("🚀 Lancer la prédiction"):

            with st.spinner("Calcul en cours..."):
                time.sleep(1)

            pred = model.predict(df_input)[0]

            st.success(f"Flux estimé : {pred:,.2f} FCFA")

            st.bar_chart(pd.DataFrame({
                "Valeur": [revenu, depenses, pred]
            }, index=["Revenu", "Dépenses", "Flux"]))


    elif page == "🔄 Comparaison de scénarios":
        
        st.markdown("## 🔄 Comparaison de scénarios")

        nb = st.slider("Nombre de scénarios", 1, 5, 2)

        results = []

        for i in range(nb):

            st.markdown(f"### Scénario {i+1}")

            rev = st.number_input(f"Revenu {i}", value=10000.0, key=f"rev{i}")
            dep = st.number_input(f"Dépenses {i}", value=8000.0, key=f"dep{i}")
            taux = st.slider(f"Taux {i}", 0.0, 20.0, 5.0, key=f"taux{i}")
            cap = st.number_input(f"Capital {i}", value=50000.0, key=f"cap{i}")

            marge = (rev - dep) / (rev + 1)
            ratio = dep / (rev + 1)
            charge = taux * cap

            df_temp = pd.DataFrame([{
                "revenu": rev,
                "depenses": dep,
                "taux_interet": taux,
                "capital": cap,
                "marge": marge,
                "ratio_dep_rev": ratio,
                "charge_interet": charge
            }])

            pred = model.predict(df_temp)[0]

            results.append(pred)

        st.bar_chart(pd.DataFrame(results, columns=["Flux"]))

        st.success(f"Meilleur scénario : S{np.argmax(results)+1}")


    elif page == "🎛️ Simulation dynamique":
        
        st.markdown("## 🎛️ Simulation dynamique")

        revenu = st.slider("Revenu", 1000, 20000, 10000)
        depenses = 8000
        taux_interet = 5.0
        capital = 50000

        marge = (revenu - depenses) / (revenu + 1)
        ratio = depenses / (revenu + 1)
        charge = taux_interet * capital

        df_sim = pd.DataFrame([{
            "revenu": revenu,
            "depenses": depenses,
            "taux_interet": taux_interet,
            "capital": capital,
            "marge": marge,
            "ratio_dep_rev": ratio,
            "charge_interet": charge
        }])

        pred = model.predict(df_sim)[0]

        st.line_chart(pd.DataFrame({
            "Flux": [pred]
        }))

        st.markdown("## 📊 Insights automatiques")

        if pred > 0:
            st.info("📈 Flux positif")
        else:
            st.warning("⚠️ Flux négatif")

        st.snow()

    