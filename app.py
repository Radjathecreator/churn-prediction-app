import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# CONFIG PAGE
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# CHARGEMENT MODÈLE
@st.cache_resource
def load_model():
    model   = joblib.load('churn_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

model, scaler, columns = load_model()

# FEATURE ENGINEERING
def oui_non(val):
    return "Yes" if val == "Oui" else "No"

def preparer_client_individuel(gender, senior, partner, dependents,
                                tenure, phone_service, multiple_lines,
                                internet, online_security, online_backup,
                                device_protect, tech_support, streaming_tv,
                                streaming_movies, contract, paperless,
                                payment, monthly_charges, total_charges):
    client = {
        'gender'           : gender,
        'SeniorCitizen'    : 1 if senior == "Oui" else 0,
        'Partner'          : oui_non(partner),
        'Dependents'       : oui_non(dependents),
        'tenure'           : tenure,
        'PhoneService'     : oui_non(phone_service),
        'MultipleLines'    : "Yes" if multiple_lines == "Oui"
                             else "No" if multiple_lines == "Non"
                             else multiple_lines,
        'InternetService'  : internet,
        'OnlineSecurity'   : "Yes" if online_security == "Oui"
                             else "No" if online_security == "Non"
                             else online_security,
        'OnlineBackup'     : "Yes" if online_backup == "Oui"
                             else "No" if online_backup == "Non"
                             else online_backup,
        'DeviceProtection' : "Yes" if device_protect == "Oui"
                             else "No" if device_protect == "Non"
                             else device_protect,
        'TechSupport'      : "Yes" if tech_support == "Oui"
                             else "No" if tech_support == "Non"
                             else tech_support,
        'StreamingTV'      : "Yes" if streaming_tv == "Oui"
                             else "No" if streaming_tv == "Non"
                             else streaming_tv,
        'StreamingMovies'  : "Yes" if streaming_movies == "Oui"
                             else "No" if streaming_movies == "Non"
                             else streaming_movies,
        'Contract'         : contract,
        'PaperlessBilling' : oui_non(paperless),
        'PaymentMethod'    : payment,
        'MonthlyCharges'   : monthly_charges,
        'TotalCharges'     : total_charges
    }

    client['ChargePerMonth'] = total_charges / (tenure + 1)
    client['HighRisk']       = int(
        senior == "Oui" and
        contract == "Month-to-month" and
        tech_support == "Non"
    )
    services = [phone_service, online_security, online_backup,
                device_protect, tech_support,
                streaming_tv, streaming_movies]
    client['NbServices']    = sum(1 for s in services if s == "Oui")
    client['LowEngagement'] = int(
        contract == "Month-to-month" and
        payment == "Electronic check"
    )

    df = pd.DataFrame([client])
    df = pd.get_dummies(df, dtype=int)
    df = df.reindex(columns=columns, fill_value=0)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'ChargePerMonth', 'NbServices']
    df[num_cols] = scaler.transform(df[num_cols])
    return df

def preparer_masse(df_raw):
    df = df_raw.copy()

    # Feature Engineering
    df['ChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['HighRisk']       = (
        (df['SeniorCitizen'] == 1) &
        (df['Contract'] == 'Month-to-month') &
        (df['TechSupport'] == 'No')
    ).astype(int)

    services = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']
    df['NbServices']    = df[services].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    df['LowEngagement'] = (
        (df['Contract'] == 'Month-to-month') &
        (df['PaymentMethod'] == 'Electronic check')
    ).astype(int)

    # Supprimer colonnes inutiles si présentes
    for col in ['customerID', 'Churn']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encoding
    df = pd.get_dummies(df, dtype=int)
    df = df.reindex(columns=columns, fill_value=0)

    # Normalisation
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'ChargePerMonth', 'NbServices']
    df[num_cols] = scaler.transform(df[num_cols])
    return df

def categorie_risque(score):
    if score >= 60:
        return '🔴 Risque élevé'
    elif score >= 30:
        return '🟡 Risque modéré'
    else:
        return '🟢 Faible risque'

# ==============================
# HEADER
# ==============================
st.title("📡 Churn Predictor — Télécom")
st.markdown("Prédisez le risque de départ de vos clients pour mieux les fidéliser.")
st.divider()

# ==============================
# ONGLETS
# ==============================
tab1, tab2 = st.tabs(["👤 Profil individuel", "📂 Analyse par lot"])

# ==============================
# ONGLET 1 — CLIENT INDIVIDUEL
# ==============================
with tab1:
    st.info("👈 Remplissez les informations du client dans le menu latéral puis cliquez sur Analyser.")
    
with tab1:
    with st.sidebar:
        st.header("👤 Informations client")

        st.subheader("Profil")
        gender     = st.selectbox("Genre", ["Male", "Female"])
        senior     = st.selectbox("Senior ?", ["Non", "Oui"])
        partner    = st.selectbox("Partenaire ?", ["Non", "Oui"])
        dependents = st.selectbox("Dépendants ?", ["Non", "Oui"])
        tenure     = st.slider("Ancienneté (mois)", 0, 72, 12)

        st.subheader("Services")
        phone_service    = st.selectbox("Téléphone", ["Non", "Oui"])
        multiple_lines   = st.selectbox("Lignes multiples",
                                        ["Non", "Oui", "No phone service"])
        internet         = st.selectbox("Internet",
                                        ["DSL", "Fiber optic", "No"])
        online_security  = st.selectbox("Sécurité en ligne",
                                        ["Non", "Oui", "No internet service"])
        online_backup    = st.selectbox("Sauvegarde",
                                        ["Non", "Oui", "No internet service"])
        device_protect   = st.selectbox("Protection appareil",
                                        ["Non", "Oui", "No internet service"])
        tech_support     = st.selectbox("Support technique",
                                        ["Non", "Oui", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV",
                                        ["Non", "Oui", "No internet service"])
        streaming_movies = st.selectbox("Streaming Films",
                                        ["Non", "Oui", "No internet service"])

        st.subheader("Contrat")
        contract         = st.selectbox("Type de contrat",
                                        ["Month-to-month",
                                         "One year", "Two year"])
        paperless        = st.selectbox("Facturation sans papier",
                                        ["Non", "Oui"])
        payment          = st.selectbox("Paiement",
                                        ["Electronic check",
                                         "Mailed check",
                                         "Bank transfer (automatic)",
                                         "Credit card (automatic)"])
        monthly_charges  = st.slider("Charges mensuelles (USD)", 18, 118, 65)
        total_charges    = st.slider("Charges totales (USD)", 0, 8500, 1500)

        analyser = st.button("🔍 Analyser ce profil",
                             use_container_width=True,
                             type="primary")

    if analyser:
        df_client = preparer_client_individuel(
            gender, senior, partner, dependents, tenure,
            phone_service, multiple_lines, internet,
            online_security, online_backup, device_protect,
            tech_support, streaming_tv, streaming_movies,
            contract, paperless, payment,
            monthly_charges, total_charges
        )

        proba = model.predict_proba(df_client)[0][1] * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Modèle", "XGBoost V3")
        col2.metric("AUC-ROC", "82.57%")
        col3.metric("Recall Churners", "66.31%")
        
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Score de risque", f"{proba:.1f}%")
            st.progress(int(proba))

        with col2:
            if proba >= 60:
                st.error("🔴 Risque ÉLEVÉ")
            elif proba >= 30:
                st.warning("🟡 Risque MODÉRÉ")
            else:
                st.success("🟢 Risque FAIBLE")

        with col3:
            st.metric("Ancienneté", f"{tenure} mois")
            st.metric("Charges mensuelles", f"{monthly_charges} USD")

        st.divider()

        st.subheader("💡 Action recommandée")
        if proba >= 60:
            st.error("""
            **Action immédiate requise**
            - Appel commercial dans les 48h
            - Offre de rétention personnalisée
            - Réduction sur le prochain mois
            """)
        elif proba >= 30:
            st.warning("""
            **Surveillance recommandée**
            - Email de satisfaction
            - Offre préventive
            - Vérification qualité service
            """)
        else:
            st.success("""
            **Aucune action nécessaire**
            - Client fidèle et satisfait
            - Ne pas dépenser de budget inutilement
            """)

# ==============================
# ONGLET 2 — ANALYSE PAR LOT
# ==============================
with tab2:
    st.subheader("📂 Analyse par lot")
    st.markdown("Importez un fichier CSV contenant votre base de clients. Le format des colonnes doit être identique aux données Telco d'origine.")

    # Chargement dynamique du fichier mock (si disponible) au lieu d'un DataFrame vide
    csv_mock_data = ""
    if os.path.exists("mock_clients.csv"):
        mock_df = pd.read_csv("mock_clients.csv")
        csv_mock_data = mock_df.to_csv(index=False)
    else:
        # Fallback si le fichier n'est pas trouvé
        template = pd.DataFrame(columns=[
            'customerID', 'gender', 'SeniorCitizen', 'Partner',
            'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ])
        csv_mock_data = template.to_csv(index=False)

    st.download_button(
        label="📥 Télécharger le template CSV",
        data=csv_mock_data,
        file_name="mock_clients.csv",
        mime="text/csv",
        help="Téléchargez ce fichier d'essai pré-rempli pour tester l'application."
    )

    st.divider()

    # Upload fichier
    uploaded_file = st.file_uploader(
        "Déposez votre fichier client (CSV)",
        type=['csv']
    )

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)

        st.success(f"✅ {len(df_raw)} clients importés avec succès.")
        st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("🚀 Lancer les prédictions", type="primary"):

            with st.spinner("Analyse des profils en cours..."):

                # Garder customerID pour affichage
                ids = df_raw['customerID'] if 'customerID' in df_raw.columns \
                      else pd.Series(range(len(df_raw)), name='ID')

                # Préparer et prédire
                df_prepared = preparer_masse(df_raw)
                probas       = model.predict_proba(df_prepared)[:, 1] * 100

                # Résultats
                resultats = pd.DataFrame({
                    'Client ID'      : ids.values,
                    'Score Risque %' : probas.round(1),
                    'Catégorie'      : [categorie_risque(p) for p in probas]
                }).sort_values('Score Risque %', ascending=False)

            st.success("✅ Analyse terminée !")
            st.divider()

            # Dashboard récapitulatif
            st.subheader("📊 Tableau de bord global")

            col1, col2, col3, col4 = st.columns(4)

            eleve   = (probas >= 60).sum()
            modere  = ((probas >= 30) & (probas < 60)).sum()
            faible  = (probas < 30).sum()
            moyenne = probas.mean()

            with col1:
                st.metric("🔴 Risque élevé",
                          f"{eleve} clients",
                          f"{eleve/len(probas)*100:.1f}%")
            with col2:
                st.metric("🟡 Risque modéré",
                          f"{modere} clients",
                          f"{modere/len(probas)*100:.1f}%")
            with col3:
                st.metric("🟢 Faible risque",
                          f"{faible} clients",
                          f"{faible/len(probas)*100:.1f}%")
            with col4:
                st.metric("📈 Score de churn moyen",
                          f"{moyenne:.1f}%")

            st.divider()

            # Filtre par catégorie
            st.subheader("🔍 Explorer les résultats détaillés")

            filtre = st.selectbox(
                "Filtrer par niveau de risque",
                ["Tous", "🔴 Risque élevé",
                 "🟡 Risque modéré", "🟢 Faible risque"]
            )

            if filtre != "Tous":
                df_affiche = resultats[resultats['Catégorie'] == filtre]
            else:
                df_affiche = resultats

            st.dataframe(df_affiche, use_container_width=True)
            st.markdown(f"**{len(df_affiche)} profil(s) affiché(s)**")

            st.divider()

            # Top 10 clients les plus à risque
            st.subheader("🚨 Top 10 clients prioritaires (Risque maximum)")
            st.dataframe(
                resultats.head(10),
                use_container_width=True
            )

            st.divider()

            # Export
            st.subheader("📥 Exporter le rapport")

            st.download_button(
                label="📥 Exporter tous les résultats (CSV)",
                data=resultats.to_csv(index=False),
                file_name="resultats_churn_complets.csv",
                mime="text/csv"
            )

            st.download_button(
                label="🔴 Exporter uniquement les clients à 'Risque élevé'",
                data=resultats[resultats['Catégorie']=='🔴 Risque élevé']\
                     .to_csv(index=False),
                file_name="clients_risque_eleve_prioritaires.csv",
                mime="text/csv"
            )

#footer
st.divider()
st.markdown("""
<div style='text-align: center; color: grey; font-size: 12px;'>
Churn Predictor — Modèle XGBoost V3 | AUC-ROC 82.57% | Radja Kurniawan
</div>
""", unsafe_allow_html=True)
