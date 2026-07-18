import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# CHARGEMENT DU MODÈLE
# ==============================

@st.cache_resource
def load_model():
    model   = joblib.load('model/churn_model.pkl')
    scaler  = joblib.load('model/scaler.pkl')
    columns = joblib.load('model/columns.pkl')
    return model, scaler, columns

model, scaler, columns = load_model()

# ==============================
# INTERFACE
# ==============================

st.title("Prédiction du Churn Télécom")
st.markdown("Entrez les informations du client pour estimer son risque de départ.")

st.divider()

# ==============================
# INPUTS CLIENT
# ==============================

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Profil client")
    gender          = st.selectbox("Genre", ["Male", "Female"])
    senior          = st.selectbox("Senior ?", ["Non", "Oui"])
    partner         = st.selectbox("Partenaire ?", ["Non", "Oui"])
    dependents      = st.selectbox("Dépendants ?", ["Non", "Oui"])
    tenure          = st.slider("Ancienneté (mois)", 0, 72, 12)

with col2:
    st.subheader("📱 Services")
    phone_service   = st.selectbox("Service téléphonique", ["Non", "Oui"])
    multiple_lines  = st.selectbox("Lignes multiples", ["Non", "Oui", "No phone service"])
    internet        = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Sécurité en ligne", ["Non", "Oui", "No internet service"])
    online_backup   = st.selectbox("Sauvegarde en ligne", ["Non", "Oui", "No internet service"])
    device_protect  = st.selectbox("Protection appareil", ["Non", "Oui", "No internet service"])
    tech_support    = st.selectbox("Support technique", ["Non", "Oui", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV", ["Non", "Oui", "No internet service"])
    streaming_movies= st.selectbox("Streaming Films", ["Non", "Oui", "No internet service"])

with col3:
    st.subheader("💳 Contrat & Facturation")
    contract        = st.selectbox("Type de contrat", 
                                   ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Facturation sans papier", ["Non", "Oui"])
    payment         = st.selectbox("Méthode de paiement", 
                                   ["Electronic check", "Mailed check",
                                    "Bank transfer (automatic)", 
                                    "Credit card (automatic)"])
    monthly_charges = st.slider("Charges mensuelles (USD)", 18, 118, 65)
    total_charges   = st.slider("Charges totales (USD)", 0, 8500, 1500)

st.divider()

# ==============================
# PRÉDICTION
# ==============================

if st.button("🔍 Analyser ce client", use_container_width=True):

    # Conversion des inputs
    def oui_non(val):
        return "Yes" if val == "Oui" else "No"

    # Construire le dictionnaire client
    client = {
        'gender'            : gender,
        'SeniorCitizen'     : 1 if senior == "Oui" else 0,
        'Partner'           : oui_non(partner),
        'Dependents'        : oui_non(dependents),
        'tenure'            : tenure,
        'PhoneService'      : oui_non(phone_service),
        'MultipleLines'     : "Yes" if multiple_lines == "Oui" else "No" if multiple_lines == "Non" else multiple_lines,
        'InternetService'   : internet,
        'OnlineSecurity'    : "Yes" if online_security == "Oui" else "No" if online_security == "Non" else online_security,
        'OnlineBackup'      : "Yes" if online_backup == "Oui" else "No" if online_backup == "Non" else online_backup,
        'DeviceProtection'  : "Yes" if device_protect == "Oui" else "No" if device_protect == "Non" else device_protect,
        'TechSupport'       : "Yes" if tech_support == "Oui" else "No" if tech_support == "Non" else tech_support,
        'StreamingTV'       : "Yes" if streaming_tv == "Oui" else "No" if streaming_tv == "Non" else streaming_tv,
        'StreamingMovies'   : "Yes" if streaming_movies == "Oui" else "No" if streaming_movies == "Non" else streaming_movies,
        'Contract'          : contract,
        'PaperlessBilling'  : oui_non(paperless),
        'PaymentMethod'     : payment,
        'MonthlyCharges'    : monthly_charges,
        'TotalCharges'      : total_charges
    }

    # Feature Engineering — mêmes variables que le modèle
    client['ChargePerMonth'] = total_charges / (tenure + 1)
    client['HighRisk'] = int(
        senior == "Oui" and 
        contract == "Month-to-month" and 
        tech_support == "Non"
    )
    services_list = [phone_service, online_security, online_backup,
                     device_protect, tech_support, 
                     streaming_tv, streaming_movies]
    client['NbServices'] = sum(1 for s in services_list if s == "Oui")
    client['LowEngagement'] = int(
        contract == "Month-to-month" and 
        payment == "Electronic check"
    )

    # Créer DataFrame
    df_client = pd.DataFrame([client])

    # One-Hot Encoding
    df_client = pd.get_dummies(df_client, dtype=int)

    # Aligner les colonnes avec le modèle
    df_client = df_client.reindex(columns=columns, fill_value=0)

    # Normalisation
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                'ChargePerMonth', 'NbServices']
    df_client[num_cols] = scaler.transform(df_client[num_cols])

    # Prédiction
    proba = model.predict_proba(df_client)[0][1] * 100
    prediction = model.predict(df_client)[0]

    # ==============================
    # RÉSULTAT
    # ==============================

    st.subheader("📊 Résultat")

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if proba >= 60:
            st.error(f"Risque ÉLEVÉ de churn : {proba:.1f}%")
        elif proba >= 30:
            st.warning(f"Risque MODÉRÉ de churn : {proba:.1f}%")
        else:
            st.success(f"Risque FAIBLE de churn : {proba:.1f}%")

    with col_res2:
        st.metric("Score de risque", f"{proba:.1f}%")
        st.progress(int(proba))

    # Action recommandée
    st.subheader("Action recommandée")
    if proba >= 60:
        st.error("""
        **Action immédiate requise**
        - Appel commercial dans les 48h
        - Proposer une offre de rétention personnalisée
        - Envisager une réduction sur le prochain mois
        """)
    elif proba >= 30:
        st.warning("""
        **Surveillance recommandée**
        - Envoyer un email de satisfaction
        - Proposer une offre préventive
        - Vérifier la qualité du service
        """)
    else:
        st.success("""
        **Aucune action nécessaire**
        - Client fidèle et satisfait
        - Ne pas dépenser de budget marketing inutilement
        """)
