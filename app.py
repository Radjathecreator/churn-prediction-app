import streamlit as st
import pandas as pd
import numpy as np
import joblib
# import shab
import matplotlib.pyplot as plt

# ==============================
# CONFIG PAGE
# ==============================
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ==============================
# CHARGEMENT MODÈLE
# ==============================
@st.cache_resource
def load_model():
    model   = joblib.load('churn_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

model, scaler, columns = load_model()

# ==============================
# HEADER
# ==============================
st.title("📡 Prédiction du Churn Télécom")
st.markdown("Analysez le risque de départ d'un client en temps réel.")
st.divider()

# ==============================
# SIDEBAR — INPUTS
# ==============================
st.sidebar.header("👤 Informations client")

with st.sidebar:
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
    monthly_charges  = st.slider("Charges mensuelles (USD)", 
                                  18, 118, 65)
    total_charges    = st.slider("Charges totales (USD)", 
                                  0, 8500, 1500)

# ==============================
# PRÉDICTION
# ==============================

def oui_non(val):
    return "Yes" if val == "Oui" else "No"

def preparer_client():
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

    # Feature Engineering
    client['ChargePerMonth'] = total_charges / (tenure + 1)
    client['HighRisk']       = int(
        senior == "Oui" and
        contract == "Month-to-month" and
        tech_support == "Non"
    )
    services = [phone_service, online_security, online_backup,
                device_protect, tech_support,
                streaming_tv, streaming_movies]
    client['NbServices']     = sum(1 for s in services if s == "Oui")
    client['LowEngagement']  = int(
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

# ==============================
# BOUTON ANALYSE
# ==============================
if st.sidebar.button("🔍 Analyser ce client", 
                      use_container_width=True,
                      type="primary"):

    df_client = preparer_client()
    proba     = model.predict_proba(df_client)[0][1] * 100

    # Colonnes résultats
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

    # Action recommandée
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

    st.divider()

    # SHAP — Explication
    # st.subheader("🔍 Pourquoi ce score ?")
    # explainer   = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(df_client)

    # fig, ax = plt.subplots(figsize=(10, 4))
    # shap.plots.waterfall(
        #explainer(df_client)[0],
        #max_display=10,
        #show=False
    #)
    #st.pyplot(fig)
    #plt.close()
