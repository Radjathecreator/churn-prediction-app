# 📡 Churn Predictor App - Télécom

Une application web interactive développée avec [Streamlit](https://streamlit.io/) permettant de prédire le risque de désabonnement (*churn*) des clients d'une entreprise de télécommunications. 

Ce projet s'appuie sur un modèle de Machine Learning (XGBoost) entraîné pour identifier les clients à risque afin d'optimiser les campagnes de rétention.

## 🌟 Fonctionnalités

L'application est divisée en deux onglets principaux :
*   **👤 Profil individuel :** Saisie manuelle des informations d'un client via un formulaire interactif pour obtenir un score de risque en temps réel et des recommandations d'actions.
*   **📂 Analyse par lot (*Batch*) :** Importation d'un fichier CSV contenant une base de clients. L'application génère un tableau de bord global et permet d'exporter les prédictions (clients à risque élevé, modéré ou faible).

## 🛠️ Technologies utilisées

*   **Langage :** Python 3
*   **Interface Web :** Streamlit
*   **Data Science :** Pandas, NumPy
*   **Machine Learning :** Scikit-Learn, XGBoost
*   **Déploiement :** Streamlit Community Cloud (ou autre, à préciser)

## 📁 Structure du Projet

```text
churn-prediction-app/
│
├── app.py                 # Script principal de l'application Streamlit
├── churn_model.pkl        # Modèle de Machine Learning entraîné
├── scaler.pkl             # Scaler pour la normalisation des données
├── columns.pkl            # Noms des colonnes attendues par le modèle
├── mock_clients.csv       # Fichier CSV d'exemple pour tester l'application
├── Projet_Churn.ipynb     # Notebook Jupyter contenant l'analyse exploratoire et l'entraînement
└── requirements.txt       # Liste des dépendances Python
