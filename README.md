# 📡 Churn Predictor App - Télécom

Lien vers le projet : https://churn-prediction-app-lupxwgrsbdlsutvggpzfsz.streamlit.app/

Une application web interactive développée avec [Streamlit](https://streamlit.io/) permettant de prédire le risque de désabonnement (*churn*) des clients d'une entreprise de télécommunications. 

Ce projet s'appuie sur un modèle de Machine Learning (XGBoost) entraîné pour identifier les clients à risque afin d'optimiser les campagnes de rétention.

## 🌟 Fonctionnalités

L'application est divisée en deux onglets principaux :
*   **👤 Profil individuel :** Saisie manuelle des informations d'un client via un formulaire interactif pour obtenir un score de risque en temps réel et des recommandations d'actions.
*   **📂 Analyse par lot (*Batch*) :** Importation d'un fichier CSV contenant une base de clients. L'application génère un tableau de bord global et permet d'exporter les prédictions (clients à risque élevé, modéré ou faible).

<img width="1808" height="850" alt="image" src="https://github.com/user-attachments/assets/746db726-e489-4e93-821e-94011ee6ffd6" />
<img width="1781" height="660" alt="image" src="https://github.com/user-attachments/assets/073cb2ae-192d-403a-b432-e511cdfbbff4" />
<img width="1779" height="852" alt="image" src="https://github.com/user-attachments/assets/3fac5d94-9f43-4a12-b075-a38a3cc30520" />




## 🛠️ Technologies utilisées

*   **Langage :** Python 3
*   **Interface Web :** Streamlit
*   **Data Science :** Pandas, NumPy
*   **Machine Learning :** Scikit-Learn, XGBoost
*   **Déploiement :** Streamlit Community Cloud (ou autre, à préciser)

## 📊 Source des données
Le modèle a été entraîné sur le célèbre jeu de données Telco Customer Churn disponible sur Kaggle.
Lien : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Auteur
Radja Kurniawan - Data Analyst / Data Scientist - https://www.linkedin.com/in/radjakurniawan/

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


