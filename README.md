# API Credit Scoring & Machine Learning Pipeline

Ce projet implémente une solution complète de **Credit Scoring** (évaluation du risque de crédit). Il comprend une phase d'exploration et d'entraînement d'un modèle de Machine Learning, ainsi que le déploiement de ce modèle sous forme d'**API REST** en utilisant le framework **FastAPI**.

L'objectif de l'application est de prédire si un crédit doit être accordé ou refusé à un client en fonction de sa situation financière et de ses antécédents.

## 🚀 Fonctionnalités
*   **Pipeline de Machine Learning** : Préparation des données, standardisation et classification.
*   **API FastAPI** : Endpoint de prédiction en temps réel exploitant le modèle entraîné.
*   **Validation des données** : Utilisation de `Pydantic` pour s'assurer de la conformité des données envoyées à l'API.
*   **Probabilités associées** : L'API renvoie la décision (Accord/Refus) ainsi que les probabilités de confiance associées.

---

## 📁 Structure du Projet

*   `TP_MOSSIERE_VERLANDE_VILLAUME.ipynb` : Le notebook contenant l'analyse exploratoire, le prétraitement des données et l'entraînement de la pipeline de classification.
*   `api.py` : Le code source de l'API FastAPI.
*   `credit_scoring.csv` : Le jeu de données utilisé pour l'entraînement.
*   `credit_scoring.pkl` : Le modèle entraîné exporté au format Pickle.
*   `requirements.txt` : Liste des dépendances Python nécessaires au projet.

---

## 🛠️ Installation


### 1. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```

### 2. Les dépendances
bibliothèques indispensables (`fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `numpy`)

---

## 📈 Entraînement du Modèle

Le processus de recherche, d'évaluation et de création de la pipeline finale est entièrement détaillé dans le fichier de référence principale du projet :  
📄 **`TP_MOSSIERE_VERLANDE_VILLAUME`**

Une fois le script ou le notebook exécuté, le fichier sérialisé `credit_scoring.pkl` est généré et utilisé par l'API pour effectuer les prédictions.

---

## 🖥️ Utilisation de l'API

### 1. Démarrer le serveur local
Pour lancer l'API FastAPI, exécutez la commande suivante :
```bash
uvicorn api:app --reload
```
L'API sera accessible à l'adresse suivante : `http://127.0.0.1:8000`.  
Vous pouvez consulter la documentation interactive (Swagger UI) sur `http://127.0.0.1:8000/docs`.

### 2. Exemple de requête de prédiction (POST)
Endpoint : `http://127.0.0.1:8000/predict/`

**Format JSON attendu (Body) :**
```json
{
  "Income": 150.0,
  "Seniority": 10.0,
  "Price": 1400.0,
  "Amount": 1000.0,
  "Age": 32.0,
  "Assets": 4000.0,
  "Expenses": 75.0,
  "Records": 1.0,
  "Time": 36.0,
  "Job": 1.0,
  "Debt": 0.0,
  "Home": 1.0,
  "Marital": 1.0
}
```

**Exemple de réponse retournée par l'API :**
```json
{
  "prediction": 1,
  "message": "Crédit Accordé",
  "probabilite_refus": 0.154,
  "probabilite_accord": 0.846
}
```

---

## 👥 Auteurs
Projet réalisé par l'équipe d'étudiants dans le cadre du fichier académique 
**MOSSIERE Erwan
VERLANDE Augustin 
VILLAUME Evan**.
