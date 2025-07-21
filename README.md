# House Price Prediction - MLOps Project

Projet de prédiction de prix immobilier avec les bonnes pratiques MLOps.

## Dataset
[House Data from Kaggle](https://www.kaggle.com/datasets/shree1992/housedata?select=output.csv)

## Objectif
Construire un pipeline MLOps complet pour prédire le prix des maisons avec :
- Préparation et validation des données
- Entraînement et évaluation de modèles
- API REST pour les prédictions
- Containerisation avec Docker
- CI/CD avec GitHub Actions
- Logging et monitoring

## Installation

1. Cloner le repository
```bash
git clone <repo-url>
cd house-price-prediction
```

2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Structure du projet

```
house-price-prediction/
├── src/                 # Code source
├── data/               # Données brutes et traitées
├── models/             # Modèles sauvegardés
├── logs/               # Fichiers de logs
├── tests/              # Tests unitaires
├── notebooks/          # Exploration des données
└── Dockerfile          # Configuration Docker
```

## Usage

### 1. Exploration des données
```bash
jupyter notebook notebooks/exploration.ipynb
```

### 2. Entraînement du modèle
```bash
python -m src.models.model
```

### 3. Lancement de l'API
```bash
uvicorn src.api.main:app --reload
```

### 4. Tests
```bash
pytest tests/
```

## API Endpoints

- `GET /` : Health check
- `POST /predict` : Prédiction de prix

## Docker

```bash
# Build
docker build -t house-price-api .

# Run
docker run -p 8000:8000 house-price-api
```

## CI/CD

Le pipeline GitHub Actions comprend :
- Tests automatiques
- Build Docker
- Déploiement (à configurer)

## Monitoring

Les logs sont structurés avec :
- Timestamp de requête
- Features d'entrée
- Prédiction
- Durée de traitement