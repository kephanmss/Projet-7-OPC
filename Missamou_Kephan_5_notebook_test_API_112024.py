# Plan d'attaque API : On met quoi dedans ? Predict ? Réentrainement ? Monitoring ? SHAP ?
# Dans tous les cas, il faut des scripts pour et les images Docker pour les mettre en production

import uvicorn
import mlflow.sklearn
from pydantic import BaseModel, create_model
from fastapi import FastAPI, HTTPException
from typing import Type
import pandas as pd
import shap

# Chargement des noms des features
schema_donnees = pd.read_csv('feature_names.csv', sep=';', encoding = 'utf-8')
print(schema_donnees)
# Fonction pour créer un modèle Pydantic dynamique
def create_feature_model(df: pd.DataFrame) -> Type[BaseModel]:
    # Dictionnaire pour stocker les champs du modèle
    fields = {}
    # Parcourir le DataFrame pour ajouter des champs au modèle
    for _, row in df.iterrows():
        feature_name = row['features']
        feature_type = row['type']

        # Mapping des types de données
        if feature_type.startswith('int'):
            fields[feature_name] = (int, ...)
        elif feature_type.startswith('float'):
            fields[feature_name] = (float, ...)
        else:
            fields[feature_name] = (str, ...)
    return create_model('FeatureModel', **fields)

# Création du modèle Pydantic
FeatureModel = create_feature_model(schema_donnees)

# Chemin du modèle MLflow
model_name = "Model test deploiement"
model_version = 'None'
model_uri = f"models:/{model_name}/{model_version}"

# Chargement du modèle
model = mlflow.sklearn.load_model(model_uri)

# Définition de l'application FastAPI
app = FastAPI()

@app.get("/")
def index():
    return {"message": "Scoring Crédit"}

@app.post("/predict/")
def predict_defaut_paiement(data:FeatureModel):
    try:
        data = data.dict()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return {"prediction": 'Défaut de paiement' if prediction[0] == 1 else 'Pas de défaut de paiement'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)