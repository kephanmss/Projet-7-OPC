import uvicorn
import mlflow.sklearn
import mlflow.pyfunc
from pydantic import BaseModel, create_model
from fastapi import FastAPI, HTTPException
from typing import Type
import pandas as pd
import os

# Chargement des noms des features
schema_donnees = pd.read_csv('feature_names.csv', sep=';', encoding='utf-8')

# Fonction pour créer un modèle Pydantic dynamique
def create_feature_model(df: pd.DataFrame) -> Type[BaseModel]:
    fields = {}
    for _, row in df.iterrows():
        feature_name = row['features']
        feature_type = row['type']
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
model_uri = "s3://projet-7-opc/models/my-model/"

try:
    # Chargement du modèle
    model = mlflow.pyfunc.load_model(model_uri)
    print("Modèle chargé avec succès")
except Exception as e:
    print("Erreur lors du chargement du modèle")
    print(e)

# Définition de l'application FastAPI
app = FastAPI()

@app.get("/")
def index():
    return {"message": "Scoring Crédit"}

@app.post("/predict/")
def predict_defaut_paiement(data: FeatureModel):
    try:
        model_uri = "s3://projet-7-opc/models/my-model/"
        model = mlflow.pyfunc.load_model(model_uri)
        data = data.dict()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return {"prediction": 'Défaut de paiement' if prediction[0] == 1 else 'Pas de défaut de paiement'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)