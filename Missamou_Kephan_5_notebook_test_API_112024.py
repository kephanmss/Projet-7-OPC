import requests
import json
import ast
import pandas as pd
# URL de base de l'API
base_url = "https://projet-7-opc-ba76cdb86807.herokuapp.com"

# Endpoint de prédiction
predict_endpoint = f"{base_url}/predict/"

# Exemple de données à envoyer (doit correspondre à votre FeatureModel)
with open('input_example_for_api.txt', 'r') as file:
    content = file.read()

donnees_a_predire = ast.literal_eval(content)

donnees_a_predire = json.loads(json.dumps(donnees_a_predire))

# Debug: Print what we're sending
print("Data being sent:", donnees_a_predire)

# Envoi de la requête POST avec les données au format JSON
try:
    response = requests.post(predict_endpoint, json=donnees_a_predire)
    
    # Debug: Print the full response regardless of status code
    print(f"Status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response content: {response.text}")
    
    response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP

    # Analyse de la réponse JSON
    prediction_result = response.json()
    print("Réponse de l'API:", prediction_result)

except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête: {e}")
except json.JSONDecodeError as e:
    print(f"Erreur lors de la décodage de la réponse JSON: {e}")