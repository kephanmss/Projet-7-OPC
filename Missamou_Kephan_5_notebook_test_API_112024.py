import requests
import json
import ast
# URL de base de l'API
base_url = "http://127.0.0.1:8000"  # Assurez-vous que votre API tourne sur cette adresse et port

# Endpoint de prédiction
predict_endpoint = f"{base_url}/predict/"

# Exemple de données à envoyer (doit correspondre à votre FeatureModel)
# Les noms des clés doivent correspondre aux noms des features définis dans 'feature_names.csv'
# et les types doivent correspondre à ceux définis ('int', 'float', 'str').
with open('input_example.txt', 'r') as file:
    content = file.read()

donnees_a_predire = ast.literal_eval(content)

# Envoi de la requête POST avec les données au format JSON
try:
    response = requests.post(predict_endpoint, json=donnees_a_predire)
    response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)

    # Analyse de la réponse JSON
    prediction_result = response.json()
    print("Réponse de l'API:", prediction_result)

except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête: {e}")
except json.JSONDecodeError as e:
    print(f"Erreur lors de la décodage de la réponse JSON: {e}")