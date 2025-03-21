FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers critiques
COPY requirements.txt .
COPY feature_names.csv .

# Install dependencies
RUN pip install --no-cache-dir uvicorn mlflow pydantic fastapi pandas boto3

# Copier seulement le code de l'application
COPY . .

# Exposer le port dynamique utilisé par Heroku
EXPOSE $PORT

# Lancer l'application FastAPI avec uvicorn en utilisant $PORT
CMD uvicorn Missamou_Kephan_1_API_112024:app --host 0.0.0.0 --port $PORT