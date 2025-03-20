FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers critiques
COPY uv.lock* .
COPY pyproject.toml .
COPY feature_names.csv .

# Installer uv, synchroniser les dépendances et ajouter uvicorn
RUN pip install --no-cache-dir --upgrade uv && \
    uv sync --no-cache && \
    uv add uvicorn --no-cache

# Copier seulement le code de l'application
COPY . .

# Exposer le port dynamique utilisé par Heroku
EXPOSE $PORT

# Lancer l'application FastAPI avec uvicorn en utilisant $PORT
CMD uvicorn projet7.Missamou_Kephan_1_API_112024:app --host 0.0.0.0 --port $PORT