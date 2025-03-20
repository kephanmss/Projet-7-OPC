FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier de dépendances uv.lock
COPY uv.lock* .
COPY pyproject.toml .

# Installer uv et synchroniser les dépendances
RUN pip install --no-cache-dir --upgrade uv && \
    uv sync --no-cache

# Copier seulement le code de l'application
COPY ./projet7 ./projet7

# Exposer le port dynamique utilisé par Heroku
EXPOSE $PORT

# Lancer l'application FastAPI avec uvicorn en utilisant $PORT
CMD uvicorn projet7.Missamou_Kephan_1_API_112024:app --host 0.0.0.0 --port $PORT