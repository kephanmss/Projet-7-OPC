# Le strict nécessaire
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

COPY requirements.txt .
COPY feature_names.csv .

# Le strict nécessaire
RUN pip install --no-cache-dir uvicorn mlflow pydantic fastapi pandas boto3 imblearn

COPY . .

# Exposition du port dynamique d'Heroku
EXPOSE $PORT

CMD uvicorn Missamou_Kephan_1_API_112024:app --host 0.0.0.0 --port $PORT
