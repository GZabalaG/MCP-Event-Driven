FROM python:3.11-slim-bullseye

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure resources copied
COPY resources/ .

# Each server defines its own CMD in docker-compose
