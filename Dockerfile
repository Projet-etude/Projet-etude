# Utilisation d'une image Python légère
FROM python:3.11-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y gcc python3-dev

# Copie et installation des bibliothèques Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code source
COPY ./src ./src

# Commande par défaut 
CMD ["python", "src/collecte_bluesky.py"]