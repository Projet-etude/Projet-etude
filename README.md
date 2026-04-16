# 📘 Projet étude

# 🛰️ Projet Thumalien : Détection de Fake News sur Bluesky

## 🧭 Présentation du projet
Dans un contexte où la désinformation circule massivement sur les réseaux sociaux, le projet Thumalien propose une solution automatisée pour analyser les messages publiés sur Bluesky.

---

## 🎯 Objectifs du pipeline
1. Collecter des publications en temps réel via l'API Bluesky.
2. Préparer des données pour l'identification de contenus potentiellement trompeurs.
3. Préparer des données pour l'analyse émotionnelle des messages.
4. Mesurer l'empreinte carbone de l'infrastructure (Green IT).

---

## 📁 Structure du projet
- `src/collecte_bluesky.py` : script principal de collecte automatisée par mots-clés
- `src/init_db.py` : initialisation du schéma PostgreSQL
- `src/preprocess_data.py` : nettoyage et prétraitement des textes
- `src/export_csv.py` : export des données depuis PostgreSQL vers le CSV
- `Dockerfile` : image applicative Python
- `docker-compose.yml` : orchestration application + base PostgreSQL
- `.env.example` : modèle des variables d'environnement
- `data/` : fichiers de sortie (`export_posts.csv`, `dataset_final.csv`)

---

## ⚙️ Architecture technique
Le projet est entièrement conteneurisé avec Docker pour garantir portabilité, reproductibilité et simplicité de déploiement.

### 🧰 Stack technologique
- Langage : Python 3.11
- Base de données : PostgreSQL 15
- Collecte : API Bluesky (`requests`)
- Traitement : `pandas`, `scikit-learn`
- Green IT : CodeCarbon
- Infrastructure : Docker + Docker Compose

---

## 🚀 Installation locale (préparation)
1. Cloner le dépôt
2. Créer et activer un environnement virtuel

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Configurer les variables d'environnement

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Puis renseigner dans `.env` :
- `BLUESKY_HANDLE`
- `BLUESKY_PASSWORD`
- `POSTGRES_PASSWORD`

---

## 🧪 Exécution sans Docker
⚠️ L'exécution sans Docker n'est pas supportée nativement dans la version actuelle.

Pourquoi :
- Les scripts Python utilisent le service PostgreSQL nommé `db`
- Le script d'export écrit dans `/app/data/export_posts.csv` (chemin conteneur)

✅ Pour une exécution stable et conforme, utiliser Docker Compose.

---

## 🐳 Exécution avec Docker (recommandé)

### Démarrer les services
```bash
docker-compose up --build
```

### Arrêter les services
```bash
docker-compose down
```

### 📌 Notes importantes
- Le service `app` monte `./src` dans `/app/src`
- Le service `app` monte `./data` dans `/app/data`
- Le fichier généré dans le conteneur (`/app/data/export_posts.csv`) est visible côté machine dans `./data/export_posts.csv`
- Le service PostgreSQL est accessible via le nom réseau `db`
- Les variables `BLUESKY_HANDLE`, `BLUESKY_PASSWORD` et `POSTGRES_PASSWORD` sont lues depuis `.env`

---

## 🔐 Variables d'environnement
Utiliser `.env.example` comme modèle :

```env
BLUESKY_HANDLE=ton_handle.bsky.social
BLUESKY_PASSWORD=ton_mot_de_passe_app
POSTGRES_PASSWORD=un_mot_de_passe_fort_ici
```

---

## 👥 Équipe
- **Anass Al Fatni** : Data Scientist
- **Chaymae Mansouri** : Data Scientist
- **Fatima Amrouche** : Data Engineer
- **Madiha Lakhmiri** : Data Analyst
- **Miriam El Qadi** : Data Scientist
