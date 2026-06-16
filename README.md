# 📘 Projet Thumalien

# 🛰️ Analyse de Sentiment et Topic Modeling sur Bluesky

## 🧭 Présentation du projet

Thumalien est une solution complète d'analyse automatisée des messages publiés sur **Bluesky** pour analyser les sentiments et extraire des topics.

Le projet combine :
- **Collecte de données** : récupération des posts Bluesky en temps réel
- **Machine Learning** : fine-tuning de modèles pré-entraînés pour sentiment & topic classification
- **APIs REST** : exposition des modèles pour intégration web
- **Interface web** : application permettant l'analyse de texte en temps réel

---

## 🎯 Architecture globale

```
Phase 1 : Collecte
├─ collecte_bluesky.py → PostgreSQL
├─ export_csv.py → CSV
└─ preprocess_data.py → dataset_final.csv

Phase 2 : Machine Learning
├─ clean_labeled_data.py → labeled_data_clean.csv
├─ train_nlp.py → modèle fine-tuné + métriques
├─ infer_topic.py → topics BERTopic non supervisés
└─ export_model_artifacts.py → diagrammes & config

Phase 3 : API & Web
├─ api_sentiment.py (port 5000)
├─ api_topic.py (port 6000)
└─ Frontend (React/Vue) → utilise les APIs
```

---

## 📁 Structure du projet

```
Projet-etude/
├── src/
│   ├── collecte_bluesky.py          # Collecte Bluesky → PostgreSQL
│   ├── export_csv.py                # PostgreSQL → CSV
│   ├── preprocess_data.py           # Nettoyage & préprocessing
│   ├── init_db.py                   # Initialisation PostgreSQL
│   ├── clean_labeled_data.py        # Nettoyage données labellisées
│   ├── infer_topic.py               # Inférence topics (BERTopic)
│   ├── train_nlp.py                 # Fine-tuning sentiment (🔴 Anass)
│   ├── export_model_artifacts.py    # Export métriques & diagrammes
│   ├── api_sentiment.py             # API REST sentiment (port 5000)
│   └── api_topic.py                 # API REST topics (port 6000)
├── data/
│   ├── dataset_final.csv            # Données nettoyées
│   ├── export_posts.csv             # Export brut PostgreSQL
│   ├── labeled_data.csv             # Données labellisées manuellement
│   ├── labeled_data_clean.csv       # Données labellisées nettoyées
│   └── labeled_data_with_topics.csv # Avec topics inférés
├── models/
│   ├── sentiment/
│   │   ├── checkpoint-1236/         # Checkpoints (poids + états)
│   │   │   ├── model.safetensors    # Poids du modèle
│   │   │   ├── config.json
│   │   │   ├── tokenizer.json
│   │   │   └── trainer_state.json
│   │   ├── checkpoint-824/          # Ancien checkpoint
│   │   │   ├── model.safetensors
│   │   │   └── trainer_state.json
│   │   ├── label_map.txt
│   │   └── export/                  # 🌐 Pour le web / export d'artifacts
│   │       ├── config.json
│   │       ├── metrics.json         # Métriques JSON
│   │       ├── training_loss.png    # Diagramme
│   │       └── training_metrics.png # Diagramme
│   └── topic/
│       └── bertopic_model/          # Modèle BERTopic pour infer_topic
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── API_INTEGRATION.md                # 📖 Guide APIs pour équipe web
├── WORKFLOW_EQUIPE.md                # 📖 Workflow complet
└── README.md                         # Ce fichier
```

---

## ⚙️ Stack technologique

### Backend (Collecte & Stockage)
- **Python 3.11** : langage principal
- **PostgreSQL 15** : base de données
- **Docker & Docker Compose** : conteneurisation

### Machine Learning & NLP
- **Transformers** (Hugging Face) : modèles pré-entraînés
  - `xlm-roberta-base` : fine-tuning sentiment
  - `BERTopic` avec `sentence-transformers` : topic modeling non supervisé
- **PyTorch** : framework deep learning
- **scikit-learn** : métriques & encodage labels

### API & Web
- **Flask** : framework REST API
- **Flask-CORS** : cross-origin requests
- **Matplotlib** : génération diagrammes d'entraînement

### Utilitaires
- **Pandas** : manipulation données
- **CodeCarbon** : mesure empreinte carbone
- **Requests** : appels API Bluesky

---

## 🚀 Guide complet du projet

### 📋 Prérequis

- Python 3.8+ ou Docker
- GPU NVIDIA (optionnel, GPU recommandé pour ML)
- 4GB RAM minimum
- Compte Bluesky (pour la collecte, optionnel)

---

## 📖 Phase 1 : Installation & Configuration

### Étape 1.1 : Cloner & configurer l'environnement

```bash
# Windows
git clone <repo>
cd Projet-etude
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Étape 1.2 : Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 1.3 : Configurer les variables d'environnement (optionnel)

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Renseigner dans `.env` :
```env
BLUESKY_HANDLE=ton_handle.bsky.social
BLUESKY_PASSWORD=ton_mot_de_passe_app
POSTGRES_PASSWORD=mot_de_passe_fort
```

---

## 🔄 Phase 2 : Traitement des données (ML)

### Étape 2.1 : Nettoyer les données labellisées

Si tu as déjà labellisé des données manuellement dans `data/labeled_data.csv` :

```bash
python src/clean_labeled_data.py
# Output : data/labeled_data_clean.csv
```

Cela :
- Détecte automatiquement le séparateur CSV (`,` ou `;`)
- Filtre les lignes labellisées (supprime les vides)
- Standardise les colonnes

### Étape 2.2 : Inférer les topics automatiquement (optionnel)

Si tu veux ajouter les catégories de topics automatiquement :

```bash
python src/infer_topic.py
# Input : data/labeled_data_clean.csv
# Output : data/labeled_data_with_topics.csv
```

Le modèle BERTopic va extraire des topics automatiquement depuis le texte, sans liste de catégories fixe. Vous pouvez lister les topics détectés via l'endpoint `/topics` de l'API.

---

## 🤖 Phase 3 : Fine-tuning du modèle

### Entraîner le modèle de sentiment

```bash
python src/train_nlp.py \
  --input data/labeled_data_clean.csv \
  --model xlm-roberta-base \
  --epochs 3 \
  --batch_size 8 \
  --max_length 128
```

**Paramètres** :
- `--input` : fichier CSV labellisé
- `--model` : modèle Hugging Face (par défaut : `xlm-roberta-base`)
- `--epochs` : nombre d'epochs (3-5 recommandé)
- `--batch_size` : taille du batch (8-16 sur GPU)
- `--max_length` : longueur max tokens (128-256)

**Output généré** :
```
models/sentiment/
├── checkpoint-1236/
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── trainer_state.json
├── checkpoint-824/
│   ├── model.safetensors
│   └── trainer_state.json
├── label_map.txt
└── export/
  ├── config.json
  ├── metrics.json
  ├── training_loss.png
  └── training_metrics.png
```

Note : le checkpoint `checkpoint-1236` est le plus récent — pour partager le modèle, zippe ce dossier (`models/sentiment/checkpoint-1236`) et fournis le lien au reste de l'équipe.

## 🔒 Modèles externes (non poussés dans le dépôt)

Les fichiers de modèles (checkpoints, poids) sont volumineux et ne sont pas inclus dans le repo. Fournis des liens (Google Drive / OneDrive / Dropbox) vers les archives `.zip` et demande aux membres de :

- télécharger l'archive fournie
- la décompresser dans le répertoire du projet en respectant l'arborescence ci-dessous

Arborescence attendue après extraction :

```
models/
├── sentiment/
│   └── checkpoint-1236/   # contient model.safetensors, config.json, tokenizer.json, trainer_state.json...
│   └── label_map.txt
│   └── export/
└── topic/
  └── bertopic_model/    # dossier sauvegardé par BERTopic
```

Exemples de commandes pour extraire les archives :

Windows (PowerShell) :
```powershell
Expand-Archive -Path sentiment_checkpoint-1236.zip -DestinationPath .\models\sentiment
Expand-Archive -Path topic_bertopic.zip -DestinationPath .\models\topic
```

Linux / macOS :
```bash
unzip sentiment_checkpoint-1236.zip -d models/sentiment
unzip topic_bertopic.zip -d models/topic
```

Après extraction, le code existant peut charger les modèles ainsi :

Sentiment (Transformers) :
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "models/sentiment/checkpoint-1236"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
```

Remarque : si les poids sont en `model.safetensors`, `transformers` gère ce format si `safetensors` est installé.

Topic (BERTopic) :
```python
from bertopic import BERTopic

topic_model = BERTopic.load("models/topic/bertopic_model")
```

Conseils pratiques :
- Demande aux destinataires de créer le dossier `models/sentiment` et `models/topic` à la racine du projet avant d'extraire.
- Fournis un lien vers `label_map.txt` si nécessaire (souvent dans l'archive `sentiment`).

### Liens de téléchargement des modèles

- **Topic BERTopic** : https://drive.google.com/file/d/1d2rSvYSXCa5RLkZmzoM2CiFPNTIdtuov/view?usp=drive_link
- **Sentiment checkpoint** : https://drive.google.com/file/d/1-RQTZuWMPgrhvrznGesma7TCwHCtclYJ/view?usp=drive_link

> Après téléchargement, extraire le contenu dans les dossiers `models/sentiment` et `models/topic`.

**Exemple d'utilisation avec GPU** :
```bash
python src/train_nlp.py --epochs 5 --batch_size 16
```

---

## 🌐 Phase 4 : Lancer les APIs

Après le fine-tuning, démarre les APIs REST pour servir les modèles.

### API 1 : Sentiment Classification (port 5000)

```bash
python src/api_sentiment.py
```

Teste avec :
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "C'\''est un super produit!"}'
```

**Réponse** :
```json
{
  "text": "C'est un super produit!",
  "sentiment": "Positif",
  "confidence": 0.95,
  "all_scores": [
    {"label": "Négatif", "score": 0.05},
    {"label": "Neutre", "score": 0.0},
    {"label": "Positif", "score": 0.95}
  ]
}
```

### API 2 : Topic Modeling BERTopic (port 6000, optionnel)

```bash
python src/api_topic.py
```

Teste avec :
```bash
curl -X POST http://localhost:6000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "L'\''IA révolutionne l'\''industrie"}'
```

---

## 🔗 Endpoints des APIs

### Sentiment API (port 5000)

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/predict` | POST | Prédire sentiment d'un texte |
| `/metrics` | GET | Récupérer Accuracy, F1-score |
| `/plots/training_loss.png` | GET | Courbe de loss d'entraînement |
| `/plots/training_metrics.png` | GET | Diagrammes évaluation |
| `/config` | GET | Configuration du modèle |
| `/health` | GET | État de santé API |

### Topic API (port 6000)

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/predict` | POST | Prédire topic d'un texte |
| `/batch` | POST | Prédire topics pour plusieurs textes |
| `/topics` | GET | Liste des topics détectés |
| `/health` | GET | État de santé API |

---

## 💻 Phase 5 : Intégration Frontend

L'équipe web peut intégrer les APIs dans l'interface web.

**Exemple React** :
```javascript
const [text, setText] = useState("");

const handleAnalyze = async () => {
  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  const result = await res.json();
  console.log("Sentiment:", result.sentiment);
};

return (
  <div>
    <input value={text} onChange={(e) => setText(e.target.value)} />
    <button onClick={handleAnalyze}>Analyser</button>
  </div>
);
```

**Afficher les diagrammes** :
```javascript
<img src="http://localhost:5000/plots/training_loss.png" alt="Loss" />
<img src="http://localhost:5000/plots/training_metrics.png" alt="Metrics" />
```

---

## 📚 Documentation complète

- **`API_INTEGRATION.md`** : guide détaillé des endpoints pour l'équipe web
- **`WORKFLOW_EQUIPE.md`** : workflow complet collecte → ML → web

---

## 🐳 Exécution avec Docker (Collection uniquement)

Si tu veux juste utiliser la collecte Bluesky → PostgreSQL avec Docker :

```bash
docker-compose up --build
```

Notes :
- Le service `app` monte `./src` dans `/app/src`
- Le service `app` monte `./data` dans `/app/data`
- PostgreSQL accessible via `db`
- Variables lues depuis `.env`

Pour arrêter :
```bash
docker-compose down
```

---

## 🆘 Dépannage

### L'API retourne 404 sur `/metrics`
→ Exécute : `python src/export_model_artifacts.py`

### Erreur CUDA / GPU
→ Modifie dans `api_sentiment.py` : remplace `device=0` par `device=-1` (CPU)

### Port déjà utilisé
→ Modifie le port dans le script API : `app.run(port=5001)`

### Modèle ne charge pas
→ Vérifie que `models/sentiment/checkpoint-1236/model.safetensors` (ou un checkpoint existant) existe
→ Réentraîne si nécessaire : `python src/train_nlp.py`

---

## 👥 Équipe & Responsabilités

| Rôle | Personne | Responsabilité |
|------|----------|-----------------|
| **Data Scientist** | Anass Al Fatni | Fine-tuning ML, APIs, métriques |
| **Data Scientist** | Chaymae Mansouri | Étiquetage données, validation |
| **Data Engineer** | Fatima Amrouche | Infrastructure, pipeline ETL |
| **Data Analyst** | Madiha Lakhmiri | Analyse résultats, rapports |
| **Data Scientist** | Miriam El Qadi | Feature engineering, modèles |

---

## 🔄 Communication inter-équipe

### Données à partager avec l'équipe Web

1. **API Endpoints** → `API_INTEGRATION.md`
2. **Workflow complet** → `WORKFLOW_EQUIPE.md`
3. **URLs des APIs** :
   - Sentiment : `http://localhost:5000`
   - Topic : `http://localhost:6000`
4. **Fichiers d'export** :
   - `models/sentiment/export/metrics.json`
   - `models/sentiment/export/training_loss.png`
   - `models/sentiment/export/training_metrics.png`

---

## 📊 Workflow simplifié

```
┌─────────────────────────────────────────────────────────┐
│ 1. COLLECTE (optionnel)                                 │
│    collecte_bluesky.py → PostgreSQL                     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 2. PRÉPARATION                                          │
│    export_csv.py → clean_labeled_data.py               │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 3. ML (🔴 Anass)                                        │
│    train_nlp.py → models/ + metrics + diagrammes        │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 4. API REST                                             │
│    api_sentiment.py (port 5000)                         │
│    api_topic.py (port 6000)                             │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ 5. FRONTEND (Équipe Web)                               │
│    React/Vue → utilise /predict, /metrics, /plots       │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Checklist avant livraison

- [ ] Modèle fine-tuné entraîné et évalué
- [ ] Métriques exportées (`metrics.json`)
- [ ] Diagrammes générés (`*.png`)
- [ ] API sentiment testée avec `curl` ou Postman
- [ ] API topic testée (optionnel)
- [ ] Documentation (README, API_INTEGRATION, WORKFLOW)
- [ ] Git push avec tous les fichiers
- [ ] Équipe web a reçu les URLs des APIs

---

## 🚀 Quick Start (tl;dr)

```bash
# 1. Setup
pip install -r requirements.txt

# 2. ML Pipeline
python src/clean_labeled_data.py
python src/train_nlp.py --epochs 3

# 3. APIs
python src/api_sentiment.py  # Terminal 1
python src/api_topic.py      # Terminal 2

# 4. Web team uses :
# POST http://localhost:5000/predict
# GET http://localhost:5000/metrics
# GET http://localhost:5000/plots/*
```

---

## 📞 Support

- **Questions ML** → Anass Al Fatni
- **Questions données** → Équipe Data
- **Questions intégration web** → Voir `API_INTEGRATION.md`
- **Issues techniques** → Ouvrir une issue GitHub
