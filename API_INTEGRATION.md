# 🔗 Intégration API ML pour l'équipe Web

Ce document explique comment intégrer le modèle ML fine-tuné de sentiment classification dans votre application web.

---

## 📁 Fichiers produits par la pipeline ML

Après entraînement du modèle, les fichiers suivants sont générés dans `models/sentiment/export/` :

```
models/
└── sentiment/
    ├── pytorch_model.bin          # Modèle fine-tuné
    ├── config.json                # Configuration du modèle
    ├── tokenizer_config.json
    ├── label_map.txt              # Mapping des labels
    ├── trainer_state.json         # Historique d'entraînement
    └── export/
        ├── config.json            # Config pour l'API (IMPORTANT)
        ├── metrics.json           # Métriques finales en JSON
        ├── training_loss.png      # Courbe de loss d'entraînement
        └── training_metrics.png   # Courbes d'évaluation (Accuracy, F1, etc.)
```

---

## 🚀 Démarrer l'API ML

### 1. **Installation des dépendances**
```bash
pip install -r requirements.txt
```

### 2. **Lancer l'API Flask**
```bash
python src/api_sentiment.py
```

L'API sera disponible à : **http://localhost:5000**

---

## 📡 Endpoints de l'API

### **1. POST /predict** — Prédire le sentiment
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "C'\''est un super produit!"}'
```

**Réponse :**
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

---

### **2. GET /metrics** — Récupérer les métriques d'entraînement
```bash
curl http://localhost:5000/metrics
```

**Réponse :**
```json
{
  "timestamp": "2026-06-16T14:30:00.000000",
  "model": "xlm-roberta-base",
  "task": "sentiment_classification",
  "training_epochs": 3,
  "final_metrics": {
    "eval_accuracy": 0.92,
    "eval_f1_macro": 0.90,
    "eval_f1_weighted": 0.91
  }
}
```

---

### **3. GET /plots/{plot_name}** — Récupérer les diagrammes
```bash
# Courbes de loss
curl http://localhost:5000/plots/training_loss.png > loss.png

# Métriques d'évaluation
curl http://localhost:5000/plots/training_metrics.png > metrics.png
```

Utilise ces URLs directement dans une balise `<img>` HTML :
```html
<img src="http://localhost:5000/plots/training_loss.png" alt="Loss Plot" />
<img src="http://localhost:5000/plots/training_metrics.png" alt="Metrics Plot" />
```

---

### **4. GET /config** — Configuration du modèle
```bash
curl http://localhost:5000/config
```

**Réponse :**
```json
{
  "model_path": "C:\\Users\\anass\\Projet-etude\\models\\sentiment",
  "model_type": "sentiment_classifier",
  "tokenizer": "xlm-roberta-base",
  "labels": ["Négatif", "Neutre", "Positif"],
  "artifacts": {
    "loss_plot": "training_loss.png",
    "metrics_plot": "training_metrics.png",
    "metrics_json": "metrics.json"
  }
}
```

---

### **5. GET /info** — Informations générales
```bash
curl http://localhost:5000/info
```

---

## 🎨 Intégration Frontend (exemple React)

### Prédiction de sentiment
```javascript
const predictSentiment = async (text) => {
  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  
  const result = await response.json();
  console.log("Sentiment:", result.sentiment);
  console.log("Confiance:", result.confidence);
};

predictSentiment("J'adore ce produit!");
```

### Affichage des métriques
```javascript
const loadMetrics = async () => {
  const response = await fetch("http://localhost:5000/metrics");
  const metrics = await response.json();
  
  console.log("Accuracy:", metrics.final_metrics.eval_accuracy);
  console.log("F1 Score:", metrics.final_metrics.eval_f1_macro);
};

loadMetrics();
```

### Affichage des diagrammes
```html
<div>
  <h3>Training Loss</h3>
  <img src="http://localhost:5000/plots/training_loss.png" />
  
  <h3>Evaluation Metrics</h3>
  <img src="http://localhost:5000/plots/training_metrics.png" />
</div>
```

---

## 🐳 Déployer avec Docker (optionnel)

Crée un `Dockerfile` pour l'API :
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "src/api_sentiment.py"]
```

Lance avec :
```bash
docker build -t sentiment-api .
docker run -p 5000:5000 sentiment-api
```

---

## 🔌 Intégration Multi-API

Si vous avez aussi l'API pour les **topics** (BERTopic), vous pouvez les appeler en parallèle :

```javascript
const analyzeSentimentAndTopic = async (text) => {
  const sentimentRes = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  
  const topicRes = await fetch("http://localhost:6000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  
  const sentiment = await sentimentRes.json();
  const topic = await topicRes.json();
  
  return { sentiment, topic };
};
```

Vous pouvez aussi récupérer la liste des topics détectés avec :

```bash
curl http://localhost:6000/topics
```

---

## 📊 Variables d'environnement

Si vous voulez configurer l'API sans coder, créez un `.env` :
```env
MODEL_DIR=models/sentiment
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=false
```

Modifiez `src/api_sentiment.py` pour lire depuis `.env` :
```python
from dotenv import load_dotenv

load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR", "models/sentiment")
```

---

## ⚠️ Notes importantes

1. **GPU vs CPU** : L'API détecte automatiquement le GPU. Sans GPU, l'inférence sera plus lente.
2. **CORS** : La CORS est activée pour les appels depuis le frontend.
3. **Port par défaut** : 5000. Changez-le dans `api_sentiment.py` si nécessaire.
4. **Authentification** : Aucune pour l'instant. Ajoutez JWT ou API keys en production.

---

## 🆘 Dépannage

### L'API ne démarre pas
- Vérifie que les dépendances sont installées : `pip install -r requirements.txt`
- Vérifie que le modèle existe dans `models/sentiment/`

### Erreur 404 sur /metrics
- Assure-toi d'avoir exécuté `export_model_artifacts.py` après l'entraînement
- Vérifie que `models/sentiment/export/metrics.json` existe

### Erreur CORS
- La CORS est déjà activée dans `api_sentiment.py` (voir `CORS(app)`)
- Si ça ne marche pas, vérifie l'URL d'origine du frontend

---

