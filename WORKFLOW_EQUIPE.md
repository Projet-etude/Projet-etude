# 📋 Workflow Complet ML → Web (Guide d'équipe)

Ce document décrit le workflow complet pour passer du fine-tuning du modèle à son intégration dans l'application web.



## ⚙️ Phase 1 : Fine-tuning & Préparation (ML)

### Étape 1.1 : Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 1.2 : Nettoyer les données labellisées
```bash
python src/clean_labeled_data.py
# Output : data/labeled_data_clean.csv
```

### Étape 1.3 : Inférer les topics (optionnel)
```bash
python src/infer_topic.py
# Output : data/labeled_data_with_topics.csv
```

> Note : `src/infer_topic.py` utilise BERTopic pour détecter les topics et crée un modèle dans `models/topic/bertopic_model`.

> Note : `src/infer_topic.py` utilise BERTopic et crée un modèle dans `models/topic/bertopic_model`.

### Étape 1.4 : Fine-tuner le modèle de sentiment
```bash
python src/train_nlp.py --epochs 3 --batch_size 8
# Output : 
#   - models/sentiment/ (modèle fine-tuné)
#   - models/sentiment/export/ (artefacts pour le web)
```

✅ **Résultat attendu** :
```
models/sentiment/export/
├── config.json             # Configuration pour l'API
├── metrics.json            # Métriques en JSON
├── training_loss.png       # Diagramme de loss
└── training_metrics.png    # Diagramme de performances
```

---

## 🚀 Phase 2 : Lancer les APIs (ML)

Ouvre **deux terminaux** :

### Terminal 1 : API Sentiment
```bash
python src/api_sentiment.py
# Écoute sur : http://localhost:5000
```

### Terminal 2 : API Topic (BERTopic, optionnel)
```bash
python src/api_topic.py
# Écoute sur : http://localhost:6000
```

Utilise l'endpoint `/topics` pour récupérer la liste des topics détectés.

✅ **Vérification** :
```bash
curl http://localhost:5000/health
# {"status": "ok", "model": "sentiment_classifier"}

curl http://localhost:6000/health
# {"status": "ok", "model": "topic_classifier"}
```

---

## 💻 Phase 3 : Intégration Frontend (Équipe Web)

### 3.1 Prédiction de sentiment
```javascript
// Exemple React
const [text, setText] = useState("");
const [result, setResult] = useState(null);

const handlePredict = async () => {
  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  setResult(data);
};

return (
  <div>
    <textarea value={text} onChange={(e) => setText(e.target.value)} />
    <button onClick={handlePredict}>Analyser</button>
    {result && (
      <div>
        <p>Sentiment: {result.sentiment}</p>
        <p>Confiance: {(result.confidence * 100).toFixed(2)}%</p>
      </div>
    )}
  </div>
);
```

### 3.2 Affichage des métriques de performance
```javascript
const [metrics, setMetrics] = useState(null);

useEffect(() => {
  fetch("http://localhost:5000/metrics")
    .then(res => res.json())
    .then(data => setMetrics(data));
}, []);

return (
  <div className="metrics">
    {metrics && (
      <>
        <h3>Performance du modèle</h3>
        <p>Accuracy: {(metrics.final_metrics.eval_accuracy * 100).toFixed(2)}%</p>
        <p>F1 Score: {(metrics.final_metrics.eval_f1_macro * 100).toFixed(2)}%</p>
      </>
    )}
  </div>
);
```

### 3.3 Affichage des diagrammes d'entraînement
```javascript
return (
  <div className="charts">
    <div className="chart">
      <h4>Training Loss</h4>
      <img src="http://localhost:5000/plots/training_loss.png" alt="Loss" />
    </div>
    <div className="chart">
      <h4>Evaluation Metrics</h4>
      <img src="http://localhost:5000/plots/training_metrics.png" alt="Metrics" />
    </div>
  </div>
);
```

### 3.4 Prediction de topic (si utilisé)
```javascript
const predictTopic = async (text) => {
  const res = await fetch("http://localhost:6000/predict", {
    method: "POST",
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  return data.top_topic; // "Intelligence Artificielle", "Politique", etc.
};
```

---

## 🔄 Communication Équipe

### Fichiers à partager avec l'équipe web

| Fichier | Usage |
|---------|-------|
| `API_INTEGRATION.md` | Documentation complète des endpoints |
| `models/sentiment/export/config.json` | Configuration du modèle |
| `models/sentiment/export/metrics.json` | Métriques finales |
| `models/sentiment/export/training_loss.png` | Diagramme à afficher |
| `models/sentiment/export/training_metrics.png` | Diagramme à afficher |

### Checklist pour l'équipe web

- [ ] ✅ L'API sentiment tourne sur `localhost:5000`
- [ ] ✅ L'API topic tourne sur `localhost:6000` (optionnel)
- [ ] ✅ Les endpoints répondent (test `/health`)
- [ ] ✅ La prediction fonctionne : `POST /predict`
- [ ] ✅ Les métriques s'affichent : `GET /metrics`
- [ ] ✅ Les diagrammes se chargent : `GET /plots/*`
- [ ] ✅ L'interface web affiche sentiment + topic + diagrammes

---

## 🐳 Déploiement en Production

### Docker Compose (optionnel)

Crée `docker-compose.yml` :
```yaml
version: "3.8"
services:
  sentiment-api:
    build:
      context: .
      dockerfile: Dockerfile.sentiment
    ports:
      - "5000:5000"
    volumes:
      - ./models/sentiment:/app/models/sentiment
    environment:
      - MODEL_DIR=/app/models/sentiment

  topic-api:
    build:
      context: .
      dockerfile: Dockerfile.topic
    ports:
      - "6000:6000"

  web:
    build:
      context: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - sentiment-api
      - topic-api
    environment:
      - REACT_APP_SENTIMENT_API=http://sentiment-api:5000
      - REACT_APP_TOPIC_API=http://topic-api:6000
```

Lance avec :
```bash
docker-compose up --build
```

---

## 🆘 Dépannage

### L'API retourne 404 sur `/metrics`
→ Exécute : `python src/export_model_artifacts.py`

### CORS error dans le navigateur
→ C'est normal en développement. Ajoute une extension CORS ou utilise un proxy.

### GPU pas utilisé
→ Vérifie : `torch.cuda.is_available()` dans Python
→ Installe CUDA si nécessaire

### Port déjà utilisé
→ Modifie le port dans `api_sentiment.py` : `app.run(port=5001)`

---

## 📊 Exemple complet (tout-en-un)

```javascript
// Frontend Component - Sentiment & Topic Analysis
import React, { useState, useEffect } from 'react';

export default function AnalyzePanel() {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState(null);
  const [topic, setTopic] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    // Charger les métriques au montage
    fetch("http://localhost:5000/metrics")
      .then(r => r.json())
      .then(setMetrics);
  }, []);

  const handleAnalyze = async () => {
    // Prédiction sentiment
    const sentRes = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const sentData = await sentRes.json();
    setSentiment(sentData);

    // Prédiction topic
    const topRes = await fetch("http://localhost:6000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const topData = await topRes.json();
    setTopic(topData);
  };

  return (
    <div className="analysis-panel">
      <h1>Analyse Sentiment & Topic</h1>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Saisis un texte..."
      />
      <button onClick={handleAnalyze}>Analyser</button>

      {sentiment && (
        <div className="result">
          <h3>Sentiment: {sentiment.sentiment}</h3>
          <p>Confiance: {(sentiment.confidence * 100).toFixed(2)}%</p>
        </div>
      )}

      {topic && (
        <div className="result">
          <h3>Topic: {topic.top_topic}</h3>
          <p>Score: {(topic.top_score * 100).toFixed(2)}%</p>
        </div>
      )}

      <div className="charts">
        <img src="http://localhost:5000/plots/training_loss.png" alt="Loss" />
        <img src="http://localhost:5000/plots/training_metrics.png" alt="Metrics" />
      </div>

      {metrics && (
        <div className="metrics">
          <h3>Performance</h3>
          <p>Accuracy: {(metrics.final_metrics.eval_accuracy * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}
```

---

Pour des questions : voir `API_INTEGRATION.md`
