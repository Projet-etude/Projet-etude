"""
API Flask pour servir le modèle fine-tuné de sentiment classification.
Endpoints :
  - POST /predict : prédire le sentiment d'un texte
  - GET /metrics : récupérer les métriques d'entraînement
  - GET /plots : récupérer les images des diagrammes
  - GET /config : récupérer la configuration du modèle
"""

import json
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes cross-origin

# Configuration des chemins en fonction du dossier du script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "sentiment")
EXPORT_DIR = os.path.join(MODEL_DIR, "export")
CONFIG_FILE = os.path.join(EXPORT_DIR, "config.json")

# Charger la configuration
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Configuration non trouvée : {CONFIG_FILE}")

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

# Initialiser le modèle et le tokenizer
print("Chargement du modèle...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])

# Résoudre le chemin du modèle à charger : utiliser le dossier indiqué,
# sinon chercher le dernier checkpoint ou le dossier d'export
model_path_to_load = config.get("model_path", MODEL_DIR)
if not os.path.isabs(model_path_to_load):
    model_path_to_load = os.path.join(PROJECT_ROOT, model_path_to_load)

if not os.path.exists(os.path.join(model_path_to_load, "config.json")):
    # Chercher des sous-dossiers checkpoint-*
    candidates = []
    try:
        for name in os.listdir(model_path_to_load):
            path = os.path.join(model_path_to_load, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                candidates.append(path)
    except Exception:
        candidates = []

    if candidates:
        # Choisir le checkpoint avec le numéro le plus élevé si possible
        def _ck_num(p):
            import re
            m = re.search(r"checkpoint-(\d+)", p)
            return int(m.group(1)) if m else 0

        model_path_to_load = max(candidates, key=_ck_num)
    elif os.path.exists(os.path.join(EXPORT_DIR, "pytorch_model.bin")) or os.path.exists(os.path.join(EXPORT_DIR, "model.safetensors")):
        model_path_to_load = EXPORT_DIR

model = AutoModelForSequenceClassification.from_pretrained(
    model_path_to_load,
    num_labels=len(config["labels"]),
)
model.to(device)
model.eval()

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    top_k=None,
)

print(f"✅ Modèle chargé. Device : {device}")


@app.route("/health", methods=["GET"])
def health():
    """Endpoint de santé de l'API."""
    return jsonify({"status": "ok", "model": "sentiment_classifier"}), 200


def normalize_prediction_output(predictions):
    if isinstance(predictions, dict):
        return [predictions]

    # Aplatir les listes imbriquées jusqu'à la première liste de dictionnaires
    while isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
        predictions = predictions[0]

    if isinstance(predictions, list) and all(isinstance(p, dict) for p in predictions):
        return predictions

    raise ValueError(f"Format de sortie inattendu du pipeline : {type(predictions)} -> {repr(predictions)[:200]}")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Texte vide"}), 400

        text_truncated = text[:512]

        with torch.no_grad():
            predictions = classifier(text_truncated)

        print("RAW OUTPUT TYPE:", type(predictions))
        print("RAW OUTPUT:", predictions)

        predictions = normalize_prediction_output(predictions)
        best = predictions[0]

        result = {
            "text": text,
            "sentiment": best["label"],
            "confidence": float(best["score"]),
            "all_scores": [
                {
                    "label": p["label"],
                    "score": float(p["score"])
                }
                for p in predictions
            ]
        }

        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Récupérer les métriques d'entraînement en JSON."""
    try:
        metrics_file = os.path.join(EXPORT_DIR, "metrics.json")
        if not os.path.exists(metrics_file):
            return jsonify({"error": "Métriques non disponibles"}), 404
        
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
        
        return jsonify(metrics_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/plots/<plot_name>", methods=["GET"])
def get_plot(plot_name):
    """Récupérer un diagramme d'entraînement (PNG)."""
    try:
        allowed_plots = ["training_loss.png", "training_metrics.png"]
        if plot_name not in allowed_plots:
            return jsonify({"error": "Diagramme non trouvé"}), 404
        
        plot_path = os.path.join(EXPORT_DIR, plot_name)
        if not os.path.exists(plot_path):
            return jsonify({"error": f"{plot_name} non généré"}), 404
        
        return send_file(plot_path, mimetype="image/png"), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/config", methods=["GET"])
def get_config():
    """Récupérer la configuration du modèle."""
    try:
        return jsonify(config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/info", methods=["GET"])
def info():
    """Informations générales sur l'API."""
    return jsonify({
        "api_version": "1.0",
        "model": config.get("model_type"),
        "model_path": config.get("model_path"),
        "labels": config.get("labels"),
        "device": device,
        "endpoints": {
            "POST /predict": "Prédire le sentiment",
            "GET /metrics": "Obtenir les métriques",
            "GET /plots/training_loss.png": "Diagramme de loss",
            "GET /plots/training_metrics.png": "Diagramme de métriques",
            "GET /config": "Configuration du modèle",
            "GET /health": "État de santé de l'API",
        }
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
