"""
API Flask pour inférence de topics avec BERTopic.
Endpoints :
  - POST /predict : prédire le topic d'un texte
  - POST /batch : prédire les topics pour plusieurs textes
  - GET /topics : liste des topics détectés
  - GET /health : état de santé
"""

from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from bertopic import BERTopic
except ImportError as exc:
    raise ImportError(
        "Le package 'bertopic' n'est pas installé. "
        "Exécutez 'pip install -r requirements.txt' puis réessayez. "
        "Assurez-vous d'activer votre environnement virtuel si nécessaire."
    ) from exc

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path("models/topic")
MODEL_PATH = MODEL_DIR / "bertopic_model"

print("Chargement du modèle BERTopic...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modèle BERTopic introuvable : {MODEL_PATH}. "
        "Exécutez d'abord `python src/infer_topic.py` pour créer le modèle."
    )

topic_model = BERTopic.load(str(MODEL_PATH))
print("✅ Modèle BERTopic chargé.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "bertopic"}), 200


@app.route("/topics", methods=["GET"])
def topics():
    topic_info = topic_model.get_topic_info()
    simplified = [
        {
            "topic_id": int(row["Topic"]),
            "topic_name": row["Name"],
            "count": int(row["Count"]),
        }
        for _, row in topic_info.iterrows()
        if int(row["Topic"]) != -1
    ]
    return jsonify({"topics": simplified}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Aucune donnée JSON reçue"}), 400

        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Texte vide"}), 400

        # Limiter la taille du texte
        text_truncated = text[:4096]

        # Prédiction BERTopic
        result = topic_model.transform([text_truncated])

        # Certaines versions retournent (topics, probabilities)
        if isinstance(result, tuple):
            topics, probabilities = result
        else:
            topics = result
            probabilities = None

        topic_id = int(topics[0])

        # Récupérer le label du topic
        topic_name = (
            topic_model.topic_labels_.get(topic_id, "Autre")
            if topic_id != -1
            else "Autre"
        )

        # Calcul du score si disponible
        score = 0.0
        if probabilities is not None:
            try:
                prob = probabilities[0]

                if hasattr(prob, "__len__") and len(prob) > 0:
                    score = float(max(prob))
                else:
                    score = float(prob)

            except Exception:
                score = 0.0

        response = {
            "text": text,
            "topic_id": topic_id,
            "topic": topic_name,
            "score": round(score, 4),
        }

        return jsonify(response), 200

    except Exception as e:
        print("Erreur BERTopic :", str(e))
        return jsonify({"error": str(e)}), 500
    

@app.route("/batch", methods=["POST"])
def batch_predict():
    try:
        data = request.get_json()
        texts = data.get("texts", [])
        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "Provide a list of texts"}), 400

        texts_truncated = [str(text)[:4096] for text in texts]
        topics, probabilities = topic_model.transform(texts_truncated, calculate_probabilities=True)

        predictions = []
        for idx, text in enumerate(texts):
            topic_id = int(topics[idx])
            topic_name = topic_model.topic_labels_.get(topic_id, "Autre")
            score = float(max(probabilities[idx])) if probabilities and probabilities[idx] else 0.0
            predictions.append({
                "text": text,
                "topic_id": topic_id,
                "topic": topic_name,
                "score": score,
            })

        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "api_version": "1.0",
        "model_type": "bertopic",
        "model_name": str(MODEL_PATH),
        "endpoints": {
            "POST /predict": "Prédire le topic d'un texte",
            "POST /batch": "Prédire les topics pour plusieurs textes",
            "GET /topics": "Liste des topics détectés",
            "GET /health": "État de santé",
        }
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=False)
