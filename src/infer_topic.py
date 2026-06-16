import argparse
from pathlib import Path
import inspect
import pandas as pd

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "Le package 'bertopic' ou 'sentence-transformers' n'est pas installé. "
        "Exécutez 'pip install -r requirements.txt' puis réessayez. "
        "Assurez-vous d'activer votre environnement virtuel si nécessaire."
    ) from exc

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOPIC_MODEL_DIR = Path("models/topic")
DEFAULT_TOPIC_MODEL_PATH = DEFAULT_TOPIC_MODEL_DIR / "bertopic_model"


def infer_topics(
    input_path,
    output_path,
    model_dir="models/topic",
    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    force_retrain=False,
):
    """Infère les topics sur un dataset avec BERTopic et sauvegarde le modèle."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "bertopic_model"

    print(f"Chargement du dataset : {input_path}")
    df = pd.read_csv(input_path)
    print(f"Total de lignes : {len(df)}")

    text_col = None
    for col in ["text", "text_clean", "text_original", "content"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            "Aucune colonne de texte trouvée "
            "(text, text_clean, text_original ou content)."
        )

    texts = df[text_col].fillna("").astype(str).tolist()
    if not texts:
        raise ValueError("Le fichier d'entrée ne contient aucun texte valide.")

    if model_path.exists() and not force_retrain:
        print(f"Chargement du modèle BERTopic existant : {model_path}")
        topic_model = BERTopic.load(str(model_path))
        topics, probabilities = topic_model.transform(texts, calculate_probabilities=True)
    else:
        print("Entraînement du modèle BERTopic...")
        embedding_model = SentenceTransformer(embedding_model_name)
        topic_model = BERTopic(embedding_model=embedding_model, verbose=False)
        fit_transform_sig = inspect.signature(topic_model.fit_transform)
        if "calculate_probabilities" in fit_transform_sig.parameters:
            topics, probabilities = topic_model.fit_transform(texts, calculate_probabilities=True)
        else:
            topics, probabilities = topic_model.fit_transform(texts)
        topic_model.save(str(model_path))
        print(f"Modèle sauvegardé : {model_path}")

    if probabilities is None:
        scores = [0.0 for _ in topics]
    else:
        scores = []
        for prob in probabilities:
            if prob is None:
                scores.append(0.0)
            elif hasattr(prob, "__len__"):
                scores.append(float(prob[0]) if len(prob) else 0.0)
            else:
                scores.append(float(prob))

    topic_labels = {
        topic_id: topic_label
        for topic_id, topic_label in topic_model.topic_labels_.items()
    }

    df["topic_id"] = topics
    df["topic"] = [
        topic_labels.get(topic_id, "Autre")
        if topic_id != -1
        else "Autre"
        for topic_id in topics
    ]
    df["topic_score"] = scores

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nRésultats sauvegardés dans : {output_path}")
    print("\nDistribution des topics :")
    print(df["topic"].value_counts())
    print(f"\nScore moyen de confiance : {df['topic_score'].mean():.3f}")

    print("\nTop topics du modèle :")
    print(topic_model.get_topic_info().head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inférence de topics avec BERTopic."
    )

    parser.add_argument(
        "--input",
        default="data/labeled_data_clean.csv",
        help="Chemin du fichier d'entrée",
    )
    parser.add_argument(
        "--output",
        default="data/labeled_data_with_topics.csv",
        help="Chemin du fichier de sortie",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_TOPIC_MODEL_DIR),
        help="Répertoire de sauvegarde du modèle BERTopic",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Modèle d'embedding pour BERTopic",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Forcer l'entraînement d'un nouveau modèle BERTopic",
    )

    args = parser.parse_args()

    infer_topics(
        args.input,
        args.output,
        model_dir=args.model_dir,
        embedding_model_name=args.embedding_model,
        force_retrain=args.force_retrain,
    )
