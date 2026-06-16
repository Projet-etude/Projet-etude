import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Import optionnel pour l'export des artefacts
try:
    from export_model_artifacts import export_training_artifacts
    HAS_EXPORT = True
except ImportError:
    HAS_EXPORT = False


@dataclass
class TextDataset(torch.utils.data.Dataset):
    encodings: dict
    labels: list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def detect_text_column(df):
    for col in ["text_clean", "text_original", "content", "text"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Aucune colonne de texte détectée. Ajoute `text_clean`, `text_original`, `content` ou `text` dans ton dataset."
    )


def detect_label_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    # try case-insensitive fallback
    lowered = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lowered:
            return lowered[col.lower()]
    return None


def read_input_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, delimiter=";")

    # supprimer les colonnes vides générées par Excel ou par des séparateurs supplémentaires
    df = df.loc[:, df.columns.astype(str).str.strip().astype(bool)]
    return df


def filter_labeled_rows(df, label_col):
    if label_col not in df.columns:
        return df
    series = df[label_col].astype(str).str.strip()
    valid = series.replace("nan", "", regex=False).astype(str).str.strip() != ""
    return df[valid].copy()


def prepare_data(df, text_col, label_col, tokenizer, max_length):
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    label_encoder = LabelEncoder()
    label_ids = label_encoder.fit_transform(labels)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return TextDataset(encodings, label_ids), label_encoder


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def train_task(
    df,
    task_name,
    label_col,
    model_name,
    output_dir,
    text_col,
    num_epochs,
    batch_size,
    max_length,
):
    print(f"\n=== Entraînement : {task_name} ===")

    df_task = df[[text_col, label_col]].dropna()
    if df_task.empty:
        raise ValueError(f"Aucune donnée valide pour la tâche `{task_name}`.")

    train_df, eval_df = train_test_split(
        df_task,
        test_size=0.2,
        random_state=42,
        stratify=df_task[label_col],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, train_encoder = prepare_data(train_df, text_col, label_col, tokenizer, max_length)
    eval_dataset, eval_encoder = prepare_data(eval_df, text_col, label_col, tokenizer, max_length)

    if list(train_encoder.classes_) != list(eval_encoder.classes_):
        raise ValueError("Les classes de labels d'entraînement et de validation ne correspondent pas.")

    num_labels = len(train_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=task_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(task_output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Entraînement du modèle...")
    trainer.train()

    print("Évaluation finale...")
    metrics = trainer.evaluate()
    print(metrics)

    y_true = eval_dataset.labels
    y_pred = trainer.predict(eval_dataset).predictions.argmax(axis=-1)
    report = classification_report(
        y_true,
        y_pred,
        target_names=train_encoder.classes_,
        zero_division=0,
    )

    print("\nClassification report :")
    print(report)

    label_map = {int(i): label for i, label in enumerate(train_encoder.classes_)}
    with open(os.path.join(task_output_dir, "label_map.txt"), "w", encoding="utf-8") as f:
        for idx, label in label_map.items():
            f.write(f"{idx}\t{label}\n")

    print(f"Modèle sauvegardé dans : {task_output_dir}")

    return metrics, report, train_encoder.classes_


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning de modèle NLP pour sentiment classification.")
    parser.add_argument(
        "--input",
        default="data/labeled_data_clean.csv",
        help="Chemin du fichier CSV labellisé et nettoyé",
    )
    parser.add_argument(
        "--model",
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        help="Modèle pré-entraîné Transformers à utiliser",
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Répertoire de sortie pour les modèles entraînés",
    )
    parser.add_argument(
        "--task",
        choices=["sentiment"],
        default="sentiment",
        help="Tâche à entraîner (sentiment uniquement pour ce script)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=8, help="Taille de batch")
    parser.add_argument("--max_length", type=int, default=128, help="Longueur maximale de tokenization")
    args = parser.parse_args()

    df = read_input_csv(args.input)
    text_col = detect_text_column(df)

    sentiment_col = detect_label_column(df, ["sentiment", "sentiment_label", "label_sentiment", "label sentiment"])

    if sentiment_col:
        df = filter_labeled_rows(df, sentiment_col)

    if df.empty:
        raise ValueError("Aucune ligne labellisée disponible pour l'entraînement.")

    if not sentiment_col:
        raise ValueError(
            "Impossible d'entraîner : aucune colonne de label de sentiment détectée."
        )
    
    train_task(
        df,
        task_name="sentiment",
        label_col=sentiment_col,
        model_name=args.model,
        output_dir=args.output_dir,
        text_col=text_col,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Exporter les artefacts pour le frontend
    if HAS_EXPORT:
        print("\nExport des artefacts pour l'intégration web...")
        task_output_dir = os.path.join(args.output_dir, "sentiment")
        export_training_artifacts(task_output_dir)
    else:
        print("\n⚠️  export_model_artifacts non disponible. Exécute manuellement :")
        print(f"python src/export_model_artifacts.py")


if __name__ == "__main__":
    main()
