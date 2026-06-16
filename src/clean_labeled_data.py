import argparse
import pandas as pd


def read_input_csv(path):
    """Détecte et charge le CSV avec le bon séparateur."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, delimiter=";")

    # Supprimer les colonnes vides
    df = df.loc[:, df.columns.astype(str).str.strip().astype(bool)]
    return df


def detect_text_column(df):
    """Détecte automatiquement la colonne de texte."""
    for col in ["text_clean", "text_original", "content", "text"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Aucune colonne de texte détectée. Ajoute `text_clean`, `text_original`, `content` ou `text`."
    )


def detect_label_column(df, candidates):
    """Détecte la colonne de label (case insensitive)."""
    for col in candidates:
        if col in df.columns:
            return col
    lowered = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lowered:
            return lowered[col.lower()]
    return None


def clean_labeled_data(input_path, output_path):
    """Charge les données, filtre les lignes labellisées et sauvegarde."""
    print(f"Chargement : {input_path}")
    df = read_input_csv(input_path)
    print(f"Total de lignes : {len(df)}")

    # Détecter les colonnes nécessaires
    text_col = detect_text_column(df)
    print(f"Colonne de texte détectée : {text_col}")

    sentiment_col = detect_label_column(
        df, ["sentiment", "sentiment_label", "label_sentiment", "label sentiment"]
    )
    
    if not sentiment_col:
        raise ValueError("Aucune colonne de sentiment détectée.")
    
    print(f"Colonne de sentiment détectée : {sentiment_col}")

    # Filtrer les lignes labellisées (sentiment non vide)
    df_clean = df.copy()
    df_clean[sentiment_col] = df_clean[sentiment_col].astype(str).str.strip()
    df_clean = df_clean[df_clean[sentiment_col] != ""]
    df_clean = df_clean[df_clean[sentiment_col] != "nan"]

    # Normalisation des labels
    label_mapping = {
        "positif": "Positif",
        "positif ": "Positif",
        "positive": "Positif",
        "positive ": "Positif",
        "négatif": "Négatif",
        "négative": "Négatif",
        "negative": "Négatif",
        "negative ": "Négatif",
        "neutre": "Neutre",
        "neutre ": "Neutre",
        "neutral": "Neutre",
        "neutral ": "Neutre",
    }

    df_clean[sentiment_col] = (
        df_clean[sentiment_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(label_mapping)
    )

    df_clean = df_clean[df_clean[sentiment_col].notna()]

    # Garder seulement les colonnes utiles
    columns_to_keep = [text_col, "language", sentiment_col]
    columns_to_keep = [c for c in columns_to_keep if c in df_clean.columns]
    df_clean = df_clean[columns_to_keep]

    # Renommer les colonnes pour cohérence
    df_clean.columns = ["text", "language", "sentiment"]

    print(f"Lignes après filtrage : {len(df_clean)}")

    # Sauvegarde
    df_clean.to_csv(output_path, index=False, sep=",")
    print(f"Données nettoyées sauvegardées dans : {output_path}")

    # Afficher les stats
    print(f"\nDistribution des sentiments :")
    print(df_clean["sentiment"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nettoyage du dataset labellisé.")
    parser.add_argument(
        "--input",
        default="data/labeled_data.csv",
        help="Chemin du fichier labellisé brut",
    )
    parser.add_argument(
        "--output",
        default="data/labeled_data_clean.csv",
        help="Chemin du fichier nettoyé",
    )
    args = parser.parse_args()

    clean_labeled_data(args.input, args.output)
