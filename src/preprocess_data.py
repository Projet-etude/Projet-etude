import pandas as pd
import re
import os

def clean_text(text):
    if not isinstance(text, str): return ""
    
    # 1. Passage en minuscules
    text = text.lower()
    
    # 2. Supprimer les URLs et les mentions (@user)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    
    # 3. Garder uniquement les lettres latines et accents FR
    # suppression automatique du Chinois, Japonais, Arabes, etc.
    text = re.sub(r'[^a-zàâçéèêëîïôûùµ\s]', ' ', text)
    
    # 4. Nettoyage des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fix_and_clean():
    input_path = 'data/export_posts.csv'
    output_path = 'data/dataset_final.csv'

    if not os.path.exists(input_path):
        print(f"ERREUR : {input_path} introuvable.")
        return

    df = pd.read_csv(input_path)
    df = df.iloc[:, [0, 1]] 
    df.columns = ['text_original', 'language']

    # Filtrage sur la détection initiale de l'API
    df = df[df['language'].isin(['fr', 'en'])]

    print("Nettoyage intensif en cours...")
    df['text_clean'] = df['text_original'].apply(clean_text)

    # --- FILTRES DE SÉCURITÉ SUPPLÉMENTAIRES ---
    
    # A. Suppression des lignes où 'text_clean' est vide 
    # ( suuprimer les posts qui n'avaient que des hashtags, des emojis ou du chinois)
    df = df[df['text_clean'] != ""]

    # B. Supprimer les phrases de moins de 3 mots 
    df = df[df['text_clean'].str.split().str.len() > 2]

    # C. Supprimer les doublons
    df = df.drop_duplicates(subset=['text_clean'])

    # Sauvegarde
    df.to_csv(output_path, index=False)
    
    print(f"--- NETTOYAGE RÉUSSI ---")
    print(f"Lignes restantes : {len(df)}")
    print(df.head(10))

if __name__ == "__main__":
    fix_and_clean()