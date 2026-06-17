# Dashboard — Bluesky Analytics

Dashboard interactif de visualisation et d'analyse des commentaires Bluesky, construit avec Streamlit et Plotly.

## Lancer le dashboard

```bash
streamlit run dashboard.py
```

Ouvrir ensuite dans le navigateur : **http://localhost:8501**

## Prérequis

```bash
pip install streamlit plotly pandas wordcloud matplotlib
```

## Structure des données attendues

Le dashboard lit automatiquement le fichier `data/labeled_data_with_topics.csv` avec les colonnes suivantes :

| Colonne | Description |
|---|---|
| `text` | Texte du post Bluesky |
| `language` | Langue détectée |
| `sentiment` | `Positif`, `Neutre` ou `Négatif` |
| `topic_id` | Identifiant du topic (`-1` = non classifié) |
| `topic` | Nom du topic |
| `topic_score` | Score de confiance du topic (0.0 à 1.0) |

Les métriques du modèle sont lues depuis `models/sentiment/export/metrics.json`.

## Filtres disponibles (sidebar)

Tous les filtres s'appliquent en temps réel sur l'ensemble des pages et visuels.

| Filtre | Type | Description |
|---|---|---|
| **Sentiment** | Multiselect | Filtrer par Positif, Neutre, Négatif |
| **Topics** | Multiselect | Sélectionner un ou plusieurs topics |
| **Posts non classifiés** | Checkbox | Inclure/exclure les posts sans topic |
| **Score de confiance** | Slider (0.0 – 1.0) | Garder uniquement les posts dans la plage choisie |
| **Mot-clé** | Texte libre | Rechercher un mot dans le contenu des posts |

Un compteur `X / 4 238 posts` indique en permanence le nombre de posts correspondant aux filtres actifs.

## Pages

### Vue d'ensemble
- 6 KPI cards : nombre de posts, topics détectés, % positif/neutre/négatif, score de confiance moyen
- Donut chart de répartition des sentiments
- Bar chart horizontal du nombre de posts par sentiment
- Word Cloud global des mots les plus fréquents
- Tableau d'aperçu des données filtrées (200 premiers posts)

### Sentiments
- 4 KPI cards : sentiment dominant, comptages par catégorie
- Donut chart détaillé avec légende
- Top 15 mots les plus fréquents par sentiment (sélecteur interactif)
- Bar chart empilé : répartition des sentiments par topic (Top 12)
- Word Clouds séparés pour les posts Positifs et Négatifs

### Topics
- 3 KPI cards : nombre de topics, topic dominant, moyenne de posts par topic
- Slider pour choisir le nombre de topics à afficher (5 à 30)
- Bar chart horizontal des Top N topics
- Donut chart de répartition des Top 10 topics
- Treemap des 25 premiers topics
- Radar chart : comparaison des sentiments entre les 8 premiers topics
- Word Cloud interactif par topic (sélection via liste déroulante)
- Box plot de distribution des scores de confiance par topic

### Performance du modèle
- 5 KPI cards : Accuracy, F1 Macro, F1 Weighted, Epochs, Confiance moyenne
- Bar chart de comparaison des métriques
- Courbes d'entraînement (Loss + Métriques)
- Histogramme de distribution des scores de confiance (données filtrées)

## Fichiers requis

```
Projet-etude/
├── dashboard.py
├── data/
│   └── labeled_data_with_topics.csv
└── models/
    └── sentiment/
        └── export/
            ├── metrics.json
            ├── training_loss.png
            └── training_metrics.png
```
