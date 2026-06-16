import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def export_training_artifacts(output_dir="models/sentiment"):
    """
    Exporte tous les artefacts d'entraînement (modèle, métriques, diagrammes).
    Génère aussi un fichier de configuration pour le frontend.
    """
    
    # Créer le dossier d'export
    export_dir = os.path.join(output_dir, "export")
    os.makedirs(export_dir, exist_ok=True)
    
    # Lire les résultats depuis le log d'entraînement
    trainer_state_file = os.path.join(output_dir, "trainer_state.json")
    
    # Si le fichier n'existe pas à la racine, chercher dans les checkpoints
    if not os.path.exists(trainer_state_file):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Prendre le dernier checkpoint (le plus récent numériquement)
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            trainer_state_file = os.path.join(output_dir, latest_checkpoint, "trainer_state.json")
            print(f"Recherche dans le checkpoint : {latest_checkpoint}")
    
    if not os.path.exists(trainer_state_file):
        print(f"Attention : {trainer_state_file} non trouvé.")
        return
    
    with open(trainer_state_file, "r") as f:
        trainer_state = json.load(f)
    
    # Extraire les logs d'entraînement
    logs = trainer_state.get("log_history", [])
    
    # Créer un DataFrame pour les métriques
    metrics_data = []
    for log in logs:
        if "loss" in log or "eval_loss" in log:
            metrics_data.append(log)
    
    if not metrics_data:
        print("Aucune métrique trouvée dans les logs.")
        return
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Générer les diagrammes
    print("Génération des diagrammes...")
    
    # 1. Courbes de loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if "loss" in df_metrics.columns:
        axes[0].plot(df_metrics["loss"], marker="o", label="Training Loss")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid()
    
    if "eval_loss" in df_metrics.columns:
        eval_df = df_metrics.dropna(subset=["eval_loss"])
        axes[1].plot(eval_df["eval_loss"], marker="s", color="orange", label="Validation Loss")
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid()
    
    loss_plot = os.path.join(export_dir, "training_loss.png")
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  → {loss_plot}")
    
    # 2. Metrics (Accuracy, F1, etc.)
    if "eval_accuracy" in df_metrics.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        eval_df = df_metrics.dropna(subset=["eval_accuracy"])
        
        axes[0].plot(eval_df["eval_accuracy"], marker="o", color="green")
        axes[0].set_title("Accuracy")
        axes[0].set_ylabel("Score")
        axes[0].grid()
        
        if "eval_f1_macro" in df_metrics.columns:
            f1_df = df_metrics.dropna(subset=["eval_f1_macro"])
            axes[1].plot(f1_df["eval_f1_macro"], marker="s", color="blue")
            axes[1].set_title("F1 (Macro)")
            axes[1].set_ylabel("Score")
            axes[1].grid()
        
        if "eval_f1_weighted" in df_metrics.columns:
            f1w_df = df_metrics.dropna(subset=["eval_f1_weighted"])
            axes[2].plot(f1w_df["eval_f1_weighted"], marker="^", color="purple")
            axes[2].set_title("F1 (Weighted)")
            axes[2].set_ylabel("Score")
            axes[2].grid()
        
        metrics_plot = os.path.join(export_dir, "training_metrics.png")
        plt.tight_layout()
        plt.savefig(metrics_plot, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  → {metrics_plot}")
    
    # 3. Exporter les métriques en JSON pour le frontend
    metrics_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "task": "sentiment_classification",
        "final_metrics": {},
        "training_epochs": trainer_state.get("epoch", 0),
        "best_model_checkpoint": trainer_state.get("best_model_checkpoint", ""),
    }
    
    # Récupérer les dernières métriques d'évaluation
    final_eval_logs = [log for log in logs if "eval_" in str(log.keys())]
    if final_eval_logs:
        final_log = final_eval_logs[-1]
        metrics_summary["final_metrics"] = {
            k: v for k, v in final_log.items() 
            if k.startswith("eval_") and not k.startswith("eval_loss")
        }
    
    metrics_json = os.path.join(export_dir, "metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  → {metrics_json}")
    
    # 4. Copier le label_map
    label_map_src = os.path.join(output_dir, "label_map.txt")
    if os.path.exists(label_map_src):
        label_map_dst = os.path.join(export_dir, "label_map.txt")
        with open(label_map_src, "r") as src:
            content = src.read()
        with open(label_map_dst, "w") as dst:
            dst.write(content)
        print(f"  → {label_map_dst}")
    
    # 5. Créer un fichier de configuration pour l'API
    api_config = {
        "model_path": os.path.abspath(output_dir),
        "model_type": "sentiment_classifier",
        "tokenizer": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "labels": ["Négatif", "Neutre", "Positif"],  # À adapter selon tes labels
        "artifacts": {
            "loss_plot": "training_loss.png",
            "metrics_plot": "training_metrics.png",
            "metrics_json": "metrics.json",
            "label_map": "label_map.txt",
        }
    }
    
    api_config_file = os.path.join(export_dir, "config.json")
    with open(api_config_file, "w") as f:
        json.dump(api_config, f, indent=2)
    print(f"  → {api_config_file}")
    
    print(f"\n✅ Artefacts exportés dans : {export_dir}")


if __name__ == "__main__":
    export_training_artifacts()
