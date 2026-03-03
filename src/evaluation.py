import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
import os
import json
from sklearn.metrics import brier_score_loss

def evaluate_model(y_true, y_pred, model_name, output_dir, y_probs=None):
    """
    Calculates metrics and plots a confusion matrix.
    """
    
    report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"], output_dict=True)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall_malignant": report["Malignant"]["recall"],
        "recall_benign": report["Benign"]["recall"]
    }
    
    print(f"\nEvaluation Results for {model_name}:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Malignant", "Benign"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(cm_path)
    plt.close()
    
    # ROC Curve
    if y_probs is not None:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name}")
        plt.legend()

        roc_path = os.path.join(
            output_dir,
            f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
        )
        plt.savefig(roc_path)
        plt.close()

        metrics["roc_auc"] = roc_auc
        
        # ---- Brier Score ----
        brier = brier_score_loss(y_true, y_probs)
        metrics["brier_score"] = brier

    return metrics

def save_all_metrics(all_metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nAll metrics saved to {output_path}")
