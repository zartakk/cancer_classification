import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import json

def evaluate_model(y_true, y_pred, model_name, output_dir):
    """
    Calculates metrics and plots a confusion matrix.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    print(f"\nEvaluation Results for {model_name}:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
    
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
    
    return metrics

def save_all_metrics(all_metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nAll metrics saved to {output_path}")
