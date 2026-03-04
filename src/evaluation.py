import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, brier_score_loss
)
import os
import json


def evaluate_model(y_true, y_pred, model_name, output_dir,
                   y_pred_probs=None, save_plots=True):
    """
    Calculates metrics and optionally plots confusion matrix and ROC curve.
    """

    report = classification_report(
        y_true, y_pred,
        target_names=["Malignant", "Benign"],
        output_dict=True
    )

    metrics = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, average='weighted'),
        "f1":               f1_score(y_true, y_pred, average='weighted'),
        "recall_malignant": report["Malignant"]["recall"], # how many cancer patients were correctly caught
        "recall_benign":    report["Benign"]["recall"], #how many healthy patients were correctly cleared
    }

    print(f"\nEvaluation Results for {model_name}:")
    print(f"  Accuracy                      : {metrics['accuracy']:.4f}")
    print(f"  Precision                     : {metrics['precision']:.4f}")
    print(f"  F1                            : {metrics['f1']:.4f}")
    print(f"  Recall    (Malignant)         : {metrics['recall_malignant']:.4f}")
    print(f"  Recall    (Benign)            : {metrics['recall_benign']:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Malignant", "Benign"]))

    if save_plots:
        # ---- Confusion Matrix ----
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign']
        )
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = os.path.join(
            output_dir,
            f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        )
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()

    # ---- ROC Curve & AUC  
    if y_pred_probs is not None:
      
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs, pos_label=0)
        roc_auc = auc(fpr, tpr)
        metrics["roc_auc"] = roc_auc

        # Brier Score: measures probability reliability (lower = better)
        brier = brier_score_loss(y_true, y_pred_probs, pos_label=0)
        metrics["brier_score"] = brier

        print(f"  ROC AUC                       : {roc_auc:.4f}")
        print(f"  Brier Score                   : {brier:.4f}  ")

        if save_plots:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f"ROC (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')
            plt.xlabel("False Positive Rate (Healthy flagged as Sick)")
            plt.ylabel("True Positive Rate / Recall (Cancer Caught)")
            plt.title(f"ROC Curve: {model_name}\n(Positive class = Malignant)")
            plt.legend(loc='lower right')
            roc_path = os.path.join(
                output_dir,
                f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            )
            plt.savefig(roc_path, bbox_inches='tight')
            plt.close()

    return metrics


def save_all_metrics(all_metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nAll metrics saved to {output_path}")
