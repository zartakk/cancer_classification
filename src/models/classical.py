from xml.parsers.expat import model
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model, save_all_metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

      

def logistic_regression_model(X_train, X_test, y_train, y_test, output_dir):
    """Trains and evaluates a Logistic Regression model."""
    print("\n--- Training Logistic Regression ---")
    model = LogisticRegression(random_state=42, max_iter=5000)
    model.fit(X_train, y_train)
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    base_metrics = evaluate_model(
        y_test,
        y_pred,
        "Logistic Regression",
        output_dir,
        y_pred_probs
    )

    calibrated_metrics = calibrate_and_evaluate(
        model,
        "Logistic Regression",
        X_train, X_test,
        y_train, y_test,
        output_dir
    )

    return {
        "base": base_metrics,
        "calibrated": calibrated_metrics
    }

def random_forest_model(X_train, X_test, y_train, y_test, output_dir):
    """Trains and evaluates a Random Forest model."""
    print("\n--- Training Random Forest ---")
    model = RandomForestClassifier(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)
    
    
    # ---- Feature Importance ----
    importances = model.feature_importances_
    feature_names = X_train.columns

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(20, 15))
    plt.barh(
        feature_importance_df["feature"][:10],
        feature_importance_df["importance"][:10]
    )
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances - Random Forest")
    plt.xlabel("Importance")

    fi_path = os.path.join(output_dir, "feature_importance_random_forest.png")
    plt.savefig(fi_path)
    plt.close()

    y_pred_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    base_metrics = evaluate_model(
        y_test,
        y_pred,
        "Random Forest",
        output_dir,
        y_pred_probs
    )

    calibrated_metrics = calibrate_and_evaluate(
        model,
        "Random Forest",
        X_train, X_test,
        y_train, y_test,
        output_dir
    )

    return {
        "base": base_metrics,
        "calibrated": calibrated_metrics
    }
   

def svm_model(X_train, X_test, y_train, y_test, output_dir):
    """Trains and evaluates a Support Vector Machine model."""
    print("\n--- Training SVM ---")
    model = SVC(random_state=42, probability=True)
    model.fit(X_train, y_train)

    y_pred_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    base_metrics = evaluate_model(
        y_test,
        y_pred,
        "SVM",
        output_dir,
        y_pred_probs
    )

    calibrated_metrics = calibrate_and_evaluate(
        model,
        "SVM",
        X_train, X_test,
        y_train, y_test,
        output_dir
    )

    return {
        "base": base_metrics,
        "calibrated": calibrated_metrics
    }
def calibrate_and_evaluate(model, model_name,
                           X_train, X_test,
                           y_train, y_test,
                           output_dir):
    """
    Calibrates a trained model using Platt scaling
    and evaluates the calibrated version.
    """

    print(f"\n--- Calibrating {model_name} ---")

    calibrated_model = CalibratedClassifierCV(
        model,
        method='sigmoid',
        cv=5
    )

    calibrated_model.fit(X_train, y_train)

    y_probs_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_calibrated = (y_probs_calibrated >= 0.5).astype(int)

    return evaluate_model(
        y_test,
        y_pred_calibrated,
        model_name + " (Calibrated)",
        output_dir,
        y_probs_calibrated
    )


def train_classical_models():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_data()
    
    output_dir = "/home/zartak/Documents/breast_cancer_classification/results/plots"
    metrics_path = "/home/zartak/Documents/breast_cancer_classification/results/classical_metrics.json"
    os.makedirs(output_dir, exist_ok=True)
    
    models_results = {}
    
    # Run each model function
    models_results["Logistic Regression"] = logistic_regression_model(X_train, X_test, y_train, y_test, output_dir)
    models_results["Random Forest"] = random_forest_model(X_train, X_test, y_train, y_test, output_dir)
    models_results["SVM"] = svm_model(X_train, X_test, y_train, y_test, output_dir)
    
    # Save metrics
    save_all_metrics(models_results, metrics_path)
    
    return models_results

if __name__ == "__main__":
    train_classical_models()
