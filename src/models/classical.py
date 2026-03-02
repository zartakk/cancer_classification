from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model, save_all_metrics
import os

def logistic_regression_method(X_train, X_test, y_train, y_test, output_dir):
    """Trains and evaluates a Logistic Regression model."""
    print("\n--- Training Logistic Regression ---")
    model = LogisticRegression(random_state=42, max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, "Logistic Regression", output_dir)

def random_forest_method(X_train, X_test, y_train, y_test, output_dir):
    """Trains and evaluates a Random Forest model."""
    print("\n--- Training Random Forest ---")
    method = RandomForestClassifier(random_state=42, n_estimators=200)
    method.fit(X_train, y_train)
    y_pred = method.predict(X_test)
    return evaluate_model(y_test, y_pred, "Random Forest", output_dir)

def svm_method(X_train, X_test, y_train, y_test, output_dir):
    """Trains and evaluates a Support Vector Machine model."""
    print("\n--- Training SVM ---")
    method = SVC(random_state=42, probability=True)
    method.fit(X_train, y_train)
    y_pred = method.predict(X_test)
    return evaluate_model(y_test, y_pred, "SVM", output_dir)

def train_classical_models():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_data()
    
    output_dir = "/home/zartak/Documents/breast_cancer_classification/results/plots"
    metrics_path = "/home/zartak/Documents/breast_cancer_classification/results/classical_metrics.json"
    os.makedirs(output_dir, exist_ok=True)
    
    models_results = {}
    
    # Run each model function
    models_results["Logistic Regression"] = logistic_regression_method(X_train, X_test, y_train, y_test, output_dir)
    models_results["Random Forest"] = random_forest_method(X_train, X_test, y_train, y_test, output_dir)
    models_results["SVM"] = svm_method(X_train, X_test, y_train, y_test, output_dir)
    
    # Save metrics
    save_all_metrics(models_results, metrics_path)
    
    return models_results

if __name__ == "__main__":
    train_classical_models()
