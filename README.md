# Breast Cancer Classification

Predicts whether a breast tumour is **Malignant** or **Benign** using the [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (569 samples, 30 features). Three classical ML models and a deep learning baseline are trained and evaluated.

---

## Models

| Model | Accuracy | Malignant Recall | AUC |
|---|---|---|---|
| Logistic Regression | 98.25% | **97.62%** | 0.9954 |
| SVM | 98.25% | **97.62%** | 0.9950 |
| Random Forest | 95.61% | 92.86% | 0.9932 |
| Deep Learning (MLP) | 95.61% | 97.62% | — |

**Logistic Regression and SVM are the recommended models** — both miss only 1 cancer case in 42 on the test set.  
The deep learning model is included for comparison but offers no advantage over classical models on this small structured dataset.

---

## Project Structure

```
breast_cancer_classification/
│
├── src/
│   ├── data_loader.py        # loads dataset, saves to data/
│   ├── preprocessing.py      # train/test split + StandardScaler
│   ├── data_exploration.py   # EDA plots (distributions, correlation matrix)
│   ├── evaluation.py         # metrics, confusion matrix, ROC curve
│   └── models/
│       ├── classical.py      # Logistic Regression, Random Forest, SVM
│       └── deep_learning.py  # MLP neural network (PyTorch)
│
├── data/                     # auto-generated CSV of the dataset
├── results/
│   ├── plots/                # all generated plots (PNG)
│   ├── classical_metrics.json
│   └── deep_learning_metrics.json
│
├── notebook/                 # Jupyter notebooks for exploration
├── requirements.txt
```

---

## How to Run

**1. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run classical models (Logistic Regression, Random Forest, SVM)**

```bash
python -m src.models.classical
```

Outputs metrics to `results/classical_metrics.json` and plots to `results/plots/`.

**3. Run the deep learning model**

```bash
python -m src.models.deep_learning
```

Outputs metrics to `results/deep_learning_metrics.json`.

---

> **Key note on evaluation**: All metrics treat **Malignant (class 0) as the positive class**.  
> Malignant Recall is the primary metric — missing a cancer diagnosis is the costliest possible error.
