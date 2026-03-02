import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .data_loader import load_data  
import os

def plot_distributions(df, output_dir):
    """
    Plots the distributions of a subset of features.
    """
    features = list(df.columns[:6]) # first 6 features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(2, 3, i+1)
        sns.histplot(data=df, x=feature, hue='target', kde=True, element="step")
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()

def plot_correlation_matrix(df, output_dir):
    """
    Plots the correlation matrix of the features.
    """
    plt.figure(figsize=(30, 20))
    corr = df.drop('target', axis=1).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

def plot_target_distribution(df, output_dir):
    """
    Plots the distribution of the target classes.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x="target",
              data=df,
              hue="target",
              palette=["blue", "orange"],
              legend=True)

    plt.title('Target Class Distribution (0: Malignant, 1: Benign)')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()

if __name__ == "__main__":
    df = load_data()
    output_dir = "/home/zartak/Documents/breast_cancer_classification/results/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating plots...")
    plot_distributions(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    plot_target_distribution(df, output_dir)
    print(f"Plots saved to {output_dir}")
