import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model, save_all_metrics
import os

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

def train_dl_model():
    X_train, X_test, y_train, y_test, _ = preprocess_data()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model, Loss, Optimizer
    model = SimpleNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop
    epochs = 50
    print(f"\n--- Training Deep Learning Model (PyTorch) for {epochs} epochs ---")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_probs = model(X_test_tensor)
        y_pred = (y_pred_probs > 0.5).float().numpy().flatten()
    
    output_dir = "/home/zartak/Documents/breast_cancer_classification/results/plots"
    metrics_path = "/home/zartak/Documents/breast_cancer_classification/results/deep_learning_metrics.json"

    metrics = evaluate_model(y_test, y_pred, "Deep Learning", output_dir)

    save_all_metrics({"Deep Learning": metrics}, metrics_path)
    
    return metrics

if __name__ == "__main__":
    train_dl_model()
