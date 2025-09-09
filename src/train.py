import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
import os
import time # Import the time module

from src.model import build_regression_model

def train_model(config_path='config/model_configs.yaml'):
    """
    Trains models, tracks losses, and measures training time, utilizing GPU if available.
    """
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/metrics', exist_ok=True)

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load configurations
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    global_hyperparams = configs.get('hyperparameters', {})

    # Load data
    data = pd.read_csv('data/processed/dataset.csv')
    X = data[['X']].values
    y = data['y'].values.reshape(-1, 1)

    # Split and move data to the selected device
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Loop through and train each model
    for model_name, model_config in configs.items():
        if model_name == 'hyperparameters':
            continue

        print(f"--- Training {model_name} ---")

        # Handle hyperparameters
        current_hyperparams = global_hyperparams.copy()
        if 'hyperparameters' in model_config:
            current_hyperparams.update(model_config['hyperparameters'])
        
        epochs = current_hyperparams.get('epochs', 100)
        batch_size = current_hyperparams.get('batch_size', 10)
        lr = current_hyperparams.get('learning_rate', 0.001)

        print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Build model and move it to the selected device
        model = build_regression_model(model_config['layers']).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses = [], []
        
        # --- Timing the Training ---
        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            # Note: DataLoader already puts tensors on the correct device if the dataset's tensors are
            for inputs, targets in train_loader:
                # inputs and targets are already on the correct device
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(X_train_tensor), y_train_tensor).item()
                val_loss = criterion(model(X_val_tensor), y_val_tensor).item()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % (epochs // 5) == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # --- End Timing and Report ---
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"Training finished in {training_duration:.2f} seconds.")

        # Save model and loss history
        torch.save(model.state_dict(), f'models/{model_name}.pth')
        print(f"Model {model_name} saved.")
        
        loss_df = pd.DataFrame({'epoch': range(1, epochs + 1), 'train_loss': train_losses, 'val_loss': val_losses})
        loss_df.to_csv(f'reports/metrics/{model_name}_losses.csv', index=False)
        print(f"Loss history for {model_name} saved.\n")

if __name__ == '__main__':
    train_model()