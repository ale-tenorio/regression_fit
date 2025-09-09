import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from src.model import build_regression_model

def evaluate_and_plot(config_path='config/model_configs.yaml'):
    """Evaluates models, using GPU for inference if available."""

    # --- Device Configuration for Inference ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}\n")

    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)

    data = pd.read_csv('data/processed/dataset.csv')
    X = data[['X']].values
    y = data['y'].values

    plt.figure(figsize=(14, 8))
    plt.scatter(X, y, label='Original Data', color='black', alpha=0.5, s=15)
    
    # Prepare plot data and move to device
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32).to(device)

    for model_name, config in configs.items():
        if model_name == 'hyperparameters':
            continue
        
        # Build model, load state, and move to device
        model = build_regression_model(config['layers'])
        model.load_state_dict(torch.load(f'models/{model_name}.pth'))
        model.to(device) # Move model to the selected device
        model.eval()

        with torch.no_grad():
            # Get predictions and move them back to cpu for plotting with numpy/matplotlib
            y_pred = model(X_plot_tensor).cpu().numpy()
        plt.plot(X_plot, y_pred, label=f'{model_name} Fit', linewidth=2)

    # Polynomial fit (runs on CPU via scikit-learn)
    poly_features = PolynomialFeatures(degree=10)
    X_poly = poly_features.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    X_plot_poly = poly_features.transform(X_plot)
    y_poly_pred = poly_reg.predict(X_plot_poly)
    plt.plot(X_plot, y_poly_pred, label='10th Order Polynomial Fit', linestyle='--', color='gray', linewidth=2)

    plt.title('Model Comparison: Neural Networks vs. Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.ylim(min(y) - 1, max(y) + 1)
    
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/model_comparison.png')
    print("Plot saved to reports/figures/model_comparison.png")
    plt.show()

if __name__ == '__main__':
    evaluate_and_plot()