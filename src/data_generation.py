import numpy as np
import pandas as pd
import os

def generate_data(n_samples=200, save_path='data/processed'):
    """Generates and saves a synthetic dataset."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10
    y = 3 * np.sin(X).ravel() + np.random.randn(n_samples) * 0.5 + X.ravel()
    
    dataset = pd.DataFrame({'X': X.ravel(), 'y': y})
    dataset.to_csv(os.path.join(save_path, 'dataset.csv'), index=False)
    print(f"Dataset saved to {os.path.join(save_path, 'dataset.csv')}")

if __name__ == '__main__':
    generate_data()