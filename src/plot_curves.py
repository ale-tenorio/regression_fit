import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

def plot_learning_curves(config_path='config/model_configs.yaml'):
    """
    Plots the training and validation learning curves for all models
    defined in the config file.
    """
    # Load model names from the config file
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)

    # Setup the plot
    # We use subplots to give each model its own learning curve chart
    model_names = [name for name in configs if name != 'hyperparameters']
    num_models = len(model_names)
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 6 * num_models), sharex=True)
    if num_models == 1: # Make axes an array even if there's one subplot
        axes = [axes]

    fig.suptitle('Model Learning Curves', fontsize=16)

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        loss_file = f'reports/metrics/{model_name}_losses.csv'
        
        if not os.path.exists(loss_file):
            print(f"Loss file not found for {model_name}. Skipping.")
            continue
        
        # Read and plot data
        loss_df = pd.read_csv(loss_file)
        ax.plot(loss_df['epoch'], loss_df['train_loss'], label='Training Loss', c='black')
        ax.plot(loss_df['epoch'], loss_df['val_loss'], label='Validation Loss', linestyle='--', c='blue')
        
        ax.set_title(model_name)
        ax.set_ylabel('MSE (Loss)')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.7, color='gray')

    # Set common X-axis label
    plt.xlabel('Epoch')
    
    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust for suptitle
    os.makedirs('reports/figures', exist_ok=True)
    save_path = 'reports/figures/learning_curves.png'
    plt.savefig(save_path)
    print(f"Learning curves plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_learning_curves()