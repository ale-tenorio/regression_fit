# ML_fit_line
## Project structure
```
your_project_name/
├── config/
│   └── model_configs.yaml      # <-- Define models and hyperparameters here
├── data/
│   └── processed/              # <-- Stores the generated dataset
├── models/                     # <-- Stores trained model weights (.pth files)
├── reports/
│   ├── figures/                # <-- Stores output plots (.png files)
│   └── metrics/                # <-- Stores training/validation loss history (.csv files)
├── src/
│   ├── data_generation.py      # Script to create the synthetic dataset
│   ├── model.py                # Defines the PyTorch NN model builder
│   ├── train.py                # Main script for training the models
│   ├── evaluate.py             # Script to plot model predictions vs. actual data
│   └── plot_curves.py          # Script to plot learning curves
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```
## How to use
### 1. Clone the repo
```
git clone <your-repository-url>
cd your_project_name
```
### 2. Create Virtual Environment
```
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Run
Follow these steps in order to generate the data, train the models, and visualize the results.
```
bash
python -m src.data_generation
python -m src.train
python -m src.plot_curves
python -m src.evaluate
```
