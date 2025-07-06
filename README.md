# Concrete_NN - Neural Network for Concrete Strength Regression

This project implements a custom Neural Network from scratch using PyTorch to predict the **compressive strength of concrete** based on its components. The model is designed for regression tasks using tabular data.

## Project Structure
Concrete_NN/

├── notebooks/   # Jupyter notebooks for training and evaluation

│   └── Regression_NN.ipynb

├── data/        # Dataset for training and testing

│   └── concrete_data.csv

├── requirements.txt

└── README.md

## Model Architecture

- **Input**: Numerical features representing concrete mix components (e.g., cement, water, aggregates).
- **Layers**:
  - Fully connected dense layers with ReLU activations
  - Dropout layers for regularization
  - Final output layer with a single neuron for regression output
- **Output**: Predicted compressive strength (continuous value)

## Training and Evaluation

- **Framework**: PyTorch
- **Optimization**:
  - Optimizer: Adam
  - Loss Function: Mean Squared Error (MSE)
  - Learning Rate Scheduler: StepLR (optional)
- **Metrics**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score

## Results

- **Validation MSE**: Mean = 43.48, Std = 0.00
- **Observations**:
  - The model achieves good generalization on the validation dataset.
  - Further improvements can be made by tuning hyperparameters or using feature engineering.

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/LordAmadeus1/Concrete_NN.git
   cd Concrete_NN

2. **Run the notebook:**
   jupyter notebook notebooks/Regression_NN.ipynb
   
