# Detailed Cell-by-Cell Breakdown: Notebook 03 — Deep Sequence Model (LSTM)

This document provides an exhaustive, cell-by-cell explanation of **03_deep_model.ipynb**, covering the code logic, mathematical formulations, outputs, physical interpretations, and the architectural transition from classical baselines to deep learning.

---

## Architectural Transition: From Notebook 2 (Baseline) to Notebook 3 (LSTM)

In **Notebook 2 (Classical ML Baselines)**:
- **Feature Collapse**: Sliding 30-cycle windows were manually aggregated into static summary statistics (**mean**, **standard deviation**, and **slope**) per sensor channel.
- **Dimensionality**: Reduced the temporal sequence of `(30, 14)` to a flat 42-dimensional vector (`[30, 14] -> 42`).
- **Limitation**: The temporal order of time steps within the window was lost, and model performance relied entirely on human feature engineering assumptions.

In **Notebook 3 (Deep Sequence Model - LSTM)**:
- **3D Sequence Retention**: Preserves the raw sequence dimension `(batch_size, sequence_length=30, features=14)`.
- **Representation Learning**: The LSTM network automatically learns hidden temporal dynamics, trend gradients, and inter-sensor cross-correlations directly from raw time-series data.
- **Group-Based Splitting**: Replaces simple random window splitting with **unit-level splitting** (engines 1–80 for training, 81–100 for validation) to strictly eliminate data leakage between overlapping windows of the same engine.

---

## Detailed Cell-by-Cell Analysis

### **Cell 1 (Markdown) — Notebook Overview & Objectives**
* **Purpose**: Sets up the goal, justification for choosing PyTorch, and the 7-step pipeline methodology.
* **Key Concepts**:
  * **Why PyTorch**: Offers modularity (`Dataset`, `DataLoader`, `nn.Module`) and explicit control over epoch-by-epoch validation tracking.
  * **Pipeline Steps**: Group-based validation split $\rightarrow$ Min-Max Scaling $\rightarrow$ 3D Sliding Window Sequence Creation $\rightarrow$ PyTorch DataLoader $\rightarrow$ 2-layer LSTM Network $\rightarrow$ 35-Epoch Training $\rightarrow$ Test Set Evaluation (RMSE & PHM08 Score).

---

### **Cell 2 (Code) — Imports, Configuration & Device Setup**
* **Purpose**: Loads essential Python modules, initializes PyTorch device settings, sets random seeds for reproducibility, registers custom typography (`Alegreya`), and defines corporate plot palettes.
* **Code Breakdown**:
  ```python
  import torch
  import torch.nn as nn
  from torch.utils.data import Dataset, DataLoader
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.metrics import mean_squared_error
  ```
  * **Seed Setting**: `torch.manual_seed(42)` and `np.random.seed(42)` guarantee identical weight initialization and dataset shuffling across runs.
  * **Device Assignment**: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` dynamically selects GPU hardware acceleration if available, falling back to CPU.
* **Execution Output**:
  ```text
  Alegreya font registered successfully.
  Using device: cpu
  ```

---

### **Cell 3 (Markdown)**
* **Purpose**: Section Header for **Section 1: Group-Based Split, Scaling, and Windowing**.

---

### **Cell 4 (Code) — Data Preparation & 3D Sequence Generation**
* **Purpose**: Performs unit-level train/validation splitting, fits scaling parameters strictly on training units, and formats the time-series into 3D sliding window tensors.
* **Code Breakdown**:
  1. **Piecewise Target RUL**: Calls `get_piecewise_rul(train_df, cap=125)` to clip targets at 125 cycles.
  2. **Group-Based Validation Split**:
     - `train_units`: Engines 1 to 80 (`unit <= 80`).
     - `val_units`: Engines 81 to 100 (`unit > 80`).
     - *Why this matters*: Prevents overlapping windows from the same engine appearing in both train and validation sets, eliminating data leakage.
  3. **Data Scaling**: `MinMaxScaler` is fit exclusively on `train_units[KEEP_SENSORS]` and then used to transform `val_units` and `test_units`.
  4. **Sliding Window Helper (`create_sequences`)**:
     - Loops over each engine unit.
     - Slides a window of size $W=30$ over the 14 kept sensors.
     - Generates input tensor $X$ of shape `(N_samples, 30, 14)` and target vector $y$ of shape `(N_samples,)` (taking the target RUL at the last step of each window).
  5. **Test Set Sequences**: Builds a single window of size 30 using the **final 30 cycles** recorded for each of the 100 test engines.
* **Execution Output**:
  ```text
  X_train shape: (13818, 30, 14) | y_train shape: (13818,)
  X_val shape:   (3913, 30, 14)  | y_val shape:   (3913,)
  X_test shape:  (100, 30, 14)   | y_test shape:  (100,)
  ```

---

### **Cell 5 (Markdown)**
* **Purpose**: Section Header for **Section 2: Define PyTorch Dataset and Model**.

---

### **Cell 6 (Code) — PyTorch Dataset & LSTM Network Architecture**
* **Purpose**: Wraps sequence arrays into PyTorch `Dataset`/`DataLoader` objects and constructs the neural network model.
* **Code Breakdown**:
  * **`CMAPSSDataset` Class**: Converts NumPy arrays into 32-bit floating point PyTorch Tensors (`torch.float32`) and adds a unit dimension to targets via `.unsqueeze(1)`.
  * **`DataLoader`**: Configures batch size to 128 (`batch_size=128`), enabling mini-batch Stochastic Gradient Descent. `shuffle=True` is enabled for training and `False` for validation.
  * **`LSTMRulModel` Class**:
    - **LSTM Backbone**: 2 stacked LSTM layers (`num_layers=2`) with `input_dim=14`, `hidden_dim=64`, `batch_first=True`, and `dropout=0.2` between layers.
    - **Forward Pass Extraction**: Processes sequence $X \in \mathbb{R}^{B \times 30 \times 14}$ and extracts the final time step's hidden state:
      $$h_{last} = \text{out}[:, -1, :] \in \mathbb{R}^{B \times 64}$$
    - **Regression Head**: `nn.Sequential(Linear(64, 32), ReLU(), Dropout(0.2), Linear(32, 1))` maps the 64-dimensional temporal context vector down to a scalar RUL prediction.
* **Execution Output**:
  ```text
  LSTMRulModel(
    (lstm): LSTM(14, 64, num_layers=2, batch_first=True, dropout=0.2)
    (regressor): Sequential(
      (0): Linear(in_features=64, out_features=32, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.2, inplace=False)
      (3): Linear(in_features=32, out_features=1, bias=True)
    )
  )
  ```

---

### **Cell 7 (Markdown)**
* **Purpose**: Section Header for **Section 3: Training Loop**.

---

### **Cell 8 (Code) — Model Training & Validation Tracking**
* **Purpose**: Executes the 35-epoch training loop, optimizing parameters using Adam and computing loss metrics.
* **Code Breakdown**:
  * **Criterion**: `nn.MSELoss()` (Mean Squared Error: $\frac{1}{N} \sum (y_{pred} - y_{true})^2$).
  * **Optimizer**: `torch.optim.Adam(model.parameters(), lr=0.001)`.
  * **Training Phase**: Sets `model.train()`, iterates over `train_loader`, performs zeroing of gradients (`optimizer.zero_grad()`), forward pass, loss calculation, backpropagation (`loss.backward()`), and parameter update (`optimizer.step()`).
  * **Validation Phase**: Sets `model.eval()` and wraps iteration in `with torch.no_grad():` to disable gradient computation and conserve memory.
  * Prints average train and validation loss at each epoch.
* **Execution Output**:
  ```text
  Epoch 1/35  | Train Loss: 6984.72 | Val Loss: 5823.05
  Epoch 15/35 | Train Loss: 793.50  | Val Loss: 422.24
  Epoch 30/35 | Train Loss: 313.32  | Val Loss: 203.50
  Epoch 35/35 | Train Loss: 299.66  | Val Loss: 189.57
  Training complete!
  ```

---

### **Cell 9 (Markdown)**
* **Purpose**: Section Header for **Section 4: Plot Loss Curves**.

---

### **Cell 10 (Code) — Loss Curve Visualization**
* **Purpose**: Plots training vs. validation MSE loss over the 35 epochs to analyze convergence and check for overfitting.
* **Code Breakdown**: Uses Matplotlib to chart `history['train_loss']` in primary blue (`#4A7CDE`) and `history['val_loss']` in dark navy (`#2E2C7C`), saving the plot to `../figures/lstm_loss_curves.png`.
* **Execution Output**: Displays and saves the loss curve plot.
  - *Key Observation*: Validation loss closely tracks training loss without diverging upward, proving that dropout ($0.2$) and group-based splitting successfully prevented overfitting.

---

### **Cell 11 (Markdown)**
* **Purpose**: Section Header for **Section 5: Evaluation on Test Set**.

---

### **Cell 12 (Code) — Test Set Evaluation & Model Persistence**
* **Purpose**: Evaluates the trained model on the 100 test engines, computes RMSE and PHM08 scores, and saves predictions and model weights to disk.
* **Code Breakdown**:
  1. Converts `X_test` to tensor and passes it through the model in evaluation mode (`with torch.no_grad()`).
  2. Applies `np.clip(predictions, 0, None)` to prevent negative RUL predictions.
  3. Calculates **Test RMSE**: $\sqrt{\text{MSE}(y_{test}, y_{pred})} = 14.87$ cycles.
  4. Calculates **Test PHM08 Score**: $S = 352.64$ using `compute_phm08_score()`.
  5. Saves array of test predictions to `../data/processed/lstm_predictions.npy`.
  6. Saves model weights to `../models/lstm_model.pt`.
* **Execution Output**:
  ```text
  LSTM Model Evaluation:
    Test RMSE: 14.87 cycles
    Test PHM08 Score: 352.64
  Predictions saved for comparison notebook.
  Model weights saved successfully.
  ```

---

### **Cell 13 (Markdown)**
* **Purpose**: Section Header for **Section 6: Sample Engine Predictions Plot**.

---

### **Cell 14 (Code) — Scatter Plot of Predicted vs. True RUL**
* **Purpose**: Visualizes prediction accuracy across the 100 test engines using a scatter plot relative to a perfect prediction reference line ($y=x$).
* **Code Breakdown**:
  * Plots `y_test` vs. `predictions` using slate blue scatter points (`#6B78B0`).
  * Draws a dashed reference line $y=x$ (`#3CA4F0`).
  * Saves figure to `../figures/lstm_predictions_scatter.png`.
* **Execution Output**: Displays and saves scatter plot.
  - *Key Observation*: Points cluster tightly around the $y=x$ line, especially for engines near failure ($RUL < 40$).

---

### **Cell 15 (Markdown) — Summary Observations & Next Steps**
* **Purpose**: Summarizes model performance and introduces **Notebook 04 (Results Comparison & Benchmarking)**.
* **Key Observations**:
  - The deep LSTM model achieves a strong Test RMSE of 14.87 cycles and PHM08 score of 352.64.
  - Outperforms Linear Regression (16.29) and learns sequence representations directly without manual feature engineering.
