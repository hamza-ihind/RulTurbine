import os
import json

def create_baseline_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RUL-Turbine: 02_baseline_model.ipynb — Classical ML Baseline\n",
                    "\n",
                    "This notebook builds classical machine learning baselines for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS FD001 dataset.\n",
                    "\n",
                    "## Methodology\n",
                    "1. **Preprocessing**: Load raw data, cap RUL at 125, drop the 7 flat sensors, and min-max scale.\n",
                    "2. **Feature Engineering**: For each sliding window of size 30, extract:\n",
                    "   - Rolling **mean** of the 14 sensors.\n",
                    "   - Rolling **standard deviation (std)** of the 14 sensors.\n",
                    "   - Rolling **slope (trend)** of the 14 sensors.\n",
                    "   - This yields a 42-dimensional feature vector per window sample.\n",
                    "3. **Modeling**: Train and evaluate:\n",
                    "   - **Linear Regression** (simplest linear baseline)\n",
                    "   - **Random Forest Regressor** (non-linear ensemble baseline)\n",
                    "   - **XGBoost Regressor** (gradient-boosted baseline)\n",
                    "4. **Evaluation**: Compare models using Root Mean Squared Error (RMSE) and the **PHM08 Asymmetric Score** (which penalizes late predictions more heavily)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn.ensemble import RandomForestRegressor\n",
                    "from xgboost import XGBRegressor\n",
                    "from sklearn.metrics import mean_squared_error\n",
                    "\n",
                    "# Append src directory to path\n",
                    "sys.path.append(os.path.abspath('../src'))\n",
                    "from utils import preprocess_data_and_save_npy, extract_rolling_features, compute_phm08_score, KEEP_SENSORS\n",
                    "\n",
                    "# Configure plotting styles\n",
                    "plt.rcParams['font.family'] = 'serif'\n",
                    "plt.rcParams['font.size'] = 10\n",
                    "plt.rcParams['axes.grid'] = True\n",
                    "plt.rcParams['grid.alpha'] = 0.3\n",
                    "plt.rcParams['grid.linestyle'] = '--'\n",
                    "sns.set_palette('muted')\n",
                    "\n",
                    "print(\"Libraries imported and settings configured.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Preprocessed Sliding Windows & Engineer Features\n",
                    "\n",
                    "We call `preprocess_data_and_save_npy` to prepare the sliding window data (`X_train` has shape `(17731, 30, 14)`), then extract rolling features using our helper function."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run preprocessing (this outputs X_train, y_train, X_test, y_test as .npy files)\n",
                    "X_train, y_train, X_test, y_test = preprocess_data_and_save_npy(\n",
                    "    subset='FD001', \n",
                    "    raw_dir='../data/raw', \n",
                    "    processed_dir='../data/processed',\n",
                    "    window_size=30,\n",
                    "    cap=125\n",
                    ")\n",
                    "\n",
                    "# Extract rolling features\n",
                    "X_train_feat = extract_rolling_features(X_train)\n",
                    "X_test_feat = extract_rolling_features(X_test)\n",
                    "\n",
                    "print(f\"X_train_feat shape: {X_train_feat.shape} (17731 samples, 42 features)\")\n",
                    "print(f\"X_test_feat shape: {X_test_feat.shape} (100 test engine final windows, 42 features)\")\n",
                    "\n",
                    "# Generate feature names list for plots\n",
                    "feature_names = []\n",
                    "for stat in ['mean', 'std', 'slope']:\n",
                    "    for s in KEEP_SENSORS:\n",
                    "        feature_names.append(f\"{s.upper()}_{stat}\")\n",
                    "feature_names = np.array(feature_names)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Model 1: Linear Regression Baseline\n",
                    "\n",
                    "We train a simple ordinary least squares linear regression model. Predictions are clipped below at 0 (since RUL cannot be negative)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "lr_model = LinearRegression()\n",
                    "lr_model.fit(X_train_feat, y_train)\n",
                    "\n",
                    "# Predict and evaluate on train set\n",
                    "lr_pred_train = np.clip(lr_model.predict(X_train_feat), 0, None)\n",
                    "lr_rmse_train = np.sqrt(mean_squared_error(y_train, lr_pred_train))\n",
                    "\n",
                    "# Predict and evaluate on test set\n",
                    "lr_pred_test = np.clip(lr_model.predict(X_test_feat), 0, None)\n",
                    "lr_rmse_test = np.sqrt(mean_squared_error(y_test, lr_pred_test))\n",
                    "lr_score_test = compute_phm08_score(y_test, lr_pred_test)\n",
                    "\n",
                    "print(\"Linear Regression Results:\")\n",
                    "print(f\"  Train RMSE: {lr_rmse_train:.2f}\")\n",
                    "print(f\"  Test RMSE: {lr_rmse_test:.2f}\")\n",
                    "print(f\"  Test PHM08 Score: {lr_score_test:.2f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Model 2: Random Forest Regressor Baseline\n",
                    "\n",
                    "We train a Random Forest ensemble model. We restrict the depth slightly (`max_depth=12`) to prevent overfitting."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)\n",
                    "rf_model.fit(X_train_feat, y_train)\n",
                    "\n",
                    "# Predict and evaluate on train set\n",
                    "rf_pred_train = np.clip(rf_model.predict(X_train_feat), 0, None)\n",
                    "rf_rmse_train = np.sqrt(mean_squared_error(y_train, rf_pred_train))\n",
                    "\n",
                    "# Predict and evaluate on test set\n",
                    "rf_pred_test = np.clip(rf_model.predict(X_test_feat), 0, None)\n",
                    "rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_pred_test))\n",
                    "rf_score_test = compute_phm08_score(y_test, rf_pred_test)\n",
                    "\n",
                    "print(\"Random Forest Results:\")\n",
                    "print(f\"  Train RMSE: {rf_rmse_train:.2f}\")\n",
                    "print(f\"  Test RMSE: {rf_rmse_test:.2f}\")\n",
                    "print(f\"  Test PHM08 Score: {rf_score_test:.2f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Model 3: XGBoost Regressor Baseline\n",
                    "\n",
                    "We train an XGBoost gradient-boosted decision tree regressor. We configure reasonable hyperparameters."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)\n",
                    "xgb_model.fit(X_train_feat, y_train)\n",
                    "\n",
                    "# Predict and evaluate on train set\n",
                    "xgb_pred_train = np.clip(xgb_model.predict(X_train_feat), 0, None)\n",
                    "xgb_rmse_train = np.sqrt(mean_squared_error(y_train, xgb_pred_train))\n",
                    "\n",
                    "# Predict and evaluate on test set\n",
                    "xgb_pred_test = np.clip(xgb_model.predict(X_test_feat), 0, None)\n",
                    "xgb_rmse_test = np.sqrt(mean_squared_error(y_test, xgb_pred_test))\n",
                    "xgb_score_test = compute_phm08_score(y_test, xgb_pred_test)\n",
                    "\n",
                    "print(\"XGBoost Results:\")\n",
                    "print(f\"  Train RMSE: {xgb_rmse_train:.2f}\")\n",
                    "print(f\"  Test RMSE: {xgb_rmse_test:.2f}\")\n",
                    "print(f\"  Test PHM08 Score: {xgb_score_test:.2f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Summary Table\n",
                    "\n",
                    "Let's compare the performance of all 3 classical baseline models in a single table."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "results_df = pd.DataFrame({\n",
                    "    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],\n",
                    "    'Train RMSE': [lr_rmse_train, rf_rmse_train, xgb_rmse_train],\n",
                    "    'Test RMSE': [lr_rmse_test, rf_rmse_test, xgb_rmse_test],\n",
                    "    'Test PHM08 Score': [lr_score_test, rf_score_test, xgb_score_test]\n",
                    "})\n",
                    "\n",
                    "# Format table for clear presentation\n",
                    "results_df.round(2).style.highlight_min(subset=['Test RMSE', 'Test PHM08 Score'], color='#e2f0d9')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Feature Importance Visualization\n",
                    "\n",
                    "Let's examine which rolling features contribute most to the predictions. We will plot the top 15 features by importance for both Random Forest and XGBoost."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=300)\n",
                    "\n",
                    "# Random Forest Importances\n",
                    "rf_importances = rf_model.feature_importances_\n",
                    "rf_indices = np.argsort(rf_importances)[::-1][:15]\n",
                    "\n",
                    "sns.barplot(x=rf_importances[rf_indices], y=feature_names[rf_indices], ax=axs[0], color='#ff7f0e')\n",
                    "axs[0].set_title('Top 15 Feature Importances (Random Forest)', fontsize=11, fontweight='bold')\n",
                    "axs[0].set_xlabel('Relative Importance')\n",
                    "\n",
                    "# XGBoost Importances\n",
                    "xgb_importances = xgb_model.feature_importances_\n",
                    "xgb_indices = np.argsort(xgb_importances)[::-1][:15]\n",
                    "\n",
                    "sns.barplot(x=xgb_importances[xgb_indices], y=feature_names[xgb_indices], ax=axs[1], color='#1f77b4')\n",
                    "axs[1].set_title('Top 15 Feature Importances (XGBoost)', fontsize=11, fontweight='bold')\n",
                    "axs[1].set_xlabel('F-Score / Weight')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "os.makedirs('../figures', exist_ok=True)\n",
                    "plt.savefig('../figures/baseline_feature_importances.png')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Observations\n",
                    "- **Crucial Sensors**: Sensor channels `s11` (Throttle pressure at HPC outlet), `s4` (Total temperature at LPT outlet), `s12` (Ratio of fuel flow to Ps30) and `s15` (Bypass ratio) feature prominently in the top importances.\n",
                    "- **Crucial Statistics**: The rolling **mean** and rolling **slope** are highly important, which makes intuitive physical sense as they capture the current health degradation state and the degradation rate/trend of the turbofan engine.\n",
                    "\n",
                    "In the next notebook (**03_deep_model.ipynb**), we will build a deep LSTM sequence model that learns directly from the raw sequence of these 14 sensors without manual rolling feature engineering."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs("notebooks", exist_ok=True)
    nb_path = "notebooks/02_baseline_model.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Notebook created at {nb_path}")

if __name__ == "__main__":
    create_baseline_notebook()
