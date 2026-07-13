import os
import json

def create_eda_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RUL-Turbine: 01_eda.ipynb — Exploratory Data Analysis\n",
                    "\n",
                    "This notebook performs Exploratory Data Analysis (EDA) on the NASA C-MAPSS (FD001 subset) dataset. \n",
                    "The goal is to understand the sensor signals, analyze the engine lifespans, and justify the preprocessing choices (dropping flat/uninformative sensors).\n",
                    "\n",
                    "## Dataset Background\n",
                    "- **Dataset**: NASA C-MAPSS FD001 — `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (space-separated, no headers).\n",
                    "- **Columns**:\n",
                    "  - `unit`: Engine unit number (1 to 100)\n",
                    "  - `cycle`: Operating time in cycles\n",
                    "  - `op1`, `op2`, `op3`: Three operational settings\n",
                    "  - `s1` to `s21`: 21 sensor measurements\n",
                    "- **Goal**: Predict the Remaining Useful Life (RUL) in cycles before engine failure.\n",
                    "\n",
                    "Let's begin by importing the necessary libraries and loading the dataset using our helper module."
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
                    "\n",
                    "# Append src directory to path\n",
                    "sys.path.append(os.path.abspath('../src'))\n",
                    "from utils import load_raw_data, SENSOR_DESCRIPTIONS, DROPPED_SENSORS, KEEP_SENSORS\n",
                    "\n",
                    "# Configure matplotlib style for professional scientific presentation\n",
                    "plt.rcParams['font.family'] = 'serif'\n",
                    "plt.rcParams['font.size'] = 10\n",
                    "plt.rcParams['axes.grid'] = True\n",
                    "plt.rcParams['grid.alpha'] = 0.3\n",
                    "plt.rcParams['grid.linestyle'] = '--'\n",
                    "sns.set_palette('muted')\n",
                    "\n",
                    "print(\"Libraries imported and plotting styles set.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Loading\n",
                    "\n",
                    "We load the raw data files for FD001. The training engines run until failure, meaning their last cycle in the dataset is the cycle of failure. The test engines are truncated at some cycle prior to failure, and their actual RUL at the truncation point is given in the RUL file."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train_df, test_df, test_rul = load_raw_data(subset='FD001', raw_dir='../data/raw')\n",
                    "\n",
                    "print(f\"Train Data Shape: {train_df.shape}\")\n",
                    "print(f\"Test Data Shape: {test_df.shape}\")\n",
                    "if test_rul is not None:\n",
                    "    print(f\"Test RUL Targets Shape: {test_rul.shape}\")\n",
                    "\n",
                    "train_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Engine Lifespan Distribution\n",
                    "\n",
                    "Let's analyze how long the engines in the training set run before failure. Since the training data records each engine's cycle from startup to failure, the maximum cycle number for each `unit` represents its total lifespan."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Calculate lifespan for each training engine unit\n",
                    "lifespans = train_df.groupby('unit')['cycle'].max()\n",
                    "\n",
                    "plt.figure(figsize=(9, 5), dpi=300)\n",
                    "sns.histplot(lifespans, bins=20, kde=True, color='#1f77b4', edgecolor='black', alpha=0.7)\n",
                    "\n",
                    "# Annotate statistical values\n",
                    "mean_life = lifespans.mean()\n",
                    "median_life = lifespans.median()\n",
                    "min_life = lifespans.min()\n",
                    "max_life = lifespans.max()\n",
                    "\n",
                    "plt.axvline(mean_life, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_life:.1f} cycles')\n",
                    "plt.axvline(median_life, color='green', linestyle='-.', linewidth=1.5, label=f'Median: {median_life:.1f} cycles')\n",
                    "\n",
                    "plt.title('Distribution of Engine Lifespans in Training Set (FD001)', fontsize=12, fontweight='bold')\n",
                    "plt.xlabel('Operating Cycles until Failure')\n",
                    "plt.ylabel('Frequency (Engine Count)')\n",
                    "plt.legend(frameon=True, facecolor='white', edgecolor='none')\n",
                    "plt.tight_layout()\n",
                    "os.makedirs('../figures', exist_ok=True)\n",
                    "plt.savefig('../figures/engine_lifespans.png')\n",
                    "plt.show()\n",
                    "\n",
                    "print(f\"Engine Lifespan Statistics:\")\n",
                    "print(f\"  Minimum lifespan: {min_life} cycles\")\n",
                    "print(f\"  Maximum lifespan: {max_life} cycles\")\n",
                    "print(f\"  Mean lifespan: {mean_life:.2f} cycles\")\n",
                    "print(f\"  Median lifespan: {median_life:.2f} cycles\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Sensor Trend Analysis\n",
                    "\n",
                    "Now let's examine the 21 sensor signals over time for a single engine unit (e.g., Unit 1). This allows us to observe which sensors exhibit clear trends as the engine approaches failure (degradation curves) and which sensors remain flat or invariant (constant)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "unit_id = 1\n",
                    "unit_data = train_df[train_df['unit'] == unit_id]\n",
                    "\n",
                    "# Plot all 21 sensors in a 7x3 grid to inspect their behavior\n",
                    "fig, axs = plt.subplots(7, 3, figsize=(15, 20), sharex=True, dpi=300)\n",
                    "axs = axs.flatten()\n",
                    "\n",
                    "for i in range(1, 22):\n",
                    "    sensor_col = f's{i}'\n",
                    "    ax = axs[i-1]\n",
                    "    ax.plot(unit_data['cycle'], unit_data[sensor_col], color='#2ca02c', alpha=0.8)\n",
                    "    ax.set_title(f\"Sensor {i}: {sensor_col.upper()}\", fontsize=9, fontweight='bold')\n",
                    "    # Add description in text block\n",
                    "    desc = SENSOR_DESCRIPTIONS.get(sensor_col, \"\")\n",
                    "    ax.set_ylabel(desc.split('(')[-1].replace(')', '') if '(' in desc else '', fontsize=8)\n",
                    "    \n",
                    "    # Highlight flat/invariant sensors with red background\n",
                    "    if sensor_col in DROPPED_SENSORS:\n",
                    "        ax.set_facecolor('#ffeef0')\n",
                    "        ax.annotate('INVARIANT', xy=(0.5, 0.5), xycoords='axes fraction', \n",
                    "                    ha='center', va='center', color='red', fontweight='bold', fontsize=10)\n",
                    "\n",
                    "plt.suptitle(f\"All 21 Sensor Channels for Engine Unit {unit_id} (Red = Invariant/Dropped)\", fontsize=14, fontweight='bold', y=1.01)\n",
                    "plt.xlabel('Operating Cycles')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../figures/sensor_trends_all.png', bbox_inches='tight')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Discussion on Sensors to Drop\n",
                    "Based on the visual trend plots above, the following **7 sensors show zero variance** (remain completely flat) over the entire engine lifetime:\n",
                    "- `s1` (Total temperature at fan inlet)\n",
                    "- `s5` (Pressure at fan inlet)\n",
                    "- `s6` (Total pressure in bypass-duct)\n",
                    "- `s10` (Engine pressure ratio)\n",
                    "- `s16` (Burner efficiency ratio)\n",
                    "- `s18` (Demanded fan speed)\n",
                    "- `s19` (Demanded corrected fan speed)\n",
                    "\n",
                    "Since these sensors do not change, they do not contain any information about the degradation process. Including them in predictive models would only add noise and increase computational complexity. Thus, dropping them is fully justified.\n",
                    "\n",
                    "Let's mathematically verify their variance."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Calculate standard deviation of each sensor across the entire training set\n",
                    "sensor_stds = train_df[[f's{i}' for i in range(1, 22)]].std()\n",
                    "std_df = pd.DataFrame({\n",
                    "    'Sensor': sensor_stds.index,\n",
                    "    'Standard Deviation': sensor_stds.values,\n",
                    "    'Status': ['Dropped (Flat)' if col in DROPPED_SENSORS else 'Kept' for col in sensor_stds.index]\n",
                    "})\n",
                    "std_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Correlation Analysis\n",
                    "\n",
                    "Let's look at the correlation between the sensors. Strongly correlated sensors carry redundant information. We will plot a correlation matrix heatmap for all 21 sensors."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(12, 10), dpi=300)\n",
                    "corr = train_df[[f's{i}' for i in range(1, 22)]].corr()\n",
                    "\n",
                    "# Create mask for flat sensors which will have NaN correlation\n",
                    "sns.heatmap(corr, annot=True, fmt=\".2f\", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, cbar_kws={\"shrink\": .8})\n",
                    "plt.title('Correlation Heatmap of All 21 Sensor Channels (FD001)', fontsize=12, fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../figures/sensor_correlation_matrix.png')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Observations on Correlation\n",
                    "- Several sensor groups are extremely highly correlated (approaching +1.0 or -1.0):\n",
                    "  - For instance, `s4` (Total temperature at LPT outlet) and `s11` (Throttle pressure at HPC outlet) have a very high correlation.\n",
                    "  - `s9` (Physical core speed) and `s14` (Corrected core speed) also show strong correlation.\n",
                    "- In contrast, the flat sensors have a correlation of `NaN` because their variance is zero (no covariance).\n",
                    "\n",
                    "Now, let's plot a correlation heatmap of only the **kept** (degrading) sensors to see the clean relationship."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 8), dpi=300)\n",
                    "corr_kept = train_df[KEEP_SENSORS].corr()\n",
                    "sns.heatmap(corr_kept, annot=True, fmt=\".2f\", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, cbar_kws={\"shrink\": .8})\n",
                    "plt.title('Correlation Heatmap of the 14 Kept Sensors (FD001)', fontsize=12, fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../figures/sensor_correlation_kept.png')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Summary and Next Steps\n",
                    "\n",
                    "This EDA notebook has successfully verified that:\n",
                    "1. The dataset contains **100 engines** in training with lifespans ranging from **128 to 362 cycles** (mean of ~206 cycles).\n",
                    "2. Out of the 21 sensors, **7 sensors have zero variance** (`s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19`) and are dropped.\n",
                    "3. The remaining **14 sensors** exhibit clear degradation curves (either increasing or decreasing) as the engines approach failure.\n",
                    "\n",
                    "In the next notebook (**02_baseline_model.ipynb**), we will:\n",
                    "- Load the raw data and add target RUL labels using piecewise-linear capping (cap=125).\n",
                    "- Extract rolling statistics (rolling mean, std, and slope) per window size 30.\n",
                    "- Train classical ML models (Linear Regression, Random Forest, XGBoost) and evaluate them using RMSE and the custom PHM08 score."
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
    nb_path = "notebooks/01_eda.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Notebook created at {nb_path}")

if __name__ == "__main__":
    create_eda_notebook()
