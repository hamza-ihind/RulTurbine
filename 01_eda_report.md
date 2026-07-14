# Exploratory Data Analysis (EDA) Report: NASA C-MAPSS Dataset (FD001)

This report provides a comprehensive explanation of the exploratory data analysis carried out in **01_eda.ipynb**. It covers data loading, engine lifespan analysis, sensor trend analysis, and correlation analysis, including interpretations and key code blocks.

---

## 1. Dataset Dimensions & Loading

The C-MAPSS FD001 dataset consists of space-separated text files without headers. It contains simulated run-to-failure operational profiles of turbofan engines under a single operating condition.

### Key Code Block (Data Loading)
```python
import pandas as pd

def load_raw_data(subset='FD001', raw_dir='data/raw'):
    columns = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv(f"{raw_dir}/train_{subset}.txt", sep=r"\s+", names=columns)
    test_df = pd.read_csv(f"{raw_dir}/test_{subset}.txt", sep=r"\s+", names=columns)
    test_rul = pd.read_csv(f"{raw_dir}/RUL_{subset}.txt", sep=r"\s+", names=['RUL'])
    test_rul.insert(0, 'unit', range(1, len(test_rul) + 1))
    return train_df, test_df, test_rul
```

### Interpretation
- **Training Set**: Contains **100 engine units** run continuously from normal startup until failure.
- **Test Set**: Contains **100 engine units** whose records are truncated at some cycle prior to failure. The goal is to predict the Remaining Useful Life (RUL) of these engines at the truncation points.

---

## 2. Engine Lifespan Distribution

Analyzing the maximum cycle recorded for each engine in the training set gives the distribution of engine lifespans prior to failure.

### Key Code Block (Lifespan Calculation)
```python
lifespans = train_df.groupby('unit')['cycle'].max()
mean_life = lifespans.mean()
median_life = lifespans.median()
min_life = lifespans.min()
max_life = lifespans.max()
```

### Summary of Engine Lifespans (Training Set)

| Metric | Cycles |
| :--- | :---: |
| **Minimum Lifespan** | 128 |
| **Maximum Lifespan** | 362 |
| **Mean Lifespan** | 206.31 |
| **Median Lifespan** | 199.00 |

### Placeholder for Image
`[Image: Engine Lifespan Distribution Histogram]`

### Interpretation
- Engines run between **128 and 362 cycles** before failing, with an average lifespan of **~206 cycles**.
- The wide range indicates significant variance in degradation rates between different engine units. A fixed cycle-based maintenance schedule would be highly inefficient (either performing maintenance too early or failing mid-operation), highlighting the need for data-driven predictive maintenance (RUL prediction).

---

## 3. Sensor Trend & Flat Sensor Analysis

Turbofan engines record 21 sensor measurements per cycle. Analyzing standard deviation (variance) identifies which sensors contain diagnostic information and which are invariant.

### Key Code Block (Standard Deviation Analysis)
```python
sensor_stds = train_df[[f's{i}' for i in range(1, 22)]].std()
```

### Summary of Sensor Variations (FD001)

| Sensor Channel | Standard Deviation (std) | Status | What it Measures |
| :--- | :---: | :---: | :--- |
| **s1** | 0.0000 | **Dropped** | Total temperature at fan inlet |
| **s2** | 0.5000 | Kept | Total temperature at LPC outlet |
| **s3** | 4.6201 | Kept | Total temperature at HPC outlet |
| **s4** | 19.0715 | Kept | Total temperature at LPT outlet |
| **s5** | 0.0000 | **Dropped** | Pressure at fan inlet |
| **s6** | 0.0000 | **Dropped** | Total pressure in bypass-duct |
| **s7** | 1.0009 | Kept | Total pressure at HPC outlet |
| **s8** | 0.0748 | Kept | Physical fan speed |
| **s9** | 22.0898 | Kept | Physical core speed |
| **s10** | 0.0000 | **Dropped** | Engine pressure ratio |
| **s11** | 0.2675 | Kept | Throttle pressure at HPC outlet |
| **s12** | 0.7375 | Kept | Ratio of fuel flow to Ps30 |
| **s13** | 0.0714 | Kept | Corrected fan speed |
| **s14** | 19.0761 | Kept | Corrected core speed |
| **s15** | 0.1553 | Kept | Bypass ratio |
| **s16** | 0.0000 | **Dropped** | Burner efficiency ratio |
| **s17** | 1.2486 | Kept | LPT speed |
| **s18** | 0.0000 | **Dropped** | Demanded fan speed |
| **s19** | 0.0000 | **Dropped** | Demanded corrected fan speed |
| **s20** | 0.1704 | Kept | HPT coolant flow |
| **s21** | 0.1082 | Kept | LPT coolant flow |

### Interpretation
- **Invariant Sensors**: 7 sensors (`s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19`) show **zero variance** ($std = 0$). Since their values remain constant throughout the operational lifespan of all engines, they carry no diagnostics on component wear. Dropping them prevents noise introduction and reduces computing overhead.
- **Active Sensors**: The remaining 14 sensors show clear degradation trajectories. Some increase over time (e.g. temperatures `s3`, `s4`, and fuel flow ratio `s12`), while others decrease (e.g. fan speeds `s8` and corrected fan speed `s13`). These trends correspond directly to component wear (loss of efficiency leading to higher temperatures/pressures to maintain thrust).
- **Visualization**: Due to the volume of trend graphs generated (21 separate plots), the individual sensor trend figures are provided in the **Annexes (Annex A: Sensor Trend Visualizations)**.

---

## 4. Kept Sensors Correlation Analysis

A correlation analysis of the 14 kept sensors shows the relationships and redundancies between diagnostic channels.

### Key Code Block (Correlation Heatmap)
```python
corr_kept = train_df[KEEP_SENSORS].corr()
```

### Placeholder for Image
`[Image: Kept Sensors Correlation Heatmap]`

### Interpretation
- **Strong Positive Correlations**: High correlations ($r \approx 0.96$) are observed between temperatures `s4` and throttle pressure `s11`, and between corrected core speed `s14` and physical core speed `s9`. This indicates they capture identical degradation characteristics.
- **Strong Negative Correlations**: Highly negative correlations ($r \approx -0.92$) are visible between physical core speed `s9` and ratio of fuel flow to Ps30 `s12`.
- **Significance for Modeling**: The strong correlation groups suggest high redundancy. Ensemble tree models (Random Forest, XGBoost) and sequence models (LSTM) are selected because they are robust to multicollinearity and automatically weigh features based on predictive value rather than raw redundancy.
