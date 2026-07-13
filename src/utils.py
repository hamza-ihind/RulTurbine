import os
import numpy as np
import pandas as pd

# Define C-MAPSS dataset column names
COLUMNS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

# Sensors identified as flat/invariant (uninformative)
DROPPED_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']

# Active sensors that show degradation trends
KEEP_SENSORS = [f's{i}' for i in range(1, 22) if f's{i}' not in DROPPED_SENSORS]

# Sensor descriptions mapping for plotting
SENSOR_DESCRIPTIONS = {
    's1': 'Total temperature at fan inlet (°R)',
    's2': 'Total temperature at LPC outlet (°R)',
    's3': 'Total temperature at HPC outlet (°R)',
    's4': 'Total temperature at LPT outlet (°R)',
    's5': 'Pressure at fan inlet (psia)',
    's6': 'Total pressure in bypass-duct (psia)',
    's7': 'Total pressure at HPC outlet (psia)',
    's8': 'Physical fan speed (rpm)',
    's9': 'Physical core speed (rpm)',
    's10': 'Engine pressure ratio (P15/P2)',
    's11': 'Throttle pressure at HPC outlet (psia)',
    's12': 'Ratio of fuel flow to Ps30 (pps/psi)',
    's13': 'Corrected fan speed (rpm)',
    's14': 'Corrected core speed (rpm)',
    's15': 'Bypass ratio',
    's16': 'Burner efficiency ratio',
    's17': 'Bleed enthalpy',
    's18': 'Demanded fan speed (rpm)',
    's19': 'Demanded corrected fan speed (rpm)',
    's20': 'HPT coolant bleed (lbm/s)',
    's21': 'LPT coolant bleed (lbm/s)'
}

def load_raw_data(subset='FD001', raw_dir='data/raw'):
    """
    Loads raw C-MAPSS text files into pandas DataFrames.
    """
    # Use workspace relative path if raw_dir is relative
    if not os.path.isabs(raw_dir):
        raw_dir = os.path.abspath(raw_dir)
        
    train_path = os.path.join(raw_dir, f'train_{subset}.txt')
    test_path = os.path.join(raw_dir, f'test_{subset}.txt')
    rul_path = os.path.join(raw_dir, f'RUL_{subset}.txt')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Raw data files for {subset} not found in {raw_dir}.")
        
    # Read text files (space-separated, no header)
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLUMNS)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMNS)
    
    # RUL file has one entry per engine unit in the test set
    if os.path.exists(rul_path):
        test_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
        # Add a unit column (1-indexed)
        test_rul['unit'] = test_rul.index + 1
    else:
        test_rul = None
        
    return train_df, test_df, test_rul

def compute_phm08_score(y_true, y_pred):
    """
    Computes the PHM08 asymmetric scoring metric.
    Penalizes overestimation (late predictions) more than underestimation (early predictions).
    
    Parameters:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    d = y_pred - y_true
    
    # Asymmetric score terms
    # s_i = exp(-d_i/13) - 1 for d_i < 0 (early prediction)
    # s_i = exp(d_i/10) - 1 for d_i >= 0 (late prediction)
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return np.sum(scores)

def get_piecewise_rul(df, cap=125):
    """
    Adds RUL targets to the dataframe using piecewise-linear capping.
    For training engines, RUL degrades linearly until it hits a cap.
    """
    # Calculate max cycle for each engine unit
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    rul = max_cycle - df['cycle']
    if cap is not None:
        rul = np.clip(rul, 0, cap)
    df_copy = df.copy()
    df_copy['RUL'] = rul
    return df_copy

def preprocess_data_and_save_npy(subset='FD001', raw_dir='data/raw', processed_dir='data/processed', window_size=30, cap=125):
    """
    Performs full preprocessing:
      - Capping of RUL (piecewise-linear)
      - Min-max scaling of the 14 degrading sensors
      - Sliding window sequence creation
      - Saves outputs to .npy files
    """
    from sklearn.preprocessing import MinMaxScaler
    
    train_df, test_df, test_rul_df = load_raw_data(subset, raw_dir)
    
    # Calculate piecewise RUL for training data
    train_df = get_piecewise_rul(train_df, cap=cap)
    
    # Scale features using training set parameters
    scaler = MinMaxScaler()
    train_df[KEEP_SENSORS] = scaler.fit_transform(train_df[KEEP_SENSORS])
    test_df[KEEP_SENSORS] = scaler.transform(test_df[KEEP_SENSORS])
    
    # Create sliding windows for training data
    X_train, y_train = [], []
    for unit, group in train_df.groupby('unit'):
        group_features = group[KEEP_SENSORS].values
        group_rul = group['RUL'].values
        n_samples = len(group)
        for i in range(n_samples - window_size + 1):
            X_train.append(group_features[i:i+window_size])
            y_train.append(group_rul[i+window_size-1])
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    # Create windows for test data (take the last window of size 30 for each unit)
    X_test, y_test = [], []
    for unit, group in test_df.groupby('unit'):
        if len(group) >= window_size:
            window = group[KEEP_SENSORS].values[-window_size:]
            X_test.append(window)
            
            # Retrieve ground truth RUL for the final cycle
            true_rul = test_rul_df.loc[test_rul_df['unit'] == unit, 'RUL'].values[0]
            y_test.append(true_rul)
        else:
            raise ValueError(f"Test engine unit {unit} has length {len(group)}, less than window_size {window_size}.")
            
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    # Save as .npy files
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    
    print(f"Processed dataset saved successfully to {processed_dir}:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def extract_rolling_features(X):
    """
    Computes rolling features (mean, standard deviation, and trend/slope) over the window dimension.
    Input shape: (num_samples, window_size, num_features)
    Output shape: (num_samples, 3 * num_features)
    """
    N, W, F = X.shape
    
    # 1. Rolling Mean
    means = np.mean(X, axis=1) # (N, F)
    
    # 2. Rolling Standard Deviation
    stds = np.std(X, axis=1) # (N, F)
    
    # 3. Rolling Slope (Trend line slope within the window)
    weights = np.arange(W) - (W - 1) / 2.0
    denom = np.sum(weights ** 2)
    slopes = np.sum(X * weights.reshape(1, W, 1), axis=1) / denom # (N, F)
    
    # Concatenate features
    features = np.concatenate([means, stds, slopes], axis=1) # (N, 3 * F)
    return features

