import os
import time
from typing import Optional, Union, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from mlops.settings.constants import DATA_PATH


def load_data(data_path: str, file_name: str) -> pd.DataFrame:
    """Load data from a CSV or parquet file.
    
    Args:
        data_path: Directory containing the data file
        file_name: Name of the data file
        
    Returns:
        DataFrame containing the loaded data
    """
    file_path = os.path.join(data_path, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    # Determine file type from extension
    file_ext = os.path.splitext(file_name)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_ext in ['.txt', '.data']:
        # Try to load as CSV with different delimiters
        try:
            df = pd.read_csv(file_path, sep=',')
        except:
            try:
                df = pd.read_csv(file_path, sep='\t')
            except:
                df = pd.read_csv(file_path, sep=None, engine='python')
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")
    
    return df


def split_dataset(df: pd.DataFrame, 
                 target_name: str, 
                 split_size: float, 
                 test_val_size: float, 
                 random_seed: Optional[int] = None) -> Tuple:
    """Split dataset into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        target_name: Name of the target column
        split_size: Proportion of data for test/val
        test_val_size: How to split the test/val data
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_name]), df[target_name], 
        test_size=split_size, shuffle=True, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_val_size, shuffle=True, random_state=random_seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_logistic_regression(X_train: pd.DataFrame, 
                             y_train: pd.Series, 
                             params: Dict[str, Any], 
                             random_seed: Optional[int] = None) -> LogisticRegression:
    """Train a logistic regression model with given parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model parameters
        random_seed: Random seed for reproducibility
        
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(
        **params,
        random_state=random_seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_svm_model(X_train: pd.DataFrame, 
                   y_train: pd.Series, 
                   params: Dict[str, Any], 
                   random_seed: Optional[int] = None) -> Any:
    """Train an SVM model with given parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model parameters
        random_seed: Random seed for reproducibility
        
    Returns:
        Trained SVM model
    """
    from sklearn import svm
    model = svm.SVC(
        **params,
        random_state=random_seed
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Evaluate model and return accuracy.
    
    Args:
        model: Trained model with predict method
        X: Feature data
        y: Ground truth labels
        
    Returns:
        Accuracy score
    """
    preds = model.predict(X)
    return accuracy_score(y, preds)


def measure_inference_time(model: Any, 
                          X: pd.DataFrame, 
                          num_iterations: int = 50) -> float:
    """Measure average inference time per sample.
    
    Args:
        model: Trained model with predict method
        X: Feature data
        num_iterations: Number of prediction iterations to average
        
    Returns:
        Average inference time per sample in seconds
    """
    durations = []
    for _ in range(num_iterations):
        start = time.time()
        _ = model.predict(X)
        durations.append(time.time() - start)
    
    return np.mean(durations) / X.shape[0]


def save_model(model: Any, 
              model_path: str, 
              metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save a model to disk.
    
    Args:
        model: Trained model to save
        model_path: Path where model should be saved
        metadata: Optional metadata to save with model
    """
    import joblib
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def save_and_log_datasets(X_train: pd.DataFrame, 
                         X_val: pd.DataFrame, 
                         X_test: pd.DataFrame, 
                         save_dir: str = 'save_data') -> None:
    """Save datasets to parquet files and log them as artifacts in MLflow.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        save_dir: Directory to save the files
    """
    import mlflow
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save datasets
    train_path = f'{save_dir}/x_train.parquet'
    val_path = f'{save_dir}/x_val.parquet'
    test_path = f'{save_dir}/x_test.parquet'
    
    X_train.to_parquet(train_path)
    X_val.to_parquet(val_path)
    X_test.to_parquet(test_path)
    
    # Log artifacts
    mlflow.log_artifact(train_path)
    mlflow.log_artifact(val_path)
    mlflow.log_artifact(test_path)