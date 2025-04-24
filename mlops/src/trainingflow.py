import itertools
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import mlflow
import numpy as np
import pandas as pd
from metaflow import FlowSpec, step, Parameter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlops.src.utils import (
    load_data,
    train_logistic_regression,
    evaluate_model,
    measure_inference_time
)
from mlops.settings.constants import DATA_PATH
from mlops.settings.constants import MLFLOW_SERVER_URI
from mlops.settings.constants import MLFLOW_LOCAL_URI

# Constants
DEFAULT_TRACKING_URI = MLFLOW_SERVER_URI # 'sqlite:///mlflow.db'
DEFAULT_EXPERIMENT_NAME = 'currency_classifier'
DEFAULT_RANDOM_SEED = 51
DEFAULT_SAVE_DIR = 'save_data'
DEFAULT_FILE_NAME = "BankNoteAuthentication.csv"
DEFAULT_TARGET_NAME = "class"
DEFAULT_SPLIT_SIZE = 1/3
DEFAULT_TEST_VAL_SIZE = 1/2


# Helper function to check if module is installed
def is_module_installed(module_name: str) -> bool:
    """Check if a Python module is installed."""
    import importlib.util
    return importlib.util.find_spec(module_name) is not None


def save_and_log_datasets(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame, 
    save_dir: str = DEFAULT_SAVE_DIR
) -> None:
    """Save datasets to parquet files and log them as artifacts in MLflow.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        save_dir: Directory to save the files
    """
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


def download_data(file_path: str, dataset_url: str) -> bool:
    """Download a dataset from a URL to the specified file path.
    
    Args:
        file_path: Local path where the file should be saved
        dataset_url: URL to download the dataset from
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        print(f"Downloading dataset from {dataset_url}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # For BankNote Authentication dataset, which is just a txt file without headers
        if 'data_banknote_authentication.txt' in dataset_url:
            # Download to a temporary file first
            temp_path = file_path + '.tmp'
            urllib.request.urlretrieve(dataset_url, temp_path)
            
            # Read the data and add headers for the BankNote dataset
            try:
                df = pd.read_csv(temp_path, header=None)
                # The BankNote dataset has 5 columns: 4 features and 1 class
                df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
                # Save with headers as CSV
                df.to_csv(file_path, index=False)
                # Remove temp file
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Failed to process downloaded file: {e}")
                # If processing fails, just use the original download
                os.rename(temp_path, file_path)
        else:
            # For other datasets, just download directly
            urllib.request.urlretrieve(dataset_url, file_path)
            
        print(f"Dataset downloaded successfully to {file_path}")
        return True
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return False


class ClassifierTrainFlow(FlowSpec):
    """Metaflow pipeline for training and evaluating classification models.

    This flow handles data loading, preprocessing, model training with hyperparameter 
    tuning, and evaluation. The results are tracked using MLflow.
    """
    # MLflow parameters
    mlflow_uri = Parameter(
        name="tracking_uri",
        default=DEFAULT_TRACKING_URI,
        help="URI for MLflow tracking server",
        type=str
    )
    mlflow_experiment = Parameter(
        name="experiment_name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Name of the MLflow experiment",
        type=str
    )
    
    # Data parameters
    random_seed = Parameter(
        name="random_seed",
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility",
        type=int
    )
    data_path = Parameter(
        name="data_path",
        default=DATA_PATH,
        help="Path to the data directory",
        type=str
    )
    file_name = Parameter(
        name="file_name",
        default=DEFAULT_FILE_NAME,
        help="Dataset name in data_path directory",
        type=str
    )
    target_name = Parameter(
        name="target_name",
        default=DEFAULT_TARGET_NAME,
        help="Name of the target feature column",
        type=str
    )
    
    # Train-test split parameters
    split_size = Parameter(
        name="split_size",
        default=DEFAULT_SPLIT_SIZE,
        help="Percentage of data used for test/eval data",
        type=float
    )
    test_val_size = Parameter(
        name="test_val_size",
        default=DEFAULT_TEST_VAL_SIZE,
        help="Percentage split between test/validation data",
        type=float
    )

    @step
    def start(self) -> None:
        """Start the flow by loading data."""
        # Setup MLflow - make sure to use the values, not the Parameter objects
        mlflow_uri = str(self.mlflow_uri)
        mlflow_experiment = str(self.mlflow_experiment)
        
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Check if experiment exists, create it if needed
        try:
            experiment = mlflow.get_experiment_by_name(mlflow_experiment)
            if experiment is None:
                mlflow.create_experiment(mlflow_experiment)
            mlflow.set_experiment(mlflow_experiment)
            print(f"Using MLflow experiment: {mlflow_experiment}")
        except Exception as e:
            print(f"Warning: MLflow experiment setup failed: {e}")
            print("Continuing without MLflow tracking...")
        
        # Check if data directory exists
        data_path = str(self.data_path)
        if not os.path.exists(data_path):
            print(f"WARNING: Data directory {data_path} doesn't exist")
            # Create the directory
            os.makedirs(data_path, exist_ok=True)
            print(f"Created directory: {data_path}")
        
        # Check if file exists
        file_path = os.path.join(data_path, str(self.file_name))
        if not os.path.exists(file_path):
            print(f"ERROR: Data file not found at {file_path}")
            print("Attempting to download the dataset...")
            
            # Define dataset URL for the BankNote Authentication dataset
            dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
            
            if download_data(file_path, dataset_url):
                print("Dataset downloaded successfully, continuing with training")
            else:
                raise FileNotFoundError(
                    f"Data file not found at {file_path} and download failed. "
                    f"Please manually download the dataset from {dataset_url} to {file_path}"
                )
        
        # Load data
        self.df = load_data(data_path, str(self.file_name))
        print(f"Data loaded successfully from {file_path}")
        print(f"Data shape: {self.df.shape}")
        print(f"Data columns: {self.df.columns.tolist()}")
        
        self.next(self.split_data)

    @step
    def split_data(self) -> None:
        """Split the dataset into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split
        
        # First split: training vs (validation+test)
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            self.df.drop(columns=[self.target_name]), 
            self.df[self.target_name], 
            test_size=self.split_size, 
            shuffle=True,
            random_state=self.random_seed
        )
        
        # Second split: validation vs test
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, 
            y_temp, 
            test_size=self.test_val_size, 
            shuffle=True,
            random_state=self.random_seed
        )
        
        print(f"Data split successfully into:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Validation set: {self.X_val.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        
        self.next(self.train_classifier)

    @step
    def train_classifier(self) -> None:
        """Train multiple models with different hyperparameters."""
        # Define hyperparameter grid for logistic regression
        param_grid: Dict[str, List[Any]] = {
            'C': [0.01, 1, 10],
            'penalty': ['l2'],  # note: 'l1' requires solver='liblinear'
            'solver': ['lbfgs']  # solver that supports l2 regularization
        }

        # Generate all combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Initialize tracking variables
        best_score = -float('inf')
        best_model = None
        best_run_id = None  # Initialize best_run_id variable
        all_runs = []

        # Set up MLflow tracking
        try:
            mlflow_uri = str(self.mlflow_uri)
            mlflow_experiment = str(self.mlflow_experiment)
            
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(mlflow_experiment)
            
            # Create parent run to group all hyperparameter tuning runs
            with mlflow.start_run(run_name="hyperparameter_tuning") as parent_run:
                mlflow.log_param("model_type", "logistic_regression")
                parent_run_id = parent_run.info.run_id
                
                # Iterate through hyperparameter combinations
                for i, params in enumerate(param_combinations):
                    run_metrics = self._train_and_evaluate_model(i, params)
                    all_runs.append(run_metrics)
                    
                    # Track best model
                    if run_metrics["val_accuracy"] > best_score:
                        best_score = run_metrics["val_accuracy"]
                        best_model = run_metrics["model"]
                        best_run_id = run_metrics["run_id"]
                
                # Log the best run information
                mlflow.log_metric("best_validation_accuracy", best_score)
                if best_run_id:  # Only log best_run_id if we have one
                    mlflow.log_param("best_run_id", best_run_id)
                else:
                    print("WARNING: No model performed well enough to be selected as best")
        except Exception as e:
            print(f"Warning: MLflow tracking failed: {e}")
            print("Continuing training without MLflow tracking...")
            
            # Still perform the training without MLflow
            for i, params in enumerate(param_combinations):
                # Train model 
                model = train_logistic_regression(self.X_train, self.y_train, params, self.random_seed)
                
                # Evaluate model
                val_acc = evaluate_model(model, self.X_val, self.y_val)
                print(f"Model {i+1}: params={params}, val_acc={val_acc:.4f}")
                
                # Track best model
                if val_acc > best_score:
                    best_score = val_acc
                    best_model = model
                
                # Store run info
                all_runs.append({
                    "run_id": f"local_run_{i}",
                    "params": params,
                    "train_accuracy": evaluate_model(model, self.X_train, self.y_train),
                    "val_accuracy": val_acc,
                    "test_accuracy": evaluate_model(model, self.X_test, self.y_test),
                    "inference_time": measure_inference_time(model, self.X_test),
                    "model": model
                })

        # Store results for the next steps
        self.model = best_model
        self.score = best_score
        self.all_runs = all_runs
        
        self.next(self.register_model)
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter dictionaries, each representing one combination
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = {name: value for name, value in zip(param_names, combo)}
            combinations.append(param_dict)
            
        return combinations
    
    def _train_and_evaluate_model(self, run_index: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model with specified parameters and evaluate it.
        
        Args:
            run_index: Index of this run for tracking purposes
            params: Dictionary of model hyperparameters
            
        Returns:
            Dictionary containing run metrics and the trained model
        """
        try:
            # Create child run for this specific hyperparameter combination
            with mlflow.start_run(run_name=f"run_{run_index}", nested=True) as child_run:
                # Log hyperparameters
                mlflow.log_params(params)
                
                # Train model
                model = train_logistic_regression(self.X_train, self.y_train, params, self.random_seed)
                
                # Evaluate model on all data splits
                train_acc = evaluate_model(model, self.X_train, self.y_train)
                val_acc = evaluate_model(model, self.X_val, self.y_val)
                test_acc = evaluate_model(model, self.X_test, self.y_test)
                
                # Log metrics
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("validation_accuracy", val_acc)
                mlflow.log_metric("test_accuracy", test_acc)
                
                # Measure and log inference time
                avg_inference_time = measure_inference_time(model, self.X_test)
                mlflow.log_metric("avg_inference_time", avg_inference_time)

                # Save model as artifact
                mlflow.sklearn.log_model(model, "model")
                
                # Save datasets as artifacts (only on first run to avoid duplication)
                if run_index == 0:
                    try:
                        save_and_log_datasets(self.X_train, self.X_val, self.X_test)
                    except Exception as e:
                        print(f"Warning: Failed to save and log datasets: {e}")
            
            # Return run information
            return {
                "run_id": child_run.info.run_id,
                "params": params,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "inference_time": avg_inference_time,
                "model": model
            }
        
        except Exception as e:
            print(f"Warning: MLflow run failed: {e}")
            print(f"Training model with params: {params} without MLflow tracking")
            
            # Train and evaluate model without MLflow
            model = train_logistic_regression(self.X_train, self.y_train, params, self.random_seed)
            train_acc = evaluate_model(model, self.X_train, self.y_train)
            val_acc = evaluate_model(model, self.X_val, self.y_val)
            test_acc = evaluate_model(model, self.X_test, self.y_test)
            avg_inference_time = measure_inference_time(model, self.X_test)
            
            return {
                "run_id": f"local_run_{run_index}",
                "params": params,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "inference_time": avg_inference_time,
                "model": model
            }

    @step
    def register_model(self) -> None:
        """Register the best model in MLflow model registry."""
        if not hasattr(self, 'model') or self.model is None:
            print("No best model to register.")
            self.next(self.end)
            return
            
        # Log comparison results
        print(f"Best validation accuracy: {self.score:.4f}")
        
        # Register the best model
        try:
            mlflow_uri = str(self.mlflow_uri)
            mlflow_experiment = str(self.mlflow_experiment)
            
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(mlflow_experiment)
            
        with mlflow.start_run():
                today = datetime.now().strftime("%Y-%m-%d")
                model_name = f"currency_classifier-{today}"
                
                mlflow.sklearn.log_model(
                    self.model, 
                    artifact_path='metaflow_model', 
                    registered_model_name=model_name
                )
                
                print(f"Registered model: {model_name}")
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Model registration failed: {e}")
            print("Continuing without MLflow model registration...")
            
            # Save the model locally instead
            try:
                import joblib
                os.makedirs('models', exist_ok=True)
                today = datetime.now().strftime("%Y-%m-%d")
                model_path = f"models/currency_classifier-{today}.joblib"
                joblib.dump(self.model, model_path)
                print(f"Model saved locally to: {model_path}")
            except Exception as e:
                print(f"Warning: Local model saving failed: {e}")
            
        self.next(self.end)

    @step
    def end(self) -> None:
        """End the flow by displaying results summary."""
        try:
            if not hasattr(self, 'score') or self.score is None:
                print("No models were successfully trained and evaluated.")
                return
                
            print('Best model evaluation:')
            print(f'Validation score: {self.score:.4f}')
            
            if not hasattr(self, 'all_runs') or not self.all_runs:
                print("No hyperparameter combinations were recorded.")
                return
            
            # Display results table
            self._display_results_table()
            
            if hasattr(self, 'model') and self.model is not None:
                print(f'\nBest model: {self.model}')
                
                # Print feature importances if available
                try:
                    if hasattr(self.model, 'coef_'):
                        if hasattr(self.X_train, 'columns'):
                            feature_names = self.X_train.columns
                            coefficients = self.model.coef_[0]
                            
                            # Create a list of (feature, coefficient) tuples and sort by absolute value
                            feature_importance = [(feature, coef) for feature, coef in zip(feature_names, coefficients)]
                            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            print("\nFeature importances:")
                            for feature, importance in feature_importance:
                                print(f"  {feature}: {importance:.6f}")
                except Exception as e:
                    print(f"Could not print feature importances: {e}")
            else:
                print("\nNo best model was selected.")
        except Exception as e:
            print(f"Error in end step: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_results_table(self) -> None:
        """Display a formatted table of all experiment results."""
        if not self.all_runs:
            print("No results to display")
            return
            
        if is_module_installed('tabulate'):
            # Create a summary table with tabulate
            try:
                import tabulate
                headers = ["Run", "C", "Penalty", "Solver", "Train Acc", "Val Acc", "Test Acc", "Inference Time (ms)"]
                
                # Create and sort table data
                table_data = []
                for i, run in enumerate(self.all_runs):
                    row = [
                        i, 
                        run["params"].get("C", "N/A"), 
                        run["params"].get("penalty", "N/A"),
                        run["params"].get("solver", "N/A"),
                        f"{run.get('train_accuracy', 0):.4f}",
                        f"{run.get('val_accuracy', 0):.4f}",
                        f"{run.get('test_accuracy', 0):.4f}",
                        f"{run.get('inference_time', 0)*1000:.4f}"
                    ]
                    table_data.append(row)
                
                # Sort by validation accuracy descending
                if table_data:
                    table_data.sort(key=lambda x: float(x[5].strip()) if x[5] != "N/A" else -1, reverse=True)
                
                print('\nAll hyperparameter combinations (sorted by validation accuracy):')
                print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
            except Exception as e:
                print(f"Failed to display table: {e}")
                self._display_simple_results()
        else:
            # Fallback to simple printing without tabulate
            self._display_simple_results()
    
    def _display_simple_results(self) -> None:
        """Display results in a simple text format."""
        print('\nAll hyperparameter combinations (sorted by validation accuracy):')
        
        # Sort by validation accuracy
        sorted_runs = sorted(
            self.all_runs, 
            key=lambda x: x.get('val_accuracy', 0), 
            reverse=True
        )
        
        for i, run in enumerate(sorted_runs):
            params = run["params"]
            print(
                f"Run {i}: "
                f"C={params.get('C', 'N/A')}, "
                f"penalty={params.get('penalty', 'N/A')}, "
                f"solver={params.get('solver', 'N/A')}, "
                f"train_acc={run.get('train_accuracy', 0):.4f}, "
                f"val_acc={run.get('val_accuracy', 0):.4f}, "
                f"test_acc={run.get('test_accuracy', 0):.4f}, "
                f"inf_time={run.get('inference_time', 0)*1000:.4f}ms"
            )


if __name__=='__main__':
    ClassifierTrainFlow()