import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from metaflow import FlowSpec, step, Parameter

from mlops.src.utils import (
    load_data,
    evaluate_model
)
from mlops.settings.constants import (
    DATA_PATH, 
    MLFLOW_TRACKING_URI, 
    MLFLOW_EXPERIMENT_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_STAGE,
    PREDICTIONS_PATH
)


class ModelScoringFlow(FlowSpec):
    """MetaFlow pipeline for scoring data using a pre-trained model.
    
    This flow handles data loading, preprocessing, model loading, 
    prediction, and result saving. It can use models registered in MLflow or
    models saved locally.
    """
    # MLflow parameters
    mlflow_uri = Parameter(
        name="tracking_uri",
        default=MLFLOW_TRACKING_URI,
        help="URI for MLflow tracking server",
        type=str
    )
    mlflow_experiment = Parameter(
        name="experiment_name",
        default=MLFLOW_EXPERIMENT_NAME,
        help="Name of the MLflow experiment",
        type=str
    )
    model_name = Parameter(
        name="model_name",
        default=DEFAULT_MODEL_NAME,
        help="Name of the registered model in MLflow",
        type=str
    )
    model_stage = Parameter(
        name="model_stage",
        default=DEFAULT_MODEL_STAGE,
        help="Stage of the model to use (Production, Staging, etc.)",
        type=str
    )
    model_version = Parameter(
        name="model_version",
        default=None,
        help="Specific version of the model to use (overrides stage)",
        type=str
    )
    
    # Data parameters
    data_path = Parameter(
        name="data_path",
        default=DATA_PATH,
        help="Path to the data directory",
        type=str
    )
    file_name = Parameter(
        name="file_name",
        default="BankNoteAuthentication.csv",
        help="Dataset name in data_path directory",
        type=str
    )
    target_name = Parameter(
        name="target_name",
        default="class",
        help="Name of the target feature column",
        type=str
    )
    use_holdout = Parameter(
        name="use_holdout",
        default=True,
        help="Whether to use a holdout set from the original data",
        type=bool
    )
    split_ratio = Parameter(
        name="split_ratio",
        default=0.8,
        help="Train/test split ratio when using holdout (train size)",
        type=float
    )
    random_seed = Parameter(
        name="random_seed",
        default=42,
        help="Random seed for reproducibility",
        type=int
    )
    
    # Output parameters
    output_dir = Parameter(
        name="output_dir",
        default=PREDICTIONS_PATH,
        help="Directory to save predictions",
        type=str
    )
    save_format = Parameter(
        name="save_format",
        default="csv",
        help="Format to save predictions (csv, json, parquet)",
        type=str
    )

    @step
    def start(self) -> None:
        """Start the flow by loading and preparing data."""
        # Setup MLflow
        mlflow_uri = str(self.mlflow_uri)
        mlflow_experiment = str(self.mlflow_experiment)
        
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            experiment = mlflow.get_experiment_by_name(mlflow_experiment)
            if experiment is None:
                print(f"Creating new experiment: {mlflow_experiment}")
                mlflow.create_experiment(mlflow_experiment)
            mlflow.set_experiment(mlflow_experiment)
            print(f"Using MLflow experiment: {mlflow_experiment}")
            self.mlflow_ready = True
        except Exception as e:
            print(f"Warning: MLflow setup failed: {e}")
            print("Continuing without MLflow tracking...")
            self.mlflow_ready = False
        
        # Load data for scoring
        self.next(self.load_data)
    
    @step
    def load_data(self) -> None:
        """Load data for scoring."""
        data_path = str(self.data_path)
        file_name = str(self.file_name)
        
        # Check if data file exists
        full_path = os.path.join(data_path, file_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found at {full_path}")
        
        # Load full dataset
        self.df = load_data(data_path, file_name)
        print(f"Data loaded successfully: {self.df.shape}")
        print(f"Data columns: {self.df.columns.tolist()}")
        
        # If using holdout, proceed to split data
        # Otherwise use the whole dataset for scoring
        if self.use_holdout:
            self.next(self.prepare_data_split)
        else:
            self.scoring_data = self.df.copy()
            self.next(self.load_model)
    
    @step
    def prepare_data_split(self) -> None:
        """Prepare data either by splitting or using entire dataset."""
        from sklearn.model_selection import train_test_split
        
        target_name = str(self.target_name)
        
        # Check if target exists in the data
        if target_name not in self.df.columns:
            raise ValueError(f"Target column '{target_name}' not found in the data")
        
        # Split the data into train and scoring sets
        train_data, scoring_data = train_test_split(
            self.df, 
            test_size=1.0-float(self.split_ratio),
            random_state=self.random_seed
        )
        
        # Store only the scoring data for later
        self.scoring_data = scoring_data
        print(f"Created holdout scoring data with shape: {self.scoring_data.shape}")
        
        self.next(self.load_model)
    
    @step
    def load_model(self) -> None:
        """Load the model from MLflow model registry or local files."""
        # Prepare model identification parameters
        model_name = str(self.model_name)
        model_stage = str(self.model_stage)
        model_version = self.model_version
        
        # Try loading from MLflow first
        if self.mlflow_ready:
            try:
                print(f"Attempting to load model '{model_name}' from MLflow...")
                
                # Determine the model URI based on stage or version
                if model_version:
                    model_uri = f"models:/{model_name}/{model_version}"
                    print(f"Loading model version: {model_version}")
                else:
                    model_uri = f"models:/{model_name}/{model_stage}"
                    print(f"Loading model stage: {model_stage}")
                
                # Load the model
                self.model = mlflow.sklearn.load_model(model_uri)
                print(f"Model loaded successfully from MLflow")
                self.model_source = "mlflow"
                
                # Store model metadata if available
                try:
                    model_info = mlflow.models.get_model_info(model_uri)
                    self.model_info = {
                        "run_id": getattr(model_info, "run_id", None),
                        "model_uri": model_uri,
                        "flavor": getattr(model_info, "flavors", {}).get("python_function", {}).get("loader_module", "unknown")
                    }
                except:
                    self.model_info = {"model_uri": model_uri}
                
                self.next(self.preprocess_data)
                return
            except Exception as e:
                print(f"Failed to load model from MLflow: {e}")
                print("Attempting to load local model...")
        
        # If MLflow loading fails, try to load locally saved model
        try:
            import joblib
            from glob import glob
            
            # Look for models in the models directory
            model_files = glob(f"models/{model_name}*.joblib")
            
            if not model_files:
                raise FileNotFoundError(f"No local models found matching: {model_name}")
            
            # Use the most recent model file
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Loading most recent local model: {latest_model}")
            
            self.model = joblib.load(latest_model)
            print(f"Model loaded successfully from local file")
            self.model_source = "local"
            self.model_info = {"model_path": latest_model}
            
        except Exception as e:
            print(f"Failed to load local model: {e}")
            raise RuntimeError("Could not load model from either MLflow or local files")
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self) -> None:
        """Preprocess the data for scoring."""
        target_name = str(self.target_name)
        
        # Store the target if available for evaluation
        if target_name in self.scoring_data.columns:
            self.y_true = self.scoring_data[target_name].copy()
            # Remove target from features
            self.X_score = self.scoring_data.drop(columns=[target_name])
            print(f"Target column '{target_name}' found in scoring data")
            self.has_target = True
        else:
            # If no target column, use all columns as features
            self.X_score = self.scoring_data.copy()
            self.has_target = False
            print(f"No target column found in scoring data, using all features")
        
        # Print feature information
        print(f"Scoring data shape: {self.X_score.shape}")
        print(f"Features: {list(self.X_score.columns)}")
        
        self.next(self.generate_predictions)
    
    @step
    def generate_predictions(self) -> None:
        """Generate predictions using the loaded model."""
        # Make predictions
        print(f"Generating predictions with model...")
        
        try:
            start_time = datetime.now()
            
            # Generate probability predictions if the model supports it
            self.y_pred_proba = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    self.y_pred_proba = self.model.predict_proba(self.X_score)
                    print(f"Probability predictions generated with shape: {self.y_pred_proba.shape}")
                except Exception as e:
                    print(f"Failed to generate probability predictions: {e}")
            
            # Generate class predictions
            self.y_pred = self.model.predict(self.X_score)
            print(f"Class predictions generated with shape: {self.y_pred.shape}")
            
            # Calculate prediction time
            end_time = datetime.now()
            self.prediction_time = (end_time - start_time).total_seconds()
            print(f"Prediction completed in {self.prediction_time:.4f} seconds")
            
            # Create a dataframe with predictions
            self.predictions_df = pd.DataFrame()
            
            # Add record identifiers
            if hasattr(self.X_score, 'index') and not isinstance(self.X_score.index, pd.RangeIndex):
                self.predictions_df['id'] = self.X_score.index
            else:
                self.predictions_df['id'] = range(len(self.y_pred))
            
            # Add predictions
            self.predictions_df['prediction'] = self.y_pred
            
            # Add probabilities if available
            if self.y_pred_proba is not None:
                # For binary classification
                if self.y_pred_proba.shape[1] == 2:
                    self.predictions_df['probability'] = self.y_pred_proba[:, 1]
                # For multi-class, add each class probability
                else:
                    for i in range(self.y_pred_proba.shape[1]):
                        self.predictions_df[f'probability_class_{i}'] = self.y_pred_proba[:, i]
            
            # If we have ground truth, evaluate and save results
            if self.has_target:
                self.next(self.evaluate_predictions)
            else:
                self.next(self.save_predictions)
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate predictions: {e}")
    
    @step
    def evaluate_predictions(self) -> None:
        """Evaluate model performance if ground truth is available."""
        print("Evaluating model performance on scoring data...")
        
        try:
            # Calculate evaluation metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                confusion_matrix, classification_report, roc_auc_score
            )
            
            # Basic classification metrics
            accuracy = accuracy_score(self.y_true, self.y_pred)
            
            # Initialize metrics dict
            metrics = {
                'accuracy': accuracy,
            }
            
            try:
                # These might fail depending on the nature of the predictions
                metrics['precision'] = precision_score(self.y_true, self.y_pred, average='weighted')
                metrics['recall'] = recall_score(self.y_true, self.y_pred, average='weighted')
                metrics['f1'] = f1_score(self.y_true, self.y_pred, average='weighted')
                
                # Add AUC if probability predictions are available for binary classification
                if self.y_pred_proba is not None and len(np.unique(self.y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba[:, 1])
            except Exception as e:
                print(f"Warning: Some metrics couldn't be calculated: {e}")
            
            # Print metrics
            print("\nModel Performance Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            
            # Print confusion matrix
            cm = confusion_matrix(self.y_true, self.y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(self.y_true, self.y_pred))
            
            # Store metrics for later
            self.metrics = metrics
            
            # Add ground truth to predictions dataframe
            self.predictions_df['true_value'] = self.y_true.values
            
            # Log metrics to MLflow
            if self.mlflow_ready:
                try:
                    with mlflow.start_run():
                        # Log all metrics
                        for name, value in metrics.items():
                            mlflow.log_metric(f"scoring_{name}", value)
                        
                        # Log confusion matrix as a figure
                        try:
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            
                            plt.figure(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            
                            # Save the figure and log it
                            cm_path = 'confusion_matrix.png'
                            plt.savefig(cm_path)
                            mlflow.log_artifact(cm_path)
                            plt.close()
                        except Exception as e:
                            print(f"Failed to log confusion matrix figure: {e}")
                except Exception as e:
                    print(f"Failed to log metrics to MLflow: {e}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        # Transition to the next step
        self.next(self.save_predictions)
    
    @step
    def save_predictions(self) -> None:
        """Save predictions to the specified output format."""
        output_dir = str(self.output_dir)
        save_format = str(self.save_format).lower()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"predictions_{timestamp}"
        
        # Save predictions in the specified format
        try:
            if save_format == 'csv':
                output_path = os.path.join(output_dir, f"{base_filename}.csv")
                self.predictions_df.to_csv(output_path, index=False)
            elif save_format == 'json':
                output_path = os.path.join(output_dir, f"{base_filename}.json")
                self.predictions_df.to_json(output_path, orient='records')
            elif save_format == 'parquet':
                output_path = os.path.join(output_dir, f"{base_filename}.parquet")
                self.predictions_df.to_parquet(output_path, index=False)
            else:
                print(f"Unknown save format: {save_format}, defaulting to CSV")
                output_path = os.path.join(output_dir, f"{base_filename}.csv")
                self.predictions_df.to_csv(output_path, index=False)
            
            print(f"Predictions saved to: {output_path}")
            self.output_path = output_path
            
            # Also save metrics if they exist
            if hasattr(self, 'metrics'):
                metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
                with open(metrics_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
                print(f"Metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"Error saving predictions: {e}")
            import traceback
            traceback.print_exc()
        
        self.next(self.end)
    
    @step
    def end(self) -> None:
        """End the flow with a summary of results."""
        print("\n" + "="*60)
        print("Scoring Flow Complete")
        print("="*60)
        
        print(f"\nScored {len(self.predictions_df)} records")
        
        if hasattr(self, 'model'):
            print(f"Model source: {self.model_source}")
            if hasattr(self, 'model_info'):
                for key, value in self.model_info.items():
                    print(f"  {key}: {value}")
        
        if hasattr(self, 'prediction_time'):
            print(f"Prediction time: {self.prediction_time:.4f} seconds")
            
        if hasattr(self, 'metrics'):
            print("\nPerformance Metrics:")
            for metric_name, value in self.metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        if hasattr(self, 'output_path'):
            print(f"\nPredictions saved to: {self.output_path}")
        
        print("\nScoring flow completed successfully!")


if __name__ == '__main__':
    ModelScoringFlow() 