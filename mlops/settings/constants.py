import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data")
MODELS_PATH = os.path.join(ROOT_DIR, "models")
PREDICTIONS_PATH = os.path.join(ROOT_DIR, "predictions")

# MLflow settings
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "currency_classifier"

# Model settings
DEFAULT_MODEL_NAME = "currency_classifier"
DEFAULT_MODEL_STAGE = "Production"

MLFLOW_SERVER_URI = "https://mlflow-api-662968008319.us-west2.run.app"
MLFLOW_LOCAL_URI = "'sqlite:///mlflow.db'"
