import numpy as np
import pandas as pd
from metaflow import FlowSpec, step, Parameter, pypi_base

# Constants
MLFLOW_SERVER_URI = "https://mlflow-api-662968008319.us-west2.run.app"
DEFAULT_TRACKING_URI = MLFLOW_SERVER_URI # 'sqlite:///mlflow.db'
DEFAULT_EXPERIMENT_NAME = 'currency_classifier'
DEFAULT_RANDOM_SEED = 51
DEFAULT_SAVE_DIR = 'save_data'
DEFAULT_FILE_NAME = "BankNoteAuthentication.csv"
DEFAULT_TARGET_NAME = "class"
DEFAULT_SPLIT_SIZE = 1/3
DEFAULT_TEST_VAL_SIZE = 1/2


@pypi_base(packages={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'mlflow':'2.21.3'}, python='3.9.16')
class ClassifierTrainFlow(FlowSpec):
    """
    Metaflow pipeline for training and evaluating classification models on GCP with Kubernetes.

    This flow handles data loading, preprocessing, model training with hyperparameter 
    tuning, and evaluation. The results are tracked using MLflow.
    """
    # MLflow parameters
    mlflow_uri = Parameter(
        name="tracking_uri",
        default=MLFLOW_SERVER_URI,
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
        import mlflow

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
        
        # Load data
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"        
        self.df = pd.read_csv(url)
        print(f"Data loaded successfully")
        self.next(self.split_data)

    @step
    def split_data(self) -> None:
        from sklearn.model_selection import train_test_split
        
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.df.iloc[:, :-1],  # All columns except last
            self.df.iloc[:, -1],   # Last column
            test_size=self.split_size, 
            shuffle=True,
            random_state=self.random_seed
        )
        
        print(f"Data split successfully into:")
        print(f"  Training set: {self.train_data.shape[0]} samples")
        print(f"  Test set: {self.test_data.shape[0]} samples")
        self.lambdas = np.arange(0.001, 1, 0.3)
        self.next(self.train_lasso, foreach='lambdas')

    @step
    def train_lasso(self):
        import mlflow
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        # Setup MLflow for this run
        mlflow.set_tracking_uri(str(self.mlflow_uri))
        mlflow.set_experiment(str(self.mlflow_experiment))
        alpha = self.input
        with mlflow.start_run():
            mlflow.log_param("alpha", alpha)
            self.model = LogisticRegression(penalty="l2", C=alpha)
            self.model.fit(self.train_data, self.train_labels)
            # Validation performance
            test_preds = self.model.predict(self.test_data)
            # test_preds_binary = (test_preds >= 0.5).astype(int)
            test_acc = accuracy_score(self.test_labels, test_preds)
            mlflow.log_metric("test_accuracy", test_acc)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):

        import mlflow
        mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
        mlflow.set_experiment('mlflow-metaflow-kubernetes-experiment')

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path='metaflow_train', registered_model_name="metaflow-currency-model-k8s")
            mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__ == '__main__':
    ClassifierTrainFlow()
