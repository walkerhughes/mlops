
from metaflow import FlowSpec, step, pypi_base, Parameter, retry, timeout, catch, resources
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MLFLOW_SERVER_URI = ""

@pypi_base(packages={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'mlflow':'2.21.3'}, python='3.9.16')
class ClassifierScoringFlow(FlowSpec):
    model_uri = Parameter('model_uri', help='URI of the model to load from MLflow')
    data_url = Parameter('data_url', default='https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')

    @resources(cpu=1, memory=2000)
    @timeout(seconds=120)
    @retry(times=2)
    @catch(var='load_data_failure')
    @step
    def start(self):
        self.df = pd.read_csv(self.data_url)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        print("Data loaded and split into features and labels.")
        self.next(self.load_model)

    @resources(cpu=1, memory=2000)
    @timeout(seconds=120)
    @retry(times=2)
    @catch(var='load_model_failure')
    @step
    def load_model(self):
        self.model = mlflow.sklearn.load_model(self.model_uri)
        print(f"Model loaded from {self.model_uri}")
        self.next(self.score)

    @resources(cpu=1, memory=2000)
    @timeout(seconds=120)
    @retry(times=2)
    @catch(var='scoring_failure')
    @step
    def score(self):
        predictions = self.model.predict(self.X)
        self.accuracy = accuracy_score(self.y, predictions)
        print(f"Model accuracy on loaded data: {self.accuracy:.4f}")
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete.")

if __name__ == '__main__':
    ClassifierScoringFlow()
