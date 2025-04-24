from metaflow import FlowSpec, step, pypi_base

MLFLOW_SERVER_URI = "https://mlflow-api-662968008319.us-west2.run.app"

@pypi_base(packages={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'mlflow':'2.21.3'}, python='3.9.16')
class ClassifierTrainFlow(FlowSpec):

    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import numpy as np

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X,y, test_size=0.2, random_state=0)
        print("Data loaded successfully")
        self.lambdas = np.arange(0.001, 1, 0.3)
        self.next(self.train_lasso, foreach='lambdas')

    @step
    def train_lasso(self):
        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.input)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        # def score(inp):
        #     return inp.model, inp.model.score(inp.test_data, inp.test_labels)
        # self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        # self.model = self.results[0][0]
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
        mlflow.set_experiment('mlflow-metaflow-kubernetes-experiment')

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path = 'metaflow_train', registered_model_name="metaflow-wine-model")
            mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    ClassifierTrainFlow()