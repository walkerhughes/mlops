import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

from mlops.settings.constants import DATA_PATH

# Resolve root path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Load params
with open(os.path.join(ROOT_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

# Params
data_path = os.path.join(ROOT_DIR, params["data"]["path"])
target_column = params["data"]["target"]
test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]

# Load and split
df = pd.read_csv(data_path)
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=True, random_state=random_state
)

# Preprocess features only
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Save arrays
processed_dir = os.path.join(ROOT_DIR, "mlops/data")
os.makedirs(processed_dir, exist_ok=True)

np.save(os.path.join(processed_dir, "X_train.npy"), X_train_processed)
np.save(os.path.join(processed_dir, "X_test.npy"), X_test_processed)
np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy())
np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy())