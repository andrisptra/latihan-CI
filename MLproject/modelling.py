import os
import random
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.joing(
            os.path.dirname(os.path.abspath(__file__)), "data/train_pca.csv"
        )
    )


# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("latihan credit_scoring")

data = pd.read_csv("data/train_pca.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    test_size=0.2,
    random_state=42,
)

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

input_example = X_train[0:5]

with mlflow.start_run():
    # n_estimator = 505
    # max_depth = 37
    # mlflow.autolog()

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    predicted_quality = model.predict(X_test)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
