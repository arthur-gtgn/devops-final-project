import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub

experiment_name = "Musshroom_Classification"
dagshub.init(repo_owner='arthur-gtgn', repo_name='devops-final-project', mlflow=True) # type: ignore

df = pd.read_csv('data/mushrooms.csv')

# Splitting data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    pd.get_dummies(df.drop('class', axis=1)),
    df['class'],
    test_size=0.2,
    stratify=df['class'],
    random_state=42
)

def train_model(criterion='gini', n_estimators=100, max_depth=None, bootstrap=True) -> RandomForestClassifier:
    """Train a Random Forest model.

    Args:
        criterion (str, optional): The function to measure the quality of a split. Defaults to 'gini'.
        n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
        max_depth (int, optional): The maximum depth of the tree. Defaults to None.
        bootstrap (bool, optional): Whether to use bootstrap samples. Defaults to True.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    with mlflow.start_run(nested=True):
        params = dict(
            criterion=criterion,
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
        )

        model = RandomForestClassifier(
            **params, # type: ignore
            random_state=42,
        ).fit(X_train, y_train)

        # **Exactly one** parameter-logging call → no duplicates
        mlflow.log_params(params)

        return model

if __name__ == "__main__":
    mlflow.set_tracking_uri("https://dagshub.com/arthur-gtgn/devops-final-project.mlflow") # type: ignore
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(nested=True):
        model = train_model()
        mlflow.sklearn.log_model(model, "random_forest_model") # type: ignore
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy) # type: ignore
        print("Model accuracy:", accuracy)
