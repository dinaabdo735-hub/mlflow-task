import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

mlflow.set_experiment("California_Housing_Experiment")


def prepare_data():
  data = fetch_california_housing(as_frame=True)
  X = data.data
  y = data.target
  X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  return X_train, X_val, y_train, y_val


def evaluate_model(model, X_val, y_val):
  preds = model.predict(X_val)
  rmse = np.sqrt(mean_squared_error(y_val, preds))
  mae = mean_absolute_error(y_val, preds)
  r2 = r2_score(y_val, preds)
  return rmse, mae, r2


def main():
  X_train, X_val, y_train, y_val = prepare_data()

  runs_config = [
      {"run_name": "Run_1", "max_depth": 3, "learning_rate": 0.1},
      {"run_name": "Run_2", "max_depth": 5, "learning_rate": 0.05},
      {"run_name": "Run_3", "max_depth": 7, "learning_rate": 0.01},
  ]

  for config in runs_config:
    with mlflow.start_run(run_name=config["run_name"]):
      mlflow.log_param("max_depth", config["max_depth"])
      mlflow.log_param("learning_rate", config["learning_rate"])

      model = GradientBoostingRegressor(
          max_depth=config["max_depth"],
          learning_rate=config["learning_rate"],
          random_state=42,
      )
      model.fit(X_train, y_train)

      rmse, mae, r2 = evaluate_model(model, X_val, y_val)

      mlflow.log_metric("rmse", rmse)
      mlflow.log_metric("mae", mae)
      mlflow.log_metric("r2", r2)

      mlflow.sklearn.log_model(
          model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"]
      )


if __name__ == "__main__":
  main()