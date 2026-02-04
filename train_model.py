import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLPipeline:
    def __init__(self, experiment_name: str):
        """
        Initializes the MLflow Experiment tracking system.
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
        logging.info(f"Experiment initialized: {self.experiment_name} (ID: {self.experiment.experiment_id})")

    def train_xgboost(self, X_train, y_train, X_test, y_test, params=None):
        """
        Trains an XGBoost model and logs all metadata, metrics, and the model itself to MLflow.
        """
        if params is None:
            params = {"num_boost_round": 100, "max_depth": 6, "eta": 0.3}

        with mlflow.start_run(run_name="XGBoost_Optimization_Run") as run:
            # Initialize model
            model = XGBClassifier(
                n_estimators=params["num_boost_round"],
                max_depth=params["max_depth"],
                learning_rate=params["eta"],
                use_label_encoder=False,
                eval_metric="logloss"
            )

            # Fit model
            model.fit(X_train, y_train)

            # Predict and Score
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # --- MLflow Logging ---
            # 1. Log Hyperparameters
            mlflow.log_params(params)
            
            # 2. Log Metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # 3. Log Model with Signature (The YAML metadata you saw earlier)
            mlflow.xgboost.log_model(model, artifact_path="model")
            
            logging.info(f"Run Completed. Accuracy: {accuracy:.4f} | Run ID: {run.info.run_id}")
            return run.info.run_id

    def get_best_runs(self, limit=2):
        """
        Searches and returns the top performing runs from the experiment.
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["metrics.accuracy DESC"],
            max_results=limit
        )
        return runs

# --- Example of How to Execute the Professional Pipeline ---
if __name__ == "__main__":
    # In a real scenario, you would load your data here (X_train, y_train, etc.)
    # Example: pipeline = MLPipeline("Credit_Risk_Analysis")
    # pipeline.train_xgboost(X_train, y_train, X_test, y_test)
    print("AI Pipeline is ready for Production deployment.")

