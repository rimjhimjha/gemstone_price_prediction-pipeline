import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.utils import load_object
from src.logger.logging import logging
from src.exception.exception import CustomException

class ModelEvaluation:
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))  # Root Mean Squared Error
        mae = mean_absolute_error(actual, pred)  # Mean Absolute Error
        r2 = r2_score(actual, pred)  # R² Score
        logging.info(f"Evaluation metrics - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
        return rmse, mae, r2

    def initiate_model_evaluation(self, test_array):
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            logging.info("Model loaded successfully for evaluation.")

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate performance
            rmse, mae, r2 = self.eval_metrics(y_test, y_pred)

            # MLflow tracking
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            logging.info(f"Tracking URI: {tracking_url_type_store}")

            return {"RMSE": rmse, "MAE": mae, "R2": r2}

        except Exception as e:
            logging.error("Exception occurred during model evaluation", exc_info=True)
            raise CustomException(e, sys)
