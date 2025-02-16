import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from src.logger.logging import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            model_report: dict[str, float] = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n' + '=' * 80 + '\n')
            logging.info(f'Model Report: {model_report}')

            best_model_name, best_model_score = max(model_report.items(), key=lambda x: x[1])
            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name}, R² Score: {best_model_score}')
            print('\n' + '=' * 80 + '\n')
            logging.info(f'Best Model Found: {best_model_name}, R² Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.error('Exception occurred during model training', exc_info=True)
            raise CustomException(e, sys)
