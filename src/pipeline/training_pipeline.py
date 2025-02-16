import os
import sys
from src.logger.logging import logging
from src.exception.exception import CustomException

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

try:
    # Data Ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

    # Model Training
    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initiate_model_training(train_arr, test_arr)

except Exception as e:
    logging.error(f"Error in main pipeline: {str(e)}")
    raise CustomException(e, sys)
