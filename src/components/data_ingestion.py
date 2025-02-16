import pandas as pd
import numpy as np
import os
import sys
from src.logger.logging import logging
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")  

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)  

    def initiate_data_ingestion(self):  
        logging.info("Data ingestion started")
        try:
            data_path = "experiment/dataset/sample_submission.csv"  
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}")

            data = pd.read_csv(data_path)
            logging.info("Successfully read dataset")

            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw dataset saved at {self.ingestion_config.raw_data_path}")

            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("Train-test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(f"Train dataset saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test dataset saved at {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion process completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            ) 
        except Exception as e:
            logging.info("Exception occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
