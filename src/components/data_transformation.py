import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.logger.logging import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')

            # Define categorical and numerical columns
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the custom ranking for ordinal variables
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical Pipeline
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Exception occurred in get_data_transformation")
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully")
            logging.info(f'Train DataFrame Head:\n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head:\n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'price'
            drop_columns = ['id']

            # Drop only if the columns exist
            drop_columns = [col for col in drop_columns if col in train_df.columns]
            train_df = train_df.drop(columns=drop_columns, axis=1)
            test_df = test_df.drop(columns=drop_columns, axis=1)

            # Splitting into features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing pickle file saved successfully.")

            return train_arr, test_arr

        except Exception as e:
            logging.info("Exception occurred in initialize_data_transformation")
            raise CustomException(e, sys)
