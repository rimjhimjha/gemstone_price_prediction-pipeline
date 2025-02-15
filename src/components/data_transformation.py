import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import CustomException
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.utils.utils import save_object




@dataclass
class DataTransformationConfig:
    pass






class DataTranformation:
    def __init__(self):
        pass



    def initiate_data_Tranformation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e,sys)



