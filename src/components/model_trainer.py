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

from src.utils.utils import save_object,evaluate_model

from sklearn.linear_model import LinearRegression


@dataclass
class ModeltrainerConfig:
    pass






class Modeltrainer:
    def __init__(self):
        pass



    def initiate_Modeltrainer(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e,sys)



