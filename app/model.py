import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE

import pickle

predictor = pickle.load(open('../EDA and model/modelV2.pkl', 'rb'))
outback_df = pd.read_csv('../data/outback.csv')
outback_df.miles = outback_df.miles.astype(int)
