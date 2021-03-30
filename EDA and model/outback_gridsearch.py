import pandas as pd
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE

import pickle

outback = pd.read_csv('../data/outback.csv')

ordinal = ['condition', 'title_status'] # Ordinal Encoder
categorical = ['cylinders', 'fuel', 'transmission', 'paint_color', 'model'] # OHE
numerical = ['year', 'miles']

y = outback.USD
X = outback.drop('USD', axis=1)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

ct = ColumnTransformer(transformers = [
                            ('ordinalEncoder', encoder, ordinal),
                            ('oneHotEncoder', ohe, categorical )
                            ],
                        remainder='passthrough')

regressor = RandomForestRegressor(n_estimators=20)

pipe = Pipeline(steps=[('preprocess', ct),
                        ('model', regressor)])

parameters = {
  'model__n_estimators':list(range(5,110,5)),
  'model__criterion' : ['mae'],
  'model__max_features': ['auto', 'sqrt', 'log2'],
  'model__min_samples_split': [2,3,4,5]
  }

gs = GridSearchCV(estimator=pipe, param_grid=parameters, verbose=2)
gs.fit(X, y)

pickle.dump(gs.best_estimator_, open('modelV2.pkl', 'wb'))