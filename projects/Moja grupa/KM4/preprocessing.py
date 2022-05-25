from sklearn.base import TransformerMixin, BaseEstimator
from feature_engine.encoding import WoEEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


class DateTransformer(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.feature_names = []

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        result = x.copy()
        result.columns = ["date_" + str(i) for i in range(x.shape[1])]

        for col in result:
            result[col + '_day'] = result[col].dt.day
            result[col + '_month'] = result[col].dt.month
            result[col + '_year'] = result[col].dt.year

        column_names = ["date_" + str(i) + "_" + mon
                        for mon in ["day", "month", "year"]
                        for i in range(x.shape[1])]
        self.feature_names = column_names
        return result[column_names]

    def get_feature_names(self):
        return self.feature_names
    
    
def preprocess(x, y):
    num = x.select_dtypes(['number']).columns.tolist()
    date = x.select_dtypes(['datetime']).columns.tolist()
    ob = x.select_dtypes(['object']).columns.tolist()
    cat = x.select_dtypes(['category']).columns.tolist()
    cat = cat + ob
    woe = []

    for col in cat:
        if x[col].nunique() > 20:
            woe.append(col)
            cat.remove(col)
            x[col].astype("category")

    for col in num:
        if x[col].nunique() <= 10:
            num.remove(col)
            cat.append(col)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler()),
        ("minmax", MinMaxScaler())
    ])

    date_pipe = Pipeline([
        ("datetime_transformer", DateTransformer())
    ])

    cat_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    woe_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('woe', WoEEncoder())
    ])

    column_transformer = ColumnTransformer([
        ('numeric', numeric_pipe, num),
        ('date', date_pipe, date),
        ('cat', cat_pipe, cat),
        ('woe', woe_pipe, woe)
    ], remainder='drop')

    return column_transformer
