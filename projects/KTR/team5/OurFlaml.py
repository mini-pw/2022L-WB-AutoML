from time import perf_counter
from flaml import AutoML
from .preprocessing.preprocessor import get_pre
from sklearn.base import BaseEstimator
import pandas as pd
from typing import Any
import logging
class OurFlaml(BaseEstimator):
    def __init__(self,**kwargs):
        super().__init__()
        self.ml = AutoML(**kwargs)
        self._logger = logging.getLogger(__name__)
        
    def train(self,data_x:pd.DataFrame, data_y:pd.DataFrame, cv: Any=None, **kwargs):
        """
        Parameters
        ----------
        data_x: pandas.DataFrame - ramka danych ze zmiennymi objaśniającymi
        data_y: pandas.Series|numpy.Array - seire lub array ze zmienna objaśnianą
        cv: Sklearn.BaseCrossValidator|
        """
        pre = get_pre()
        __start = perf_counter()
        data_transformed_x = pre.fit_transform(data_x,data_y)
        self._logger.info(f"preprocessing finished in {perf_counter() - __start}")
        kwargs['eval_method']='cv'
        kwargs['split_type']=cv
        self.ml.fit(data_transformed_x, data_y, **kwargs)

    def fit(self,x: pd.DataFrame,y: pd.DataFrame,**kwargs):
        return self.train(x,y,**kwargs)

    def predict(self, x:pd.DataFrame, **pred_kwargs) -> pd.DataFrame:
        return self.ml.predict(x,**pred_kwargs)