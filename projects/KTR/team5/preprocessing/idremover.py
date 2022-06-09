from sklearn.base import TransformerMixin
import pandas as pd
import logging
from time import perf_counter
class idRemover(TransformerMixin):
    """
    finds possible ids of records and deletes them
    to fix!!!
    """
    __slots__='_sus','_logger'
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)

    def fit(self,x: pd.DataFrame,y):
        __start = perf_counter()
        self._sus = x.loc[:, (x.nunique()==x.shape[0])]
        self._logger.info(f"idremover cols staged to be removed: {self._sus.columns.tolist()} in {perf_counter()-__start}")
        return self

    def transform(self,x):
        return x.drop(self._sus, axis = 1)