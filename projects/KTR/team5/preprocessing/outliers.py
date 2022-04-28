from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import logging
from time import perf_counter
from typing import Optional

class OutlierTransformer(BaseEstimator,TransformerMixin):
    """
    Class finds and replaces values in quantile or 1-quantile with largest values that 
    are not classified as outliers

    Parameters
    ----------
    skip: list[str] - list of names of columns to skip
    quantile: float - quantile to be classifier as outliers

    Attributes
    ----------
    quantile: float - quantile to be replaced
    skip: list[str] - list of columns to ignore

    """
    __slots__ = ('skip','quantile','_limits','_logger')
    def __init__(self, skip: list = None, quantile:float = .025) -> None:
        self.quantile = quantile
        self.skip = skip
        self._limits = None
        self._logger = logging.getLogger(__name__)

    def fit(self, X: pd.DataFrame, y:Optional[pd.DataFrame]=None):
        """
        Finds the limits of values which will be replaced

        Parameters
        ----------
        X: pandas.DataFrame - DataFrame to be transformed
        y: pandas.Series - Optional Series of target variable
        """
        __start = perf_counter()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._logger.info(f"Time of fitting OutlierTransformer: {perf_counter()-__start}")
        self._limits = pd.DataFrame(X.quantile(self.quantile))
        self._limits[1 - self.quantile] = X.quantile(1 - self.quantile).values
        if self.skip is not None:
            self._limits.loc[self.skip] = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame

        Parameters
        ----------
        X: pandas.DataFrame - DataFrame to be transformed
        y: pandas.Series - Optional Series of target variable
        """
        __start = perf_counter()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        q1 = self._limits.iloc[:,0]
        q2 = self._limits.iloc[:,1]
        outliers_low = X < q1
        X = X.mask(outliers_low, q1, axis=1)
        outliers_top = X > q2
        X = X.mask(outliers_top, q2, axis=1)
        self._logger.info(f"Time of transforming OutlierTransformer: {perf_counter()-__start}")
        return X




