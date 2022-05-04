from time import perf_counter
from sklearn.base import TransformerMixin
import logging
from time import perf_counter
class ZeroVarRemover(TransformerMixin):
    __slots__='_sus','_logger'
    def __init__(self):
        super().__init__()
        self._sus = None
        self._logger = logging.getLogger(__name__)
    def fit(self,x,y):
        __start = perf_counter()
        self._sus = x.loc[:,x.nunique()==1]
        self._logger.info(f"zerovarremover cols staged to be removed: {self._sus.columns.tolist()} ")
        return self
    def transform(self,x):
        return x.drop(self._sus,axis = 1)