from sklearn.base import TransformerMixin
import logging
import sys

class ignorer(TransformerMixin):
    """ignoruje wyjątki xd i zpaisuje je do logu zamiast kończyć wykonanie
    jeśli zajdzie wyjątek dalej przekazywane są poprzednie dane
    to jest na potrzeby RareLabelEncoder i WoEEncoder, ponieważ rzucają wyjątek jeśli w danych nie ma danych kategorycznych 
    a my chcemy żeby działało zawsze


    """
    __slots__ = ['_broken','_logger','_baseobj']
    def __init__(self, baseobj):
        """
        Parameters
        ----------
        baseobj: Any - Estymator/Transfomer z metodami fit(X,y)->Self i transform(X)->DataFrame
        """
        super().__init__()
        self._broken = False
        self._logger = logging.getLogger(__name__)
        try:
            self._baseobj =baseobj
        except:
            self._logger.warning("kwargs are broken")
            self._logger.warning(sys.exc_info())
            self._broken = True
        finally:
            self._logger.info(f"{baseobj} initialized")
    def fit(self, X, y):
        try:
            assert(self._broken == False)
            self._baseobj.fit(X,y)
        except AssertionError:
            self._logger.info(f"{self._baseobj} is already broken")
            return
        except:
            self._broken = True
            self._logger.warning(f"{self._baseobj}fit is broken")
            self._logger.warning(sys.exc_info())

        finally:
            return self
    def transform(self, X):
        try:
            assert(self._broken == False)
            return self._baseobj.transform(X)
        except AssertionError:
            self._logger.info(f"{self._baseobj} is already broken")
        except:
            self._broken = True
            self._logger.warning(f"{self._baseobj}transform is broken")
            self._logger.warning(sys.exc_info())
        finally:
            return X