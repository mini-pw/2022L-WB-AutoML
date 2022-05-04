from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import WOEEncoder

from .zerovarremover import ZeroVarRemover
from .outliers import OutlierTransformer
from .idremover import idRemover
import logging
import sys
from .ignorer import ignorer



def get_pre()-> Pipeline:
    return make_pipeline(
        idRemover(), # usunięcie zmiennych które prawdopodobnie są id
        ZeroVarRemover(), # usunięcie zmiennych o zerowej wariancji
        SimpleImputer(strategy="most_frequent"), # uzupełnienie braków danych
        OutlierTransformer(), # zastąpienie outlierów wartościami skrajnymi
        ignorer(QuantileTransformer()), # upodobnienie rozkładu do rozkładu normalnego
        ignorer(StandardScaler()), # przeskalowanie do rozkładu normalnego
        ignorer(WOEEncoder()) # zamienienie objectów ich woe
    )