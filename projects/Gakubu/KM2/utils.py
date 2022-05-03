import pandas as pd
import numpy as np
import autogluon as ag
import urllib.request
import io

from sklearn.model_selection import StratifiedKFold, train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from scipy.io import arff

def create_data_frame_from_arff(url):
    """
        Funkcja wczytuje zbiór danych podany w formacie podany w formacie .ARFF
        korzystając z adresu url. 
        
        Finalnym rezultatem jest pandasowa ramka danych
    """
    # Wczytujemy plik typu .ARFF do dwóch zmiennych: data - zawiera dane, meta - zawiera m.in. nazwy kolumn 
    data, meta = arff.loadarff( 
        io.StringIO(urllib.request.urlopen(url).read().decode("utf-8"))
    )
    data = pd.DataFrame(data)
    data.columns = pd.DataFrame(meta)[0]
    return data


def make_kfold_cross_valiation(
    url,
    name,
    variable_to_predict,
    no_of_folds,
    random_state=None,
    shuffle=False,
    preset="medium_quality", # możliwe wartości: "best_quality", "high_quality", "good_quality",
                                                # "medium_quality", "optimize_for_deployment", "ignore_text"
):
    """
        Funkcja zwracające rezlataty kroswalidacji z zadaną liczbą foldów 
        w postaci tablicy, która dla każdego utworzonego modelu (autogluon tworzy ich wiele)
        określa wiele parametrów uczenia, m.in.
        
        wartości metryki, czas predykcji, czasu uczenia, parametry związane z charakterystyką uczenia
        autogluonem.
    """
    data = create_data_frame_from_arff(url)

    # preprocessing danych
    string_columns = data.select_dtypes(include="object")
    for col in string_columns:
        try:
            data[col] = data[col].str.decode(encoding="utf-8")
        except (UnicodeDecodeError, AttributeError):
            pass

    y = data[variable_to_predict]
    X = data.drop(variable_to_predict, axis=1)

    kf = StratifiedKFold(
        n_splits=no_of_folds, random_state=random_state, shuffle=shuffle
    )

    results = [None] * no_of_folds # lista pomocnicza przechowująca rezultaty poszczególnych foldów

    fold_index = 0
    for train_indices, test_indices in kf.split(X, y):
        predictor = TabularPredictor(
            label=variable_to_predict,
            eval_metric="roc_auc",
            problem_type="binary",
            learner_kwargs={"cache_data": False},
        ).fit(data.iloc[train_indices], presets=preset, time_limit=120)

        leaderboard = predictor.leaderboard(data.iloc[test_indices], silent=True)

        # najwieksze roc_auc dla zbioru testowego
        max_roc_auc_model = leaderboard.loc[
            leaderboard.score_test == max(leaderboard.score_test),
            ["model", "score_test"],
        ]
        max_roc_auc_model["dataset"] = name

        results[fold_index] = max_roc_auc_model

        fold_index += 1

        print(f"\n Fold {fold_index} \n")

    return results
