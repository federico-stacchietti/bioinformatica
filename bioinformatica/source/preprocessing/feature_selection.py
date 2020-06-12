import pandas as pd
import numpy as np
from boostaroota import BoostARoota
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import cpu_count


def boruta_filter(dataset: pd.DataFrame, labels: np.array, max_iter: int, p_value_threshold: float, random_state: int) \
        -> pd.DataFrame:
    forest = RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5)
    boruta_selector = BorutaPy(
        forest,
        n_estimators='auto',
        verbose=2,
        alpha=p_value_threshold,
        max_iter=max_iter,
        random_state=random_state
    )
    return boruta_selector.fit_transform(dataset.values, labels)


def boostaroota_filter(dataset : pd.DataFrame, labels : np.array) -> pd.DataFrame:
    br = BoostARoota(metric='logloss', silent=True)
    return br.fit_transform(dataset, labels)
