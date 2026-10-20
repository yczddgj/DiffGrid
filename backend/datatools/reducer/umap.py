import numpy as np
import umap
import joblib
from .base import BaseReducer
from config import n_neighbors
from datatools.reducer.registry import register_reducer


@register_reducer('umap')
class UMAPReducer(BaseReducer):
    def __init__(self, n_components=2):
        self.model = umap.UMAP(n_components=n_components,n_neighbors=n_neighbors,metric='cosine')

    def fit(self, features: np.ndarray):
        self.model.fit(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.model.transform(features)

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        reducer = cls()
        reducer.model = joblib.load(path)
        return reducer


@register_reducer('densmap')
class DENSMAPReducer(BaseReducer):
    def __init__(self, n_components=2):
        self.model = umap.UMAP(n_components=n_components,densmap=True,
                               n_neighbors=n_neighbors)

    def fit(self, features: np.ndarray):
        self.model.fit(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.model.transform(features)

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        reducer = cls()
        reducer.model = joblib.load(path)
        return reducer
