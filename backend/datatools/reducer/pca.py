import numpy as np
from sklearn.decomposition import PCA
import joblib
from .base import BaseReducer
from datatools.reducer.registry import register_reducer


@register_reducer('pca')
class PCAReducer(BaseReducer):
    def __init__(self, n_components=2):
        self.model = PCA(n_components=n_components)

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
