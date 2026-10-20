from abc import ABC, abstractmethod
import numpy as np


class BaseReducer(ABC):
    @abstractmethod
    def fit(self, features: np.ndarray):
        pass

    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def save(self, path: str):
        """保存降维模型（可选）"""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        """从文件加载降维模型（可选）"""
        raise NotImplementedError
