from abc import ABC, abstractmethod
from datatools.slice import Slice
from config import TRAINING_DIR


class BaseTrainer(ABC):
    def __init__(self, train_slice: Slice, test_slice: Slice = None, seed: int = 42, output_dir: str = TRAINING_DIR):
        self.train_slice = train_slice
        self.test_slice = test_slice
        self.seed = seed
        self.output_dir = output_dir
        self._setup()

    def _setup(self):
        from datatools.trainer.utils import set_seed
        set_seed(self.seed)
        import os
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def save(self):
        pass
