"""Abstract base model for deblurring models."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load_weights(self, path):
        pass

    @abstractmethod
    def predict(self, image):
        pass
