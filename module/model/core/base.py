"""base.py: Base estimator interface for all model implementations.

This module provides the BaseEstimator abstract class that defines the minimal
sklearn-compatible interface for research models, along with a registration system
for model discovery and instantiation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
import numpy as np
from numpy.typing import  ArrayLike
from argparse import Namespace
from module.datasets import DatasetLoader


MODEL_REGISTRY: Dict[str, Type[BaseEstimator]] = {}


def register_model(name: str):
    """Decorator to register a model class in the global registry.
    
    Args:
        name (str): Unique identifier for the model.
        
    Returns:
        Callable: Decorator function that registers the model class.
    """
    def decorator(cls: Type[BaseEstimator]):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def make_model(name: str) -> BaseEstimator:
    """Retrieve a model class from the registry by name.
    
    Args:
        name (str): Name of the registered model.
        
    Returns:
        Type[BaseEstimator]: The registered model class.
    """
    return MODEL_REGISTRY[name]


class BaseEstimator(ABC):
    """Minimal sklearn-compatible base class for research models.
    
    Provides a consistent interface for model training, prediction, and
    hyperparameter management across all model implementations. Follows
    scikit-learn conventions for compatibility with standard ML pipelines.
    """

    def __init__(self, **params: Any):
        """Initialize the estimator with hyperparameters.

        Args:
            **params: Model hyperparameters stored as a dictionary.
        """
        self.params: Dict[str, Any] = dict(params)

    def get_params(self) -> Dict[str, Any]:
        """Get current hyperparameters.
        
        Returns:
            Dict[str, Any]: Shallow copy of current hyperparameters.
        """
        return dict(self.params)

    def set_params(self, **params: Any) -> Self:
        """Update hyperparameters in-place.
        
        Args:
            **params: New hyperparameter values to merge.
            
        Returns:
            Self: This estimator instance for method chaining.
        """
        self.params.update(params)
        return self

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Train the model on the provided data.
        
        Args:
            X (ArrayLike): Feature matrix of shape (n_samples, n_features).
            y (ArrayLike): Target labels of shape (n_samples,).
            
        Returns:
            Self: Fitted estimator instance.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict labels for input samples.
        
        Args:
            X (ArrayLike): Feature matrix of shape (n_samples, n_features).
            
        Returns:
            ArrayLike: Predicted labels of shape (n_samples,).
        """
        raise NotImplementedError

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute classification accuracy.

        Args:
            X (ArrayLike): Feature matrix of shape (n_samples, n_features).
            y (ArrayLike): True labels of shape (n_samples,).
            
        Returns:
            float: Classification accuracy in [0, 1].
            
        Raises:
            ValueError: If predictions and labels have mismatched lengths or y is empty.
        """
        preds = self.predict(X)
        if preds.shape[0] != y.shape[0]:
            raise ValueError("pred and y length mismatch")
        if y.size == 0:
            raise ValueError("empty y")
        return float(np.mean(preds == y))
