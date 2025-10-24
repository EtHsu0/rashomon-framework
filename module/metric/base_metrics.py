"""base_metrics.py: Base metric interface and registration system.

This module provides the BaseMetric abstract class that defines the common
interface for all metric implementations, along with a plugin-based registration
system for metric discovery.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Set
from dataclasses import dataclass

from module.datasets import Split
from module.hparams import Hparams

METRIC_REGISTRY: Dict[str, Type["BaseMetric"]] = {}


def register_metric():
    """Decorator to register a metric class in the global registry.
    
    Returns:
        Callable: Decorator function that registers the metric class.
        
    Raises:
        ValueError: If metric NAME is None or already registered.
    """
    def decorator(cls: Type["BaseMetric"]):
        if cls.NAME is None:
            raise ValueError(f"Metric {cls.__name__} must define a NAME class attribute")
        if cls.NAME in METRIC_REGISTRY:
            raise ValueError(f"Metric name '{cls.NAME}' already registered by {METRIC_REGISTRY[cls.NAME].__name__}")
        METRIC_REGISTRY[cls.NAME] = cls
        return cls
    return decorator


def get_metric(name: str) -> Type["BaseMetric"]:
    """Retrieve a metric class from the registry by name.
    
    Args:
        name (str): Name of the registered metric.
        
    Returns:
        Type[BaseMetric]: The registered metric class.
    """
    return METRIC_REGISTRY[name]


@dataclass
class EvalContext:
    """Evaluation context containing model, data, and precomputed predictions.
    
    This dataclass bundles all information needed for metric computation,
    including the trained model, hyperparameters, data splits, and optionally
    precomputed predictions and probabilities to avoid redundant computation.
    """
    model: Any
    hparams: Hparams
    split: Split
    rng: Any
    # always available if requested:
    pred_train: Any = None
    pred_test: Any = None
    proba_train: Any = None
    proba_test: Any = None


class BaseMetric(ABC):
    """Base class for all metric implementations.

    Metrics follow a setup-compute-cleanup lifecycle to efficiently evaluate
    models across multiple dimensions. Metrics declare their requirements
    (predictions, probabilities, sensitive attributes, etc.) via the NEEDS
    attribute, allowing the framework to precompute only what's necessary.

    Class Attributes:
        NAME (str): Unique identifier for the metric.
        NEEDS (Set[str]): Set of required inputs (e.g., {"pred", "proba", "sens"}).
        REQUIRES_BINARY_FEATURES (bool): Whether metric needs binary features (True)
            or continuous features (False).

    Lifecycle:
        1. setup(model, hparams, split, **params) - One-time initialization
        2. compute(predictions, split, **params) - Main metric calculation
        3. cleanup() - Release resources
    """
    NAME: Optional[str] = None
    NEEDS: Set[str] = frozenset()
    REQUIRES_BINARY_FEATURES: bool = False

    def setup(self, _model, _hparams: Hparams, _split: Split, **params: Any) -> None:
        """Setup metric before computation (optional).
        
        Args:
            _model: Trained model instance.
            _hparams (Hparams): Hyperparameters.
            _split (Split): Data split.
            **params: Metric-specific configuration.
        """
        return

    @abstractmethod
    def compute(self, predictions, split: Split, **params: Any) -> Dict[str, float]:
        """Compute metric values.
        
        Args:
            predictions (dict): Dictionary containing 'train' and 'test' predictions.
            split (Split): Data split with features and labels.
            **params: Metric-specific parameters.
            
        Returns:
            Dict[str, float]: Dictionary of metric names to values.
        """
        raise NotImplementedError()

    def cleanup(self) -> None:
        """Cleanup resources after metric computation (optional)."""
        return

    def display_name(self) -> str:
        """Get human-readable name for the metric.
        
        Returns:
            str: Display name for the metric.
        """
        return self.NAME or self.__class__.__name__
