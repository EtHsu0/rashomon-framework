"""post_base.py: Base class for post-processing fairness models.

This module provides the PostBase abstract class for models that perform
post-processing on base model predictions to achieve fairness constraints.
These models learn from base model probabilities and may use sensitive
attributes at both training and inference time.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike


class PostBase(ABC):
    """Base interface for fairness post-processing models.
    
    Post-processors learn to adjust predictions from a base model to satisfy
    fairness constraints. They may require sensitive attributes at inference time
    depending on the fairness criterion being enforced.
    """
    
    requires_sensitive_at_inference: bool = True

    def __init__(self, **params: Any):
        """Initialize the post-processor with hyperparameters.
        
        Args:
            **params: Model hyperparameters.
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
            **params: New hyperparameter values.
            
        Returns:
            Self: This instance for method chaining.
        """
        self.params.update(params)
        return self

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sensitive: ArrayLike,
    ) -> "PostBase":
        """Train the post-processor on base model outputs and sensitive attributes.
        
        Args:
            X (ArrayLike): Feature matrix or base model predictions.
            y (ArrayLike): True labels.
            sensitive (ArrayLike): Sensitive attribute values.
            
        Returns:
            PostBase: Fitted post-processor instance.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
    ) -> np.ndarray:
        """Predict labels using the base model (without post-processing).
        
        Args:
            X (ArrayLike): Feature matrix.
            
        Returns:
            np.ndarray: Predicted labels.
        """
        raise NotImplementedError

    @abstractmethod
    def post_predict(self, X: ArrayLike):
        """Predict labels using the post-processed fair model.
        
        This method applies fairness constraints learned during post-processing
        and should be used for fairness evaluation.
        
        Args:
            X (ArrayLike): Feature matrix.
            
        Returns:
            np.ndarray: Fair predictions after post-processing.
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sensitive: ArrayLike,
        criterion: str = 'sp',
    ) -> None:
        """Learn post-processing transformation to satisfy fairness criteria.
        
        Args:
            X (ArrayLike): Feature matrix.
            y (ArrayLike): True labels.
            sensitive (ArrayLike): Sensitive attribute values.
            criterion (str, optional): Fairness criterion ('sp', 'eo', 'eod'). Defaults to 'sp'.
        """
        raise NotImplementedError
