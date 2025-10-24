"""rash_base.py: Base class for Rashomon set enumeration methods.

This module provides the RsetBase abstract class for models that enumerate
multiple near-optimal solutions (Rashomon sets) rather than returning a single
optimal model. These methods are useful for exploring model multiplicity and
understanding prediction stability across equivalently-good models.
"""
from typing import Any, Dict
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class RsetBase(ABC):
    """Base interface for Rashomon set enumeration methods.
    
    Rashomon set methods generate multiple models that perform similarly well
    according to a loss function, allowing analysis of model multiplicity and
    the variability of predictions across near-optimal solutions.
    """
    
    def __init__(self, **params: Any):
        """Initialize the Rashomon set enumerator.
        
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
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Enumerate the Rashomon set by training on the data.
        
        Args:
            X (ArrayLike): Feature matrix of shape (n_samples, n_features).
            y (ArrayLike): Target labels of shape (n_samples,).
            
        Returns:
            Self: Fitted instance with enumerated Rashomon set.
        """
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X: ArrayLike, idx: int) -> ArrayLike:
        """Predict labels using a specific model from the Rashomon set.
        
        Args:
            X (ArrayLike): Feature matrix of shape (n_samples, n_features).
            idx (int): Index of the model in the Rashomon set.
            
        Returns:
            ArrayLike: Predicted labels of shape (n_samples,).
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self, idx: int) -> Any:
        """Retrieve a specific model from the Rashomon set.
        
        Args:
            idx (int): Index of the model to retrieve.
            
        Returns:
            Any: The model object at the specified index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_models(self) -> int:
        """Get the total number of models in the Rashomon set.
        
        Returns:
            int: Number of enumerated models.
        """
