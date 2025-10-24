"""hparams.py: Hyperparameter configuration and management.

This module provides the Hparams class for managing model and experiment parameters,
along with a registration system for model-specific hyperparameter configurations.
"""
import os
from typing import Dict, Callable, Any
from argparse import Namespace

import numpy as np

from module.model.core.base import make_model

HparamFn = Callable[[Any, Any, Any], Any]
PARAM_REGISTRY: Dict[str, HparamFn] = {}


def register_hparams(model_name: str):
    """Register hyperparameter configuration function for a model.
    
    Args:
        model_name (str): Name of the model to register parameters for.
        
    Returns:
        Callable: Decorator function that registers the hyperparameter function.
    """
    def decorator(fn: HparamFn):
        PARAM_REGISTRY[model_name] = fn
        return fn
    return decorator


def update_hparams(hp, args, dataset):
    """Update hyperparameters using the registered configuration function.
    
    Args:
        hp (Hparams): Hyperparameter object to update.
        args (Namespace): Command-line arguments.
        dataset (DatasetLoader): Dataset loader object.
        
    Returns:
        Hparams: Updated hyperparameter object.
        
    Raises:
        AssertionError: If the model is not registered in PARAM_REGISTRY.
    """
    try:
        fn = PARAM_REGISTRY[args.model]  # or args.model_name
    except KeyError as e:
        raise AssertionError(f"Unknown model: {args.model}. Known: {list(PARAM_REGISTRY)}") from e
    return fn(hp, args, dataset)

### Class to manage model and experiment parameters
class Hparams:
    """
    Class to manage model and experiment parameters.
    """
    def __init__(self, args: Namespace=None):
        if args is None:
            return
        self.model_name = args.model
        self.model_class = make_model(self.model_name)
        self.model_params = None
        self.binary_model = False # Whether the model is binary classification, default to false, model should set this if needed
        self.encoder = None
        self.encoder_params = None

        self.rs = args.random_state
        self.rng = np.random.default_rng(self.rs)

        self.retrain = args.retrain
        self.tune = args.tune
        self.retune = getattr(args, "retune", False)
        self.selection = args.selection
        self.reset_results = args.reset_results
        self.k_folds = getattr(args, "k_folds", 5)
        self.inner_splits = getattr(args, "inner_splits", (3 if self.tune else None))

        self.io_params = {
            'output_dir': args.output_dir,
            'result_dir': args.result_dir,
            'model_dir': args.model_dir,
            'param_dir': args.param_dir,
        }
        os.makedirs(self.io_params['output_dir'], exist_ok=True)
        os.makedirs(f"{self.io_params['output_dir']}/{self.io_params['result_dir']}", exist_ok=True)
        os.makedirs(f"{self.io_params['output_dir']}/{self.io_params['model_dir']}", exist_ok=True)
        os.makedirs(f"{self.io_params['output_dir']}/{self.io_params['param_dir']}", exist_ok=True)

        self.noise_dist_qf = None # ONLY used for stability metric


    def get_state(self):
        """ Get the current state of the Hparams instance. We only save relevant state.

        Returns:
            dict: A dictionary containing the current state of the Hparams instance.
        """
        return {
            'model_name': self.model_name,
            'model_params': self.model_params,
            'encoder': self.encoder,
            'io_params': self.io_params,
            'rs': self.rs,
            'rng': self.rng,
            'k_folds': self.k_folds,
            'inner_splits': self.inner_splits,
            'retrain': self.retrain,
            'retune': self.retune,
            'tune': self.tune,
            'selection': self.selection,
            'reset_results': self.reset_results,
            'dataset_name': getattr(self, 'dataset_name', None),
            'config': getattr(self, 'config', None),
        }

    def set_state(self, state):
        """ Set the state of the Hparams instance.

        Args:
            state (dict): A dictionary containing the state to set.
        """
        self.model_name = state['model_name']
        self.model_class = make_model(self.model_name)
        self.model_params = state['model_params']
        self.encoder = state['encoder']
        self.io_params = state['io_params']
        self.rs = state['rs']
        self.rng = state['rng']
        # Optional fields for backward compatibility with older pickles
        self.k_folds = state.get('k_folds', getattr(self, 'k_folds', 5))
        self.inner_splits = state.get('inner_splits', getattr(self, 'inner_splits', None))
        self.retrain = state.get('retrain', getattr(self, 'retrain', False))
        self.retune = state.get('retune', getattr(self, 'retune', False))
        self.tune = state.get('tune', getattr(self, 'tune', False))
        self.selection = state.get('selection', getattr(self, 'selection', False))
        self.reset_results = state.get('reset_results', getattr(self, 'reset_results', False))
        self.dataset_name = state.get('dataset_name', None)
        self.config = state.get('config', None)
