"""kantch_metric.py: Adversarial robustness evaluation using Kantchelian attack.

This module implements metrics for evaluating model robustness against adversarial
examples using the Kantchelian attack for tree. Measures both attack
feasibility and required perturbation distance.
"""
from module.metric.base_metrics import BaseMetric, register_metric
from sklearn.metrics import accuracy_score
from typing import Dict
import numpy as np
import logging
from module.metric.lib.kantch.toolbox import KantchModel


@register_metric()
class KantchMetric(BaseMetric):
    """Adversarial robustness metric using Kantchelian attack.
    
    Evaluates model robustness by computing adversarial examples using L-infinity
    norm perturbations. Requires continuous features for proper attack generation.
    """

    NAME = "kantch_attack"
    REQUIRES_BINARY_FEATURES = False  # Needs continuous features
    
    def setup(self, model, hparams, split_data, **params) -> None:
        """Initialize Kantchelian attack with model and epsilon configuration.
        
        Args:
            model: Trained model to attack.
            hparams (Hparams): Hyperparameters containing config and dataset info.
            split_data (Split): Data split (unused but required by interface).
            **params: Additional parameters including epsilon and use_dataset_epsilon.
        """
        from module.utils import get_epsilon_from_config
        
        logger = logging.getLogger("KantchMetric.setup")
        
        ref_model = model
        
        # Determine epsilon with priority:
        # 1) Explicit 'epsilon' parameter from YAML
        # 2) Dataset-specific from config (if use_dataset_epsilon=true)
        # 3) Default from config
        explicit_epsilon = params.get("epsilon", None)
        use_dataset_epsilon = params.get("use_dataset_epsilon", False)
        
        # Get config and dataset info
        config = getattr(hparams, 'config', None)
        dataset_name = getattr(hparams, 'dataset_name', "")
        
        logger.debug(f"Kantch setup for dataset '{dataset_name}'")
        logger.debug(f"  use_dataset_epsilon: {use_dataset_epsilon}")
        logger.debug(f"  config available: {config is not None}")
        if config and "dataset_epsilon" in config:
            logger.debug(f"  configured datasets: {list(config['dataset_epsilon'].keys())}")
        
        if explicit_epsilon is not None:
            # Explicit override in YAML
            self.epsilon = float(explicit_epsilon)
            logger.info(f"Using explicit epsilon: {self.epsilon}")
        elif use_dataset_epsilon:
            # Use dataset-specific epsilon from config
            self.epsilon = get_epsilon_from_config(config, dataset_name, explicit_value=None)
            logger.info(f"Using dataset-specific epsilon for '{dataset_name}': {self.epsilon}")
        else:
            # Use default from config
            self.epsilon = config.get('default_epsilon', 0.1) if config else 0.1
            logger.info(f"Using default epsilon: {self.epsilon}")
        
        if hasattr(model, 'get_model') and callable(getattr(model, 'get_model')):
            ref_model = model.get_model()
        elif hasattr(model, 'model'):
            ref_model = model.model

        self.kantch_model = KantchModel.from_model(ref_model)
        self.adv_examples = self.kantch_model.adversarial_examples(split_data.X_test, split_data.y_test, options={'epsilon': self.epsilon})
        check_score = self.kantch_model.adversarial_accuracy(split_data.X_test, split_data.y_test, epsilon=self.epsilon)
        example_score = accuracy_score(split_data.y_test, self.kantch_model.predict(self.adv_examples))
        logger.info(f"Kantch attack accuracy (epsilon={self.epsilon}): {check_score:.4f}, example accuracy: {example_score:.4f}")
        if not np.isclose(check_score, example_score, atol = 1e-2):
            logger.warning(f"Kantch attack accuracy ({check_score:.4f}) and example accuracy ({example_score:.4f}) differ significantly!")
        
    def compute(self, predictions, split_data, **params) -> Dict[str, float]:
        return {
            "kantch_attack_adv_acc": accuracy_score(split_data.y_test, predictions["pred_fn"](self.adv_examples)),
        }

    def cleanup(self):
        self.adv_examples = None
