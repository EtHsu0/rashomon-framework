"""metric.py: Main metrics orchestration and evaluation class.

This module provides the Metrics class which manages metric computation across
multiple evaluation dimensions (accuracy, fairness, privacy, robustness, stability).
It handles prediction generation, metric setup, and result collection.
"""
import logging
from collections import defaultdict
import yaml
import numpy as np
import pandas as pd

from module.model.core.rash_base import RsetBase
from module.metric.base_metrics import get_metric

class Metrics:
    """ Metric Class """
    def __init__(self, args, hparams):
        config_file = args.config
        rng = hparams.rng 
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.config_file = config_file
        with open(self.config_file, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.metric_names = [metric["name"] for metric in self.config["metrics"]]
        if "metrics" in self.config:
            for metric in self.config["metrics"]:
                if "name" in metric:
                    if "fairness" in metric["name"].lower():
                        args.fairness_mode = True
                    if "stability" in metric["name"].lower():
                        hparams.noise_dist_qf = -1 # Flag to indicate stability metric is used but not initialized yet
                        

        #region Attributes that will be initialized/used later

        self.model = None
        self.eval_model = None
        self.eval_predict = None
        self.result = {}

    def _predict_label(self, pred_fn, X):
        """Internal helper to get predictions from a prediction function.
        
        Args:
            pred_fn (Callable): Prediction function.
            X (np.ndarray): Feature matrix.
            
        Returns:
            np.ndarray: Predicted labels.
        """
        return pred_fn(X)

    def get_predictions(self, model, split, pred_fn=None, use_binary_features=False, from_rash=False):
        """Get predictions from a model.
        
        Args:
            model: Model to get predictions from
            split: Data split
            pred_fn: Custom prediction function (optional)
            use_binary_features: Whether to use binary features
            from_rash: True if called from evaluate_rash (disables auto-detection)
        """
        if pred_fn is None:
            pred_fn = model.predict
        
        if from_rash:
            # From Rashomon set evaluation - trust caller's use_binary_features exactly
            use_binary = use_binary_features
        else:
            # Regular model evaluation - apply auto-detection for safety
            use_binary = use_binary_features
            
            # Auto-detection: if model trained on binary but can't transform, must use binary
            if not use_binary:
                model_trained_on_binary = hasattr(split, 'binarizer') and split.binarizer is not None
                model_has_encoder = hasattr(model, 'encoder') and model.encoder is not None
                
                # If model was trained on binary but can't transform, we must use binary
                if model_trained_on_binary and not model_has_encoder:
                    use_binary = True
        
        # Select appropriate data representation
        if use_binary and hasattr(split, 'binarizer') and split.binarizer is not None:
            # Use binary features
            binary_data = split.get_binary_data()
            return {
                "pred_fn": pred_fn,
                "train": self._predict_label(pred_fn, binary_data['X_train']),
                "test": self._predict_label(pred_fn, binary_data['X_test']),
                "model": model
            }
        else:
            # Use continuous features
            return {
                "pred_fn": pred_fn,
                "train": self._predict_label(pred_fn, split.preprocess(split.X_train)),
                "test": self._predict_label(pred_fn, split.preprocess(split.X_test)),
                "model": model
            }

    def evaluate(self, model, hparams, split_data) -> None:
        self.result = {}
        """ Evaluate single model or Rashomon set """
        if isinstance(model, RsetBase):
            return self.evaluate_rash(model, hparams, split_data)
        if hasattr(model, 'sweep') and model.sweep:
            return self.evaluate_sweep(model, hparams, split_data)

        for metric_config in self.config["metrics"]:
            metric_class = get_metric(metric_config["name"])()
            # Pass YAML parameters to setup (not just compute)
            params = {k: v for k, v in metric_config.items() if k != "name"}
            metric_class.setup(model, hparams, split_data, **params)
            
            # Check if this metric requires binary features and get appropriate predictions
            requires_binary = (metric_class.REQUIRES_BINARY_FEATURES and 
                             hasattr(split_data, 'binarizer') and split_data.binarizer is not None)
            preds = self.get_predictions(model, split_data, use_binary_features=requires_binary)
            
            # Compute result and cleanup
            result = metric_class.compute(preds, split_data, **params)
            metric_class.cleanup()
            self.result.update(result)
        return self.result

    def evaluate_sweep(self, model, hparams, split_data) -> None:
        metric_classes = []
        for metric in self.config["metrics"]:
            name = metric["name"]
            metric_class = get_metric(name)()
            params = {k: v for k, v in metric.items() if k != "name"}
            metric_class.setup(model, hparams, split_data, **params)
            metric_classes.append(metric_class)

        results = defaultdict(list)
        for criterion, model_list in model.pareto_models.items():
            # For DPF: DPF only optimizes SP, so evaluate all fairness metrics on the SP Pareto frontier
            # For post-processing models: use the actual optimization criterion
            from module.model.fairness.dpf import DPF
            if isinstance(model, DPF):
                params = {"criterion": "all"}
            else:
                params = {"criterion": criterion}
            for alpha, post_model in model_list:
                predictions = self.get_predictions(post_model, split_data, pred_fn=post_model.predict)
                results[f"{criterion}_alpha"].append(alpha)
                for metric_class in metric_classes:
                    result = metric_class.compute(predictions, split_data, **params)
                    for key, value in result.items():
                        if metric_class.NAME == "fairness":
                            results[key].append(value)
                        else:
                            results[f"{criterion}_{key}"].append(value)
        for metric_class in metric_classes:
            metric_class.cleanup()
        
        for key, value in results.items():
            results[key] = np.array(value)
        
        self.result.update(results)
        return self.result

    def evaluate_rash(self, rash_model, hparams, split_data) -> None:
        """Evaluate all metrics per-tree with minimal overhead and profiling."""
        logger = logging.getLogger("Metrics.evaluate_rash")
        
        # Setup all metrics and determine what representations we need
        metric_instances = []
        needs_binary = False
        needs_continuous = False
        
        for metric_config in self.config["metrics"]:
            metric_class = get_metric(metric_config["name"])()
            params = {k: v for k, v in metric_config.items() if k != "name"}
            metric_class.setup(rash_model, hparams, split_data, **params)
            
            # Check if this metric requires binary features
            requires_binary = (metric_class.REQUIRES_BINARY_FEATURES and 
                             hasattr(split_data, 'binarizer') and split_data.binarizer is not None)
            
            metric_instances.append((metric_class, metric_config, requires_binary))
            
            # Track what representations we need
            needs_binary |= requires_binary
            needs_continuous |= not requires_binary
        
        eval_indices = rash_model.eval_indices
        final_result = defaultdict(list)
     
        if len(eval_indices) > 1_000_000:
            logging.warning(f"Evaluating {rash_model.get_num_models()} models. This may take a while, consider sub-sampling.")

        if "membership_inference" in self.metric_names or "stability" in self.metric_names:
            # These metrics are expensive to compute, so only evaluate on special indices
            eval_indices = rash_model.special_indices
            rash_model.eval_indices = eval_indices
            logger.info(f"Using special indices for expensive metrics: {len(eval_indices)} models to evaluate.")

        buffer = []
        for i, idx in enumerate(eval_indices):
            # Generate predictions only for needed representations
            predictions_cache = {}
            
            if needs_continuous:
                cont_model = rash_model.get_model(idx, use_binary_features=False)
                predictions_cache['continuous'] = self.get_predictions(
                    cont_model, split_data, use_binary_features=False, from_rash=True)
                    
            if needs_binary:
                bin_model = rash_model.get_model(idx, use_binary_features=True)
                predictions_cache['binary'] = self.get_predictions(
                    bin_model, split_data, use_binary_features=True, from_rash=True)

            result_row = {"model_idx": idx}
            for metric_class, cfg, requires_binary in metric_instances:
                pred_key = 'binary' if requires_binary else 'continuous'
                predictions = predictions_cache[pred_key]
                
                params = {k: v for k, v in cfg.items() if k != "name"}
                result = metric_class.compute(predictions, split_data, **params)
                result_row.update(result)
            
            buffer.append(result_row)
            if (i+1) % 10_000 == 0:
                logger.info(f"Evaluated {i} models...")

        for metric_class, _, _ in metric_instances:
            metric_class.cleanup()

        results = pd.DataFrame(buffer)
        for col in results.columns:
            if col == "model_idx":
                continue
            final_result[col] = results[col].to_numpy()
        self.result.update(final_result)

        return self.result
