"""experiment.py: Main experiment orchestration and execution logic.

This module provides the Experiment class which handles the complete lifecycle
of machine learning experiments including model training, hyperparameter tuning,
prediction, evaluation, and result persistence.
"""
import logging
import os
import pickle
import time
import json
import pandas as pd
import numpy as np
# from module.model import (RsetWrapper, GrootWrapper, RoctNWrapper, CartWrapper,
                            # RoctVWrapper, FprdtWrapper, PostRFWrapper, PostLogWrapper)
from module.model.core.post_base import PostBase
from module.model.fairness.dpf import DPF
from module.hparams import Hparams
from module.utils import NumpyEncoder, generate_stability_distribution
from module.datasets import DatasetLoader
from module.metric.metric import Metrics
from module.metric.lib.kantch.toolbox import KantchModel
from module.model.core.rash_base import RsetBase
class Experiment:
    """ Experiment class for running experiments with various models and metrics. """
    def __init__(self, dataset: DatasetLoader=None, params: Hparams=Hparams(),
                    metrics: Metrics=None):
        """ Constructor for the Experiment class.

        Args:
            dataset (DatasetLoader, optional): Dataset loader object. Defaults to None.
            params (Hparams, optional): Hyperparameters object. Defaults to Hparams().
            metrics (Metrics, optional): Metrics object. Defaults to None.
        """
        self.dataset = dataset
        self.params = params

        self.model = self.params.model_class(**self.params.model_params)

        self.metrics = metrics
        self.result = {}
        self.fold_idx = None

    #region Experiment pipeline
    def tuning(self, nested_cv, fold):
        """Generic tuning wrapper that delegates to model.tune when available.

        Args:
            nested_cv: Iterable of Split for inner CV (list or generator). Can be None.
            fold (int): The current outer fold index.
        """
        logger = logging.getLogger("experiment.tuning")
        logger.info("Parameter tuning (fold %s)", fold)

        # If no nested CV provided, skip tuning gracefully
        if nested_cv is None:
            logger.warning("nested_cv is None; skipping tuning.")
            return

        # Convert generators to list to avoid single-use pitfalls
        nested_cv = list(nested_cv)

        if not hasattr(self.model, "tune"):
            logger.info("Model has no tune() method; skipping tuning.")
            return


        # Use separate cache files per job for complete race-free operation
        cache_dir = os.path.join(
            self.params.io_params["output_dir"],
            self.params.io_params["param_dir"],
            "tuning_cache"
        )
        
        # Key for this tuning
        cache_key = (self.params.model_name, self.dataset.dataset_name, fold)

        # Check if we have a cached result for this specific key
        cached_result = self._load_cache_entry(cache_dir, cache_key)
        
        # Check if we should retune or use cached result
        should_tune = getattr(self.params, 'retune', False) or cached_result is None
        
        if should_tune:
            logger.info("Running tuning for %s (retune=%s, cached=%s)", 
                       cache_key, getattr(self.params, 'retune', False), cached_result is not None)
            result = self.model.tune(nested_cv)
            result_param = result[0] if isinstance(result, tuple) else result
            
            # Save to unique file (completely race-free)
            self._save_cache_entry(cache_dir, cache_key, result_param)
            logger.info("Saved tuning result to cache file")
        else:
            result_param = cached_result
            logger.info("Using cached tuning result for %s", cache_key)

        if not isinstance(result_param, dict):
            logger.warning("tune() did not return a dict of parameters; got %s", type(result_param))
            return

        # Merge tuned params into existing model_params conservatively
        def _merge(d, upd):
            for k, v in upd.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    _merge(d[k], v)
                else:
                    d[k] = v
        if isinstance(self.params.model_params, dict):
            _merge(self.params.model_params, result_param)
        else:
            self.params.model_params = result_param

        logger.info("Updated model params after tuning: %s", self.params.model_params)

    def training(self, data):
        """ Wrapper function to call train, process the data and log

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        # Use binary data if binarize_mode is enabled, otherwise use preprocessed continuous data
        logger = logging.getLogger("experiment.training")
        use_binarized_data = hasattr(data, 'binarizer') and data.binarizer is not None
        
        if use_binarized_data:
            binary_data = data.get_binary_data()
            X = binary_data['X_train']
        else:
            X = data.preprocess(data.X_train)
        y = data.y_train
        sens = data.sens_train
        logger.info("Training model with data shape: %s", X.shape)
        # If the model accepts a fold parameter, attach it.
        if isinstance(self.params.model_params, dict) and 'fold' in self.params.model_params:
            self.params.model_params['fold'] = self.fold_idx

        self.model = self.params.model_class(**self.params.model_params)
        
        # Set encoder AFTER creating model instance (for binarized data)
        if use_binarized_data and hasattr(self.model, 'encoder'):
            logger.info(f"Setting model.encoder to data.binarizer (type={type(data.binarizer).__name__})")
            self.model.encoder = data.binarizer
        else:
            logger.info(f"NOT setting encoder: use_binarized_data={use_binarized_data}, hasattr(encoder)={hasattr(self.model, 'encoder')}, binarizer={getattr(data, 'binarizer', 'NO_ATTR')}")
        
        start_time = time.time()
        # If DPF or Post-processing model
        if issubclass(self.params.model_class, PostBase) or issubclass(self.params.model_class, DPF):
            self.model.fit(X, y, sens)
        else:
            self.model.fit(X, y)
        end_time = time.time()
        
        # Use single-model fit time if available (for sweep mode fairness models)
        # This ensures fair comparison - we measure time for 1 model, not entire sweep
        if hasattr(self.model, '_single_fit_time'):
            self.result["train_time"] = self.model._single_fit_time
            logger.info(f"Using single model fit time: {self.model._single_fit_time:.4f}s (sweep trained {len(getattr(self.model, 'pareto_models', {}).get('sp', []))} models total)")
        else:
            self.result["train_time"] = end_time - start_time
        
        # Store special tree info and eval_indices for Rashomon set models
        if issubclass(self.params.model_class, RsetBase):
            self.model.post_train(self.params.rng)

    def selecting(self, data):
        """ Function to select the best tree from the Rashomon set.

        Args:
            data
        """
        logger = logging.getLogger("experiment.selecting")
        logger.info("Selecting best tree from Rashomon set")
        assert issubclass(self.params.model_class, RsetBase), "Only Rashomon set can be selected"

        # check if self.best_tree is not an empty dictionary
        if self.model.selected_model:
            logger.info("Best tree already selected: %s", self.model.selected_model)
            return
        
        metrics = []
        for metric in self.metrics.metric_names:
            if "accuracy" == metric:
                metrics.append("default")
            if "kantch_attack" == metric or "stability" == metric:
                metrics.append("kantch")
            if "fairness" == metric:
                metrics.append("fairness")

        if not metrics:
            logger.info("No valid metrics found for selection; skipping selection.")
            return

         # Prefer writing into configured output_dir; avoid hard-coded absolute paths.

        if "fairness" in metrics:
            assert data.sens_select is not None, "Sensitive features must be provided for fairness selection"
            self.metrics.sensitive_features = data.sens_select
            logger.info("Sensitive features (Selection): %s", np.unique(data.sens_select, return_counts=True))

        if "kantch" in metrics:
            ref_model = self.model.get_model()
            X_sel = data.preprocess(data.X_select)
            assert ref_model.n_features_in_ == X_sel.shape[1], \
                f"Model feature count {ref_model.n_features_in_} does not match selection data feature count {X_sel.shape[1]}."
            logger.info("Generating Kantch adversarial examples for selection")
            kantch_model = KantchModel.from_model(self.model.get_model())
            X_adv = kantch_model.adversarial_examples(data.preprocess(data.X_select), data.y_select, options={'epsilon': 0.1})
        else:
            X_adv = None
        
        if hasattr(data, 'binarizer') and data.binarizer is not None:
            logger.info("Binarizing data for selection")
            binary_data = data.get_binary_data()
            X = binary_data['X_select']
            if hasattr(self.model, 'encoder'):
                self.model.encoder = data.binarizer
        else:
            logger.info("Using preprocessed continuous data for selection")
            X = data.preprocess(data.X_select)

        self.model.selection(X, data.y_select, metrics, self.params.rng, X_adv, data.sens_select)

        self.metrics.sensitive_features = None
        return

    def testing(self, data):
        """ Function to test the model on the given test data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            X_test (np.ndarray): Test feature matrix.
            y_test (np.ndarray): Test target vector.
            X_test_sensitive (np.ndarray, optional): test set sensitive features. Defaults to None.
        """
        logger = logging.getLogger("experiment.testing")
        logger.info("Testing model")

        if self.params.reset_results:
            train_time = self.result.get("train_time", None)
            self.result = {"train_time": train_time}
            self.clean_file(self.result_path())

        result = self.metrics.evaluate(self.model, self.params, data)

        if issubclass(self.params.model_class, RsetBase):
            if hasattr(self.model, "special_tree"):
                self.result["special_tree_indices"] = dict(self.model.special_tree)
            if hasattr(self.model, "eval_indices"):
                self.result["eval_indices"] = self.model.eval_indices
            

        self.result.update(result)
        logger.info("Result: %s", json.dumps(self.result, indent=4, cls=NumpyEncoder))

    def cross_validate(self, train=True, test=True, fold=None) -> None:
        """ Function to perform cross-validation on the dataset.

        Args:
            train (bool, optional): Train the model. Defaults to True.
            test (bool, optional): Test the model. Defaults to True.
            fold (_type_, optional): Choose a specific fold to run. Defaults to None.
        """
        logger = logging.getLogger("experiment.cross_validate")
        select_size = 0.1 if self.params.selection else 0.0
        folds_data = self.dataset.kfold_splits(
            n_splits=getattr(self.params, 'k_folds', 5),
            inner_splits=getattr(self.params, 'inner_splits', None),
            select_size=select_size,
        )
        for data in folds_data:
            fold_idx = data.fold_id
            if fold is not None and fold_idx != fold:
                continue
            logger.info("Fold %s", fold_idx)

            self.fold_idx = fold_idx
            logger.info("Label ratio in train set: %s", np.unique(data.y_train, return_counts=True))

            if not self.params.retrain:
                logger.info("--retrain flag NOT set; will attempt to load existing model")
                trained = self.load_cross_val_model()
                if not trained:
                    logger.info("Failed to load existing model; setting retrain=True")
                    self.params.retrain = True
                else:
                    logger.info("Successfully loaded existing model; will skip training")
            else:
                logger.info("--retrain flag SET; will clean and retrain model")
                self.clean_cross_val_model()

            ### Tuning
            if self.params.tune and self.params.retrain:
                self.tuning(data.nested_cv, fold_idx)

            ### sample qf from stability distribution
            if self.params.noise_dist_qf == -1: # Stability metric is used but not initialized yet
                # If binarizer exists, use the binarized feature count
                num_feature = data.X_train.shape[1]
                if hasattr(data, 'binarizer') and data.binarizer is not None:
                    num_feature = data.binarizer.num_features

                self.params.noise_dist_qf = generate_stability_distribution(
                    self.params.rng, mean=self.params.model_params.get("noise_mean", 0.9), std=self.params.model_params.get("noise_std", 0.1), size=num_feature)
                if "qf" in self.params.model_params:
                    self.params.model_params["qf"] = self.params.noise_dist_qf

            ### Training
            if train and self.params.retrain:
                self.training(data)
                self.save_cross_val_model()
            else:
                logger.info("Model parameters: %s", self.params.model_params)

            logger.info("Model fitted time: %s", self.result["train_time"])

            ### Selection
            if self.params.selection and issubclass(self.params.model_class, RsetBase):
                self.selecting(data)
                self.save_cross_val_model()
            if issubclass(self.params.model_class, PostBase):
                logger.info("Post Processing the model")
                # Prepare data in the same format as training
                if hasattr(data, 'binarizer') and data.binarizer is not None:
                    logger.info("Binarizing selection data for post-processing")
                    binary_data = data.get_binary_data()
                    X_select = binary_data['X_select']
                else:
                    logger.info("Using preprocessed continuous data for post-processing")
                    X_select = data.preprocess(data.X_select)
                self.model.post_process(X_select, data.y_select, data.sens_select)

            if test:
                self.testing(data)
            
            self.save_cross_val_model()
    #endregion

    ###
    #region I/O Save/Load functions
    ###
    def get_state(self) -> dict:
        """ Function to get the current state of the experiment.

        Returns:
            dict: A dictionary containing the current state of the experiment.
        """
        return {
            "model": self.model,
            "param_state": self.params.get_state(),
            "result": self.result
        }

    def set_state(self, model, param_state, result) -> None:
        """ Function to set the current state of the experiment.

        Args:
            model (model_wrapper): Model wrapper object.
            param_state (hparams_state): Hyperparameters state.
            result (dict): Result dictionary.
        """
        # Preserve current command-line flags before restoring state
        current_retrain = self.params.retrain
        current_retune = self.params.retune
        current_tune = self.params.tune
        current_reset_results = self.params.reset_results
        
        self.params.set_state(param_state)
        
        # Restore command-line flags (don't use saved values)
        self.params.retrain = current_retrain
        self.params.retune = current_retune
        self.params.tune = current_tune
        self.params.reset_results = current_reset_results
        
        self.model = model
        self.result = result

    def filename(self) -> str:
        """ Function to get the filename for saving the experiment state.

        Returns:
            str: The filename for saving the experiment state.
        """
        return f"{self.params.model_name}_{self.dataset.dataset_name}_fold_{self.fold_idx}.pkl"

    def param_path(self) -> str:
        """ Function to get the parameter file path.
        Returns:
            str: The parameter file path.
        """
        return self.file_path("param_dir")

    def model_path(self):
        """ Function to get the model file path.
        Returns:
            str: The model file path.
        """
        return self.file_path("model_dir")

    def result_path(self) -> str:
        """ Function to get the result file path.

        Returns:
            str: The result file path.
        """
        return self.file_path("result_dir")

    def file_path(self, path_param: str) -> str:
        """

        Args:
            path_param (str): _description_

        Returns:
            str: _description_
        """
        return os.path.join(
                self.params.io_params["output_dir"],
                self.params.io_params[path_param],
                self.filename()
            )

    def save_cross_val_model(self) -> None:
        """ Function to save the cross-validation model. """
        logger = logging.getLogger("experiment.save_cross_val_model")
        logger.info("Saving cross-validation model to %s, %s, %s", self.model_path(), self.param_path(), self.result_path())
        state = self.get_state()
        with open(self.model_path(), 'wb') as file:
            pickle.dump(state["model"], file)
        with open(self.param_path(), 'wb') as file:
            pickle.dump(state["param_state"], file)
        with open(self.result_path(), 'wb') as file:
            pickle.dump(state["result"], file)

    def load_experiment(self, model_file, param_file, result_file) -> None:
        """ Function to load the experiment state from the given files.

        Args:
            model_file (str): Model file path.
            param_file (str): Parameter file path.
            result_file (str): Result file path.
        """
        logger = logging.getLogger("experiment.load_experiment")
        logger.info("Loading experiment")
        with open(model_file, 'rb') as file:
            model_state = pickle.load(file)
        with open(param_file, 'rb') as file:
            param_state = pickle.load(file)
        with open(result_file, 'rb') as file:
            result_state = pickle.load(file)
        self.set_state(model_state, param_state, result_state)

    def load_cross_val_model(self) -> bool:
        """ Function to load the cross-validation model.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        logger = logging.getLogger("experiment.load_cross_val_model")
        
        model_file = self.model_path()
        param_file = self.param_path()
        result_file = self.result_path()
        
        logger.info("Attempting to load model from:")
        logger.info("  Model: %s", model_file)
        logger.info("  Param: %s", param_file)
        logger.info("  Result: %s", result_file)
        
        # Check which files exist
        model_exists = os.path.exists(model_file)
        param_exists = os.path.exists(param_file)
        result_exists = os.path.exists(result_file)
        
        logger.info("File existence: Model=%s, Param=%s, Result=%s", 
                   model_exists, param_exists, result_exists)
        
        if not (model_exists and param_exists and result_exists):
            logger.info("Not all required files exist - will need to retrain")
            return False
        
        try:
            with open(model_file, 'rb') as file:
                model_state = pickle.load(file)
            with open(param_file, 'rb') as file:
                param_state = pickle.load(file)
            with open(result_file, 'rb') as file:
                result_state = pickle.load(file)
        except FileNotFoundError as e:
            logger.warning("Model file not found: %s", e)
            return False
        except Exception as e:
            logger.error("Error loading model files: %s", e)
            logger.warning("Will retrain due to load error")
            return False
            
        self.set_state(model_state, param_state, result_state)
        logger.info("Successfully loaded existing model!")
        return True

    def clean_cross_val_model(self) -> None:
        """ Function to clean the cross-validation model files. """
        for filename in [self.file_path("model_dir"),
                         self.file_path("param_dir"),
                         self.file_path("result_dir")]:
            self.clean_file(filename)

    def clean_file(self, filename: str) -> None:
        """ Function to clean a file if it exists.

        Args:
            filename (str): The name of the file to be cleaned.
        """
        logger = logging.getLogger("experiment.clean_file")
        logger.info("Cleaning experiment file %s", filename)
        if os.path.exists(filename):
            os.remove(filename)

    def _load_cache_entry(self, cache_dir: str, cache_key: tuple):
        """ Load a specific cache entry if it exists.
        
        Args:
            cache_dir (str): Directory containing cache files.
            cache_key (tuple): The specific cache key to load.
            
        Returns:
            The cached result_param if found, None otherwise.
        """
        logger = logging.getLogger("experiment._load_cache_entry")
        
        if not os.path.exists(cache_dir):
            logger.debug("Cache directory does not exist: %s", cache_dir)
            return None
        
        # Build the expected filename for this specific cache key
        model_name, dataset_name, fold = cache_key
        filename = f"cache_{model_name}_{dataset_name}_fold{fold}.pkl"
        cache_path = os.path.join(cache_dir, filename)
        
        if not os.path.exists(cache_path):
            logger.debug("Cache file does not exist: %s", cache_path)
            return None
        
        # Load the specific cache file
        try:
            with open(cache_path, 'rb') as f:
                entry = pickle.load(f)
                if len(entry) == 3:
                    timestamp, stored_key, result_param = entry
                    logger.debug("Loaded cached result for %s from %s (timestamp: %s)", 
                               cache_key, cache_path, timestamp)
                    return result_param
                else:
                    logger.warning("Invalid cache entry in %s: %s", cache_path, entry)
                    return None
                    
        except Exception as e:
            logger.warning("Error reading cache file %s: %s", cache_path, e)
            return None

    def _save_cache_entry(self, cache_dir: str, cache_key: tuple, result_param) -> None:
        """ Save cache entry to a unique file (race-free).
        
        Args:
            cache_dir (str): Directory to save cache files.
            cache_key (tuple): Key for the cache entry.
            result_param: Parameter values to cache.
        """
        logger = logging.getLogger("experiment._save_cache_entry")
        
        # Ensure directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # Create filename using just the cache key components
            model_name, dataset_name, fold = cache_key
            filename = f"cache_{model_name}_{dataset_name}_fold{fold}.pkl"
            cache_path = os.path.join(cache_dir, filename)
            
            # Save entry with timestamp for record keeping
            entry = (time.strftime("%Y-%m-%d %H:%M:%S"), cache_key, result_param)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
                
            logger.debug("Saved cache entry to %s", cache_path)
            
        except Exception as e:
            logger.error("Failed to save cache entry: %s", e)
            raise

    #endregion
