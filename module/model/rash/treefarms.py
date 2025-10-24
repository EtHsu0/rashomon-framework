"""
This module provides a wrapper around the TREEFARMS model for easier interaction and evaluation.
"""
import logging
from collections import defaultdict
import os, json

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from treefarms import TREEFARMS
from gosdt import GOSDTClassifier

from module.utils import NumpyEncoder
from module.hparams import register_hparams
from module.model.core.rash_base import RsetBase
from module.model.core.base import register_model
from module.model.lib.tree_classifier import TreeClassifier
from module.threshold_guess import is_binary_matrix, ThresholdGuess
from module.metric.fairness_metric import statistical_parity_difference, equal_opportunity_difference, equalized_odds_difference

@register_model("treefarms")
class TreefarmsWrapper(RsetBase):
    def __init__(self, **kwargs):
        """ Initialize the TreefarmsWrapper with the given configuration.

        Args:
            config (dict): Configuration for the TREEFARMS model.
        """
        self.eval_size = kwargs.pop('eval_size')
        self.model = TREEFARMS(kwargs)
        self.optimal = GOSDTClassifier()
        self.best_tree = defaultdict(None)
        self.best_tree_score = defaultdict(None)

        self.special_tree = {}
        self.n_features_ = None
        self.classes_ = None
        self.encoder = None
        self.eval_indices = None
        self.special_indices = []
        self.selected_model = {}
    
    def binarize(self, X, y):
        self.encoder = ThresholdGuess({'n_estimators': 30, 'max_depth': 2, 'learning_rate': 0.1}, back_select=False)
        self.encoder.fit(X, y)
        return self.encoder.transform(X)

    def fit(self, X, y):
        """ Fit the TREEFARMS model to the data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        logger = logging.getLogger("TreefarmsWrapper.fit")
        # Binarize if needed: either continuous data OR too many features (>30)
        if not is_binary_matrix(X) or X.shape[1] > 30:
            X = self.binarize(X, y)
        
        self.model.fit(pd.DataFrame(X), pd.Series(y))
        self.best_tree = {}
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        logger.info("TreeFARMS generated with %s trees", self.get_num_models())
    
    def predict(self, X, idx=None):
        """ Predict using the model at the given index.

        Args:
            X (np.ndarray): Feature matrix.
            idx (int, optional): Index of the model to use for prediction. Defaults to None.

        Returns:
            np.ndarray: Predicted labels.
        """
        model = self.get_model()
        return model.predict(X)

    def get_model(self, idx=None, use_binary_features=False):
        """ Get the model at the given index.
        
        Args:
            idx (int, optional): Index of the model to use. Defaults to None.
            use_binary_features (bool): If True and encoder exists, return tree with binary splits.
                                      If False, return tree with continuous splits (default behavior).
        """
        if idx is None:
            optimal_indices = self.get_optimal_tree_idx()
            if not optimal_indices:
                raise ValueError("No optimal tree found in Rashomon set.")
            idx = optimal_indices[0]
        
        # If requesting binary features and we have an encoder, build binary tree
        if use_binary_features and self.encoder is not None:
            return self._build_binary_tree_classifier(idx)
        else:
            # Default behavior: return continuous tree (or binary tree if no encoder)
            # Determine n_features based on encoder presence:
            # - If encoder exists: use encoder's original feature count (for continuous input)
            # - If no encoder: use self.n_features_ (data was already binary)
            if self.encoder is not None and hasattr(self.encoder, 'n_features_in_') and self.encoder.n_features_in_ is not None:
                n_features = self.encoder.n_features_in_  # Original continuous features
            else:
                n_features = self.n_features_  # Binary features (no encoder)
            
            return build_tree_classifier_from_eq_ref_json(
                self.model.model_set.get_tree_at_idx_raw(idx), 
                encoder=self.encoder,
                n_features=n_features)
    
    def _build_binary_tree_classifier(self, idx: int) -> TreeClassifier:
        """
        Build a TreeClassifier that operates directly on binary features.
        This tree uses the binary feature indices directly without converting back to continuous splits.
        """
        return build_tree_classifier_from_eq_ref_json(
            self.model.model_set.get_tree_at_idx_raw(idx), 
            encoder=None,  # No encoder = binary features
            n_features=self.n_features_  # Pass number of binary features from training
        )
        
    def get_num_models(self) -> int:
        return self.model.model_set.model_count
    
    def get_eval_tree_idx(self) -> list:
        """ Get the indices of all evaluation trees.
        This function returns a list of indices for the evaluation trees, including
        the optimal tree, the minimum leaf tree, and the maximum leaf tree.
        Also includes special trees selected by different metrics.

        Returns:
            list: List of indices of evaluation trees.
        """
        indices = []
        for name, idx in self.special_tree.items():
            indices.append([name, idx])
        for metric, idx in self.best_tree.items():
            indices.append([metric, idx])
        return indices

    def selection(self, X, y, metrics, rng, X_adv=None, sens_feat=None):
        logger = logging.getLogger("Treefarms Selection")
        num_models = self.get_num_models()
        
        indices = self.eval_indices
        selected_model = dict()

        if is_binary_matrix(X):
            use_binary = True
        else:
            use_binary = False

        for i, idx in enumerate(indices):
            tree = self.get_model(idx, use_binary_features=use_binary)
            preds = tree.predict(X)

            if "kantch" in metrics and X_adv is not None:
                adv_tree = self.get_model(idx, use_binary_features=False) # Kantch uses continuous features for attacks
                adv_preds = adv_tree.predict(X_adv)
                adv_acc = accuracy_score(y, adv_preds)
                acc = accuracy_score(y, preds)
                # For kantch metric, find model with highest adversarial accuracy, using regular accuracy as tie-breaker
                if ("kantch" not in selected_model or 
                    adv_acc > selected_model["kantch"][1] or 
                    (adv_acc == selected_model["kantch"][1] and acc > selected_model["kantch"][2])):
                    selected_model["kantch"] = (idx, adv_acc, acc)

            if "fairness" in metrics and sens_feat is not None:
                spd = statistical_parity_difference(y, preds, sens_feat)
                eod = equal_opportunity_difference(y, preds, sens_feat)
                eods = equalized_odds_difference(y, preds, sens_feat)
                acc = accuracy_score(y, preds)
                # For each metric, find fairest model with accuracy as tie-breaker
                # Statistical Parity Difference
                if ("fairness_sp" not in selected_model or 
                    spd < selected_model["fairness_sp"][1] or 
                    (spd == selected_model["fairness_sp"][1] and acc > selected_model["fairness_sp"][2])):
                    selected_model["fairness_sp"] = (idx, spd, acc)
                # Equal Opportunity Difference
                if ("fairness_eopp" not in selected_model or 
                    eod < selected_model["fairness_eopp"][1] or 
                    (eod == selected_model["fairness_eopp"][1] and acc > selected_model["fairness_eopp"][2])):
                    selected_model["fairness_eopp"] = (idx, eod, acc)
                # Equalized Odds Difference
                if ("fairness_eo" not in selected_model or 
                    eods < selected_model["fairness_eo"][1] or 
                    (eods == selected_model["fairness_eo"][1] and acc > selected_model["fairness_eo"][2])):
                    selected_model["fairness_eo"] = (idx, eods, acc)
            if i % 10_000 == 0 and i > 0:
                logger.info(f"Selection progress: {i}/{len(indices)}")

        self.selected_model = selected_model
        logger.info(json.dumps(selected_model, indent=2, cls=NumpyEncoder))
        ## Add selected indices to special indices
        for name, model_info in selected_model.items():
            idx = model_info[0]  # Extract index from tuple (could be 2 or 3 elements)
            if name not in self.special_tree:
                self.special_tree[name] = idx
                self.special_indices.append(idx)

    def find_special_tree(self, rng, num_random_samples=3) -> list:
        """
        Find special trees based on different criteria.
        
        Creates diverse special trees to avoid overfitting:
        - 1 optimal tree (random from optimal set)
        - 1 min_leaf_optimal_tree (best training objective among min-leaf trees)
        - 1 max_leaf_optimal_tree (best training objective among max-leaf trees)
        - Up to N min_leaf_optimal_tree_i (if many optimal min-leaf trees exist after tie-breaking)
        - Up to N max_leaf_optimal_tree_i (if many optimal max-leaf trees exist after tie-breaking)
        - Up to N min_leaf_tree_i (random samples from all min-leaf trees, for diversity)

        The total number of special trees is dynamic:
        - Minimum: 3 (optimal, min_leaf_optimal, max_leaf_optimal)
        - Maximum: 3 + 3*N (if all categories have enough trees for sampling)

        Args:
            rng (generator): Random number generator.
            num_random_samples (int): Number of random samples per category (default: 5).

        Returns:
            list: List of selected special tree indices.
        Raises:
            ValueError: If any of the required tree index lists are empty.
        """
        logger = logging.getLogger("TreefarmsWrapper.find_special_tree")
        logger.info("Finding special trees with diversity sampling")
        self.special_tree = {}
        
        # Get candidate lists
        optimal_list = self.get_optimal_tree_idx()
        min_leaf_list = self.get_min_leaf_tree_idx()
        
        if not optimal_list:
            raise ValueError("No optimal tree found in Rashomon set.")
        if not min_leaf_list:
            raise ValueError("No min-leaf tree found in Rashomon set.")
        
        logger.info(f"Found {len(optimal_list)} optimal trees, {len(min_leaf_list)} min-leaf trees")
        
        # 1. Select one optimal tree (random choice, already optimal by definition)
        self.special_tree["optimal_tree"] = rng.choice(optimal_list)
        
        # 2. Select training-objective-optimized min/max leaf trees
        # Use tie-breaking based on training objective (no need to recalculate accuracy)
        min_leaf_optimal_list = self.get_min_leaf_optimal_tree_idx()
        max_leaf_optimal_list = self.get_max_leaf_optimal_tree_idx()
        
        logger.info(f"After tie-breaking: {len(min_leaf_optimal_list)} optimal min-leaf trees, {len(max_leaf_optimal_list)} optimal max-leaf trees")
        
        # 2a. Select primary optimal trees (one from each category)
        self.special_tree["min_leaf_optimal_tree"] = rng.choice(min_leaf_optimal_list)
        self.special_tree["max_leaf_optimal_tree"] = rng.choice(max_leaf_optimal_list)
        
        # 2b. If there are multiple optimal trees after tie-breaking, sample more for diversity
        # This gives us better representative coverage of the optimal set
        actual_min_optimal_samples = min(num_random_samples, len(min_leaf_optimal_list))
        if actual_min_optimal_samples > 1:
            available_min_optimal = [idx for idx in min_leaf_optimal_list if idx != self.special_tree["min_leaf_optimal_tree"]]
            if len(available_min_optimal) > 0:
                sample_size = min(num_random_samples, len(available_min_optimal))
                sampled_min_optimal = rng.choice(available_min_optimal, size=sample_size, replace=False)
                for i, idx in enumerate(sampled_min_optimal):
                    self.special_tree[f"min_leaf_optimal_tree_{i}"] = idx
                logger.info(f"Sampled {len(sampled_min_optimal)} additional optimal min-leaf trees for diversity")
        
        actual_max_optimal_samples = min(num_random_samples, len(max_leaf_optimal_list))
        if actual_max_optimal_samples > 1:
            available_max_optimal = [idx for idx in max_leaf_optimal_list if idx != self.special_tree["max_leaf_optimal_tree"]]
            if len(available_max_optimal) > 0:
                sample_size = min(num_random_samples, len(available_max_optimal))
                sampled_max_optimal = rng.choice(available_max_optimal, size=sample_size, replace=False)
                for i, idx in enumerate(sampled_max_optimal):
                    self.special_tree[f"max_leaf_optimal_tree_{i}"] = idx
                logger.info(f"Sampled {len(sampled_max_optimal)} additional optimal max-leaf trees for diversity")
        
        # 3. Select random min-leaf trees (for diversity)
        actual_min_samples = min(num_random_samples, len(min_leaf_list))
        if actual_min_samples > 0:
            # Sample without replacement, ensuring we don't duplicate the optimal tree
            available_min = [idx for idx in min_leaf_list if idx != self.special_tree["min_leaf_optimal_tree"]]
            if len(available_min) > 0:
                sample_size = min(actual_min_samples, len(available_min))
                sampled_min = rng.choice(available_min, size=sample_size, replace=False)
                for i, idx in enumerate(sampled_min):
                    self.special_tree[f"min_leaf_tree_{i}"] = idx
            else:
                logger.warning("Only one min-leaf tree available, cannot create diverse samples")
        
        # Update special_indices with all selected trees
        self.special_indices = list(self.special_tree.values())
        
        logger.info(f"Selected {len(self.special_tree)} special trees:")
        for name, idx in self.special_tree.items():
            logger.info(f"  {name}: {idx}")
        
        return self.special_indices
    
    def get_raw_idx_from_pointer_idx(self, pointer: str, i: int) -> int:
        """ Get the raw index from the pointer index.
        This function calculates the raw index of a tree based on its pointer
        and index within that pointer.

        Args:
            pointer (str): Pointer to the model set.
            i (int): Index within the model set.

        Returns:
            int: Raw index of the tree.
        """
        model_set = self.model.model_set
        count = 0
        for entry in model_set.available_metrics["metric_pointers"]:
            if entry == pointer:
                break
            count += model_set.get_model_set(entry)["count"]
        return count + i

    def get_optimal_tree_idx(self) -> list:
        """ Get the optimal tree index.
        This function finds the index of the optimal tree based on the minimum objective value
        and returns the raw indices of the trees that meet this criterion.

        Returns:
            list: List of raw indices of the optimal trees.
        """
        model_set = self.model.model_set
        metric_values = np.array(model_set.available_metrics["metric_values"])
        if metric_values.shape[0] == 0:
            logger.error("No metric values found in model set for optimal tree selection")
            return []
        min_obj_idx = np.argmin(metric_values[:,0])
        pointer = model_set.available_metrics["metric_pointers"][min_obj_idx]
        optimal_tree_model_set = model_set.storage[pointer]
        count = optimal_tree_model_set["count"]
        raw_idx = []
        logger = logging.getLogger("TreefarmsWrapper.get_optimal_tree_idx")
        for i in range(count):
            raw_idx.append(self.get_raw_idx_from_pointer_idx(pointer, i))
        return raw_idx
    
    def get_min_leaf_tree_idx(self, max_per_pointer=10000):
        """ Get the minimum leaf tree index.
        
        Returns ALL trees with minimum number of leaves (no tie-breaking).
        If a pointer has too many trees (> max_per_pointer), samples uniformly from it.
        This allows for diverse selection of min-leaf trees without expensive loops.

        Args:
            max_per_pointer (int): Maximum trees to sample from each pointer. Default: 10000.

        Returns:
            list: List of raw indices of minimum leaf trees (all or sampled per pointer).
        """
        logger = logging.getLogger("TreefarmsWrapper.get_min_leaf_tree_idx")
        model_set = self.model.model_set
        metric_values = np.array(model_set.available_metrics["metric_values"])
        if metric_values.shape[0] == 0:
            return []
        
        # Find all trees with minimum leaf count (no tie-breaking)
        min_leaf_count = min(metric_values[:,-1])
        min_leaf_indices = np.where(metric_values[:,-1] == min_leaf_count)[0]
        
        if len(min_leaf_indices) == 0:
            return []
                
        # Collect raw indices, sampling large pointers
        raw_idx = []
        for idx in min_leaf_indices:
            pointer = model_set.available_metrics["metric_pointers"][idx]
            min_leaf_tree_model_set = model_set.storage[pointer]
            count = min_leaf_tree_model_set["count"]
            
            
            if count <= max_per_pointer:
                # Small enough - add all trees
                for i in range(count):
                    raw_idx.append(self.get_raw_idx_from_pointer_idx(pointer, i))
            else:
                # Too large - sample uniformly
                logger.warning(f"Pointer '{pointer}' has {count} trees (> {max_per_pointer}), sampling")
                sampled_indices = np.random.choice(count, size=max_per_pointer, replace=False)
                for i in sampled_indices:
                    raw_idx.append(self.get_raw_idx_from_pointer_idx(pointer, i))
        
        return raw_idx
    
    def get_min_leaf_optimal_tree_idx(self, max_per_pointer=10000):
        """ Get the optimal minimum leaf tree index using training objective tie-breaking.
        
        Among all trees with minimum leaves, selects those with best training objective.
        This uses the trained model's objective value (metric_values[:,1]) for tie-breaking,
        avoiding the need to recalculate predictions.
        If the optimal pointer has too many trees (> max_per_pointer), samples uniformly from it.

        Args:
            max_per_pointer (int): Maximum trees to sample from the optimal pointer. Default: 10000.

        Returns:
            list: List of raw indices of optimal minimum leaf trees (all or sampled if too many).
        """
        logger = logging.getLogger("TreefarmsWrapper.get_min_leaf_optimal_tree_idx")
        model_set = self.model.model_set
        metric_values = np.array(model_set.available_metrics["metric_values"])
        if metric_values.shape[0] == 0:
            return []
        
        # Find all trees with minimum leaf count
        min_leaf_indices = np.where(metric_values[:,-1] == min(metric_values[:,-1]))[0]
        if len(min_leaf_indices) == 0:
            logger.error("No minimum leaf trees found in model set for min-leaf optimal tree selection")
            return []
        
        # Break ties using training objective (lower is better)
        if len(min_leaf_indices) > 1:
            best_obj_idx = min_leaf_indices[np.argmin(metric_values[min_leaf_indices, 1])]
        else:
            best_obj_idx = min_leaf_indices[0]
        
        # Get all trees at this pointer (may still be multiple trees)
        pointer = model_set.available_metrics["metric_pointers"][best_obj_idx]
        optimal_tree_model_set = model_set.storage[pointer]
        count = optimal_tree_model_set["count"]
        raw_idx = []
        
        if count <= max_per_pointer:
            # Small enough - add all trees
            for i in range(count):
                raw_idx.append(self.get_raw_idx_from_pointer_idx(pointer, i))
        else:
            # Too large - sample uniformly
            logger.warning(f"Pointer '{pointer}' has {count} optimal min-leaf trees (> {max_per_pointer}), sampling")
            sampled_indices = np.random.choice(count, size=max_per_pointer, replace=False)
            for i in sampled_indices:
                raw_idx.append(self.get_raw_idx_from_pointer_idx(pointer, i))
        
        return raw_idx

    def _select_best_accuracy_from_tree_list(self, tree_indices, X, y, rng):
        """ Select the tree with highest accuracy from a list of tree indices.
        
        Args:
            tree_indices (list): List of tree indices to evaluate.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            rng (generator): Random number generator for ties.
            
        Returns:
            int: Index of the tree with highest accuracy.
        """
        if len(tree_indices) == 1:
            return tree_indices[0]
            
        logger = logging.getLogger("TreefarmsWrapper._select_best_accuracy")
        best_acc = -1
        best_indices = []
        
        for idx in tree_indices:
            # Transform X if needed
            if not is_binary_matrix(X) and self.encoder is not None:
                X_transformed = self.encoder.transform(X)
                use_binary = False
            else:
                X_transformed = X
                use_binary = True
            
            logger.info("Use binary features: %s", use_binary)

            tree = self.get_model(idx, use_binary_features=use_binary)
            preds = tree.predict(X_transformed)
            acc = accuracy_score(y, preds)
            
            if acc > best_acc:
                best_acc = acc
                best_indices = [idx]
            elif acc == best_acc:
                best_indices.append(idx)
        
        # If we have ties in accuracy, randomly select among them
        if best_indices:
            selected = rng.choice(best_indices)
            logger.debug(f"Selected tree {selected} with accuracy {best_acc:.4f} from {len(tree_indices)} candidates")
            return selected
        else:
            # Fallback to random selection if all failed
            logger.warning("All tree evaluations failed, using random selection")
            return rng.choice(tree_indices)
    
    def get_max_leaf_optimal_tree_idx(self, max_per_pointer=10000):
        """ Get the optimal maximum leaf tree index using training objective tie-breaking.
        
        Among all trees with maximum leaves, selects those with best training objective.
        This uses the trained model's objective value (metric_values[:,1]) for tie-breaking,
        avoiding the need to recalculate predictions.
        If the optimal pointer has too many trees (> max_per_pointer), samples uniformly from it.

        Args:
            max_per_pointer (int): Maximum trees to sample from the optimal pointer. Default: 10000.

        Returns:
            list: List of raw indices of optimal maximum leaf trees (all or sampled if too many).
        """
        logger = logging.getLogger("TreefarmsWrapper.get_max_leaf_optimal_tree_idx")
        model_set = self.model.model_set
        metric_values = np.array(model_set.available_metrics["metric_values"])
        if metric_values.shape[0] == 0:
            return []
        
        # Find all trees with maximum leaf count
        max_leaf_indices = np.where(metric_values[:,-1] == max(metric_values[:,-1]))[0]
        if len(max_leaf_indices) == 0:
            return []
        
        # Break ties using training objective (lower is better)
        if len(max_leaf_indices) > 1:
            best_obj_idx = max_leaf_indices[np.argmin(metric_values[max_leaf_indices, 1])]
        else:
            best_obj_idx = max_leaf_indices[0]
        
        # Get all trees at this pointer (may still be multiple trees)
        pointer = model_set.available_metrics["metric_pointers"][best_obj_idx]
        optimal_tree_model_set = model_set.storage[pointer]
        count = optimal_tree_model_set["count"]
        raw_idx = []
        
        if count <= max_per_pointer:
            # Small enough - add all trees
            for i in range(count):
                raw_idx.append(self.get_raw_idx_from_pointer_idx(pointer, i))
        else:
            # Too large - sample uniformly
            logger.warning(f"Pointer '{pointer}' has {count} optimal max-leaf trees (> {max_per_pointer}), sampling")
            sampled_indices = np.random.choice(count, size=max_per_pointer, replace=False)
            for i in sampled_indices:
                raw_idx.append(self.get_raw_idx_from_pointer_idx(pointer, i))
        
        return raw_idx

    def post_train(self, rng):
        # Use accuracy-based selection if data is provided, otherwise random selection
        logger = logging.getLogger("TreefarmsWrapper.post_train")
        logger.info("Post-training: selecting evaluation indices")
        self.find_special_tree(rng, 0)
        num_models = self.get_num_models()
        
        sample_size = min(self.eval_size, num_models)
        if num_models > sample_size:
            indices = set(rng.choice(num_models, size=sample_size, replace=False))
        else:
            indices = set(np.arange(num_models))


        indices.update(self.special_indices)
        indices = np.array(list(indices))
        self.eval_indices = indices

    def tune(self, nested_cv) -> dict:
        """ Tune the TreeFARMS model using nested cross-validation.
        
        TreeFARMS generates a Rashomon set of decision trees. We tune:
        - depth_budget: Maximum depth of trees
        - regularization (lambda): Regularization strength for objective function
        
        Note: rashomon_bound_adder is NOT tuned as it's a TreeFARMS-specific parameter
        for controlling the Rashomon set size, not a tree quality parameter.
        
        Args:
            nested_cv: Iterable of Split objects for inner CV

        Returns:
            tuple: Best configuration and all scores.
        """
        logger = logging.getLogger("TreefarmsWrapper.tune")
        
        # Hyperparameter grid based on original commented code
        depth_budgets = [4, 5]  # Tree depth range
        regularizations = [0.01, 0.015, 0.02, 0.025]  # Lambda values
        
        nested_cv = list(nested_cv)
        best_score = -float('inf')
        best_config = None
        all_scores = {}
        
        # Grid search over hyperparameters
        for depth_budget in depth_budgets:
            for lamb in regularizations:
                logger.debug(f"Depth Budget: {depth_budget}, Lambda: {lamb}")
                
                config = {
                    'depth_budget': depth_budget,
                    'regularization': lamb,
                    'allow_small_reg': True,
                }
                
                scores = []
                for split in nested_cv:
                    # Binarize if needed: either continuous data OR too many features (>30)
                    if not is_binary_matrix(split.X_train) or split.X_train.shape[1] > 30:
                        encoder = ThresholdGuess(
                            {'n_estimators': 30, 'max_depth': 2, 'learning_rate': 0.1}, 
                            back_select=False
                        )
                        encoder.fit(split.X_train, split.y_train)
                        X_train_bin = encoder.transform(split.X_train)
                        X_val_bin = encoder.transform(split.X_test)
                    else:
                        # Data is already binary with ≤30 features, use directly
                        X_train_bin = split.X_train
                        X_val_bin = split.X_test
                    
                    # Use GOSDTClassifier for evaluation (TreeFARMS uses it internally)
                    gosdt_model = GOSDTClassifier(**config)
                    gosdt_model.fit(pd.DataFrame(X_train_bin), pd.Series(split.y_train))
                    
                    score = gosdt_model.score(pd.DataFrame(X_val_bin), pd.Series(split.y_test))
                    scores.append(score)
                    logger.debug(f"Fold {split.fold_id} Score: {score}")
                
                avg_score = np.mean(scores)
                config_key = (depth_budget, lamb)
                all_scores[config_key] = scores
                
                logger.info(f"Config {config_key}: Avg Score = {avg_score:.4f}")
                
                # Prefer larger depth when scores tie, smaller lambda is fine (loop order handles it)
                if (avg_score > best_score or 
                    (avg_score == best_score and best_config is not None and depth_budget > best_config['depth_budget'])):
                    best_score = avg_score
                    best_config = {
                        'depth_budget': depth_budget,
                        'regularization': lamb,
                    }
        
        logger.info(f"Best Config: {best_config} with score: {best_score:.4f}")
        logger.info(f"Evaluated {len(all_scores)} configurations")
        return best_config, all_scores

@register_hparams("treefarms")
def update_hparams(hparams, args, _dataset):
    """ Update hparams with TREEFARMS-specific parameters. """
    hparams.model_params = {
        'depth_budget': int(args.depth_budget) if hasattr(args, 'depth_budget') else 5,
        'regularization': float(args.regularization) if hasattr(args, 'regularization') else 0.01,
        'rashomon_bound_adder': float(args.rashomon_bound_adder) if hasattr(args, 'rashomon_bound_adder') else 0.05,
        'alpha': float(args.alpha) if hasattr(args, 'alpha') else 0.03,
        'eval_size': int(args.eval_size) if hasattr(args, 'eval_size') else 200_000,
    }
    logger = logging.getLogger("Treefarms HParams")
    logger.info("Updated hparams for TREEFARMS: %s", hparams.model_params)


def build_tree_classifier_from_eq_ref_json(obj, *, classes=(0, 1), encoder=None, n_features=None) -> TreeClassifier:
    """
    Build a TreeClassifier from nodes like:
      {"feature": 9, "relation": "==", "reference": "true", "true": {...}, "false": {...}}
    Leaves look like {"prediction": 0, ...}. Extra keys are ignored.

    Args:
        obj: Tree JSON object from TreeFARMS
        classes: Class labels
        encoder: Optional encoder (ThresholdGuess) for mapping binary -> continuous features
        n_features: Number of expected input features (used when tree uses subset of features)

    Semantics (without encoder):
      - Treat X[:, feature] as binary {0,1}. Threshold = 0.5.
      - 'true' branch is taken when (X_bin == reference_bit).
      - Left child = values <= threshold; Right child = > threshold (i.e., 0 -> left, 1 -> right).

    Semantics (with encoder, e.g., ThresholdGuess):
      - encoder.thresholds is a list of (orig_feature_index, threshold) for each boolean feature.
      - Let b = 1 if (X[:, orig_feature_index] <= threshold) else 0.
      - 'true' branch is taken when (b == reference_bit).
      - We emit a direct numeric split on (orig_feature_index, threshold).
        * Left child means X <= threshold; Right child means X > threshold.
        * If reference_bit == 1  => 'true' is LEFT   (<= threshold)
          reference_bit == 0  => 'true' is RIGHT  (> threshold)
    """

    # ---- load dict ----
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8")
    if isinstance(obj, str):
        if os.path.exists(obj):          # file path
            with open(obj, "r", encoding="utf-8") as f:
                root = json.load(f)
        else:                             # JSON string
            root = json.loads(obj)
    elif isinstance(obj, dict):
        root = obj
    else:
        raise TypeError(f"Unsupported input type: {type(obj).__name__}")

    # ---- helpers ----
    def _is_leaf(n):
        return "prediction" in n

    def _ref_to_bit(v):
        if isinstance(v, bool): return 1 if v else 0
        if isinstance(v, (int, np.integer)): return 1 if int(v) != 0 else 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "yes", "y", "t"):  return 1
            if s in ("false", "0", "no", "n", "f"):   return 0
        raise ValueError(f"Unsupported 'reference' value: {v!r}")

    # ---- build arrays (preorder; parent index < child indices) ----
    ch_left, ch_right, feat, thr = [], [], [], []
    leaf_vecs = []
    C = len(classes)

    stack = [(root, -1, False)]  # (node_dict, parent_idx, is_left)
    while stack:
        node, parent, is_left = stack.pop()
        i = len(ch_left)

        ch_left.append(-1)
        ch_right.append(-1)
        feat.append(-2)   # -2 for leaves
        thr.append(-2.0)  # ignored at leaves
        leaf_vecs.append(None)

        if _is_leaf(node):
            y = int(node["prediction"])
            v = np.zeros(C, dtype=float)
            v[y if 0 <= y < C else 0] = 1.0
            leaf_vecs[i] = v

        else:
            rel = node.get("relation", "==")
            if rel != "==":
                raise NotImplementedError(f"Only '==' relation is supported, got {rel!r}")

            ref_bit = _ref_to_bit(node.get("reference", True))
            node_feat = int(node["feature"])

            if encoder is None:
                # Binary feature; threshold at 0.5
                feat[i] = node_feat
                thr[i] = 0.5
                # Map children so that (X_bin == ref_bit) follows 'true'
                # 0 -> left, 1 -> right
                if ref_bit == 1:
                    left_child, right_child = node["false"], node["true"]
                else:  # ref_bit == 0
                    left_child, right_child = node["true"], node["false"]
            else:
                # Use continuous threshold from encoder
                if not hasattr(encoder, "thresholds") or encoder.thresholds is None:
                    raise ValueError("encoder.thresholds is missing; call encoder.fit(...) first.")

                if not (0 <= node_feat < len(encoder.thresholds)):
                    raise IndexError(
                        f"Node feature index {node_feat} out of range for encoder.thresholds (len={len(encoder.thresholds)})."
                    )
                orig_f, t = encoder.thresholds[node_feat]
                feat[i] = int(orig_f)
                thr[i] = float(t)

                # Left means X <= t, Right means X > t
                # 'true' when (1 if X<=t else 0) == ref_bit
                if ref_bit == 1:
                    # 'true' must be LEFT (<= t)
                    left_child, right_child = node["true"], node["false"]
                else:
                    # 'true' must be RIGHT (> t)
                    left_child, right_child = node["false"], node["true"]

            # push right then left so child indices > parent
            stack.append((right_child, i, False))
            stack.append((left_child,  i, True))

        if parent != -1:
            if is_left:
                ch_left[parent] = i
            else:
                ch_right[parent] = i

    n = len(ch_left)
    ch_left  = np.asarray(ch_left,  dtype=np.int32)
    ch_right = np.asarray(ch_right, dtype=np.int32)
    feat     = np.asarray(feat,     dtype=np.int32)
    thr      = np.asarray(thr,      dtype=float)

    # value: leaves one-hot; internals = sum(children)
    value = np.zeros((n, 1, C), dtype=float)
    for i in range(n):
        if ch_left[i] == -1:
            value[i, 0, :] = leaf_vecs[i]
    for i in range(n - 1, -1, -1):
        if ch_left[i] != -1:
            value[i, 0, :] = value[ch_left[i], 0, :] + value[ch_right[i], 0, :]

    # n_features_in_: Prefer provided n_features, else infer from encoder or tree features
    logger = logging.getLogger("build_tree_classifier_from_eq_ref_json")
    
    if n_features is not None:
        # Use provided n_features (most reliable - from training data shape)
        n_features_in = int(n_features)
    elif encoder is not None and hasattr(encoder, 'n_features_in_') and encoder.n_features_in_ is not None:
        # Use the stored original feature count from the encoder
        n_features_in = int(encoder.n_features_in_)
    elif encoder is not None and hasattr(encoder, 'thresholds') and encoder.thresholds is not None:
        # Fallback: infer from thresholds (may be incorrect if features were dropped)
        max_orig_feature = max(orig_f for orig_f, _ in encoder.thresholds)
        n_features_in = int(max_orig_feature + 1)
    elif np.any(feat >= 0):
        # No encoder and no n_features: use maximum feature index in the tree
        n_features_in = int(feat[feat >= 0].max() + 1)
    else:
        n_features_in = 0
    
    # logger.info("Built TreeClassifier: n_nodes=%d, n_features_in=%d, n_classes=%d", n, n_features_in, C)
    return TreeClassifier(
        children_left=ch_left,
        children_right=ch_right,
        feature=feat,
        threshold=thr,
        value=value,
        classes=classes,
        n_features_in=n_features_in,
    )
