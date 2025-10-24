"""threshold_guess.py: Feature binarization using GBDT-based threshold discovery.

This module implements GOSDT-Guesses, a method that uses Gradient Boosting Decision Trees
to automatically discover optimal thresholds for feature binarization. The ThresholdGuess
class trains a GBDT and extracts split thresholds to create binary features suitable for
interpretable models.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import logging


def is_binary_matrix(X):
    """Check if the input matrix contains only binary values (0 or 1).

    Args:
        X (np.ndarray): Input matrix to check.
        
    Returns:
        bool: True if matrix is binary, False otherwise.
        
    Raises:
        ValueError: If input array is not of integer, float, or boolean type.
    """
    X = np.asarray(X)
    if np.issubdtype(X.dtype, np.bool_):
        return True
    if np.issubdtype(X.dtype, np.integer):
        return np.all((X == 0) | (X == 1))
    if np.issubdtype(X.dtype, np.floating):
        return np.all(np.isclose(X, 0.0, atol=1e-8) | np.isclose(X, 1.0, atol=1e-8))
    raise ValueError("Input array must be of integer, float, or boolean type.")


class ThresholdGuess:
    """Feature binarizer using GBDT-discovered thresholds.
    
    This class trains a Gradient Boosting Decision Tree to discover informative
    split points, then uses these thresholds to convert continuous features into
    binary features suitable for interpretable decision tree models.
    """
    
    def __init__(self, guess_model_param, back_select=True, random_state=42, max_features: int = 30):
        """Initialize the ThresholdGuess binarizer.

        Args:
            guess_model_param (dict): Parameters for the GBDT model.
            back_select (bool, optional): Whether to perform backward feature selection. Defaults to True.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            max_features (int, optional): Maximum number of output binary features. Defaults to 30.
        """
        self.model_param = guess_model_param
        self.back_select = back_select
        self.thresholds = None
        self.rs = random_state
        self.num_features = None
        self.n_features_in_ = None  # Store original feature count
        self.feature_names_out = None
        self.feature_importances_ = None
        self.max_features = int(max_features)

    def fit_gbdt(self, X, y) -> tuple:
        """Fit the GBDT model to discover split thresholds.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.

        Returns:
            tuple: (trained_model, training_accuracy).
        """
        clf = GradientBoostingClassifier(**self.model_param, random_state=self.rs)
        clf.fit(X, y)
        out = clf.score(X, y)
        return clf, out

    def _cap_by_importance(self, X, y):
        """Limit number of features using importance-based selection.
        
        Fits a GBDT on transformed features and keeps only the top max_features
        most important features based on their importance scores.
        
        Args:
            X (np.ndarray): Original feature matrix (before transformation).
            y (np.ndarray): Target labels.
        """
        if len(self.thresholds) <= self.max_features:
            return  # nothing to do

        # Fit on transformed design to get importances
        X_new = self.transform(X)
        clf, _ = self.fit_gbdt(X_new, y)
        vi = getattr(clf, "feature_importances_", None)

        if vi is None or vi.size == 0:
            # Fall back: keep the first max_features thresholds deterministically
            self.thresholds = self.thresholds[: self.max_features]
            return

        # Keep top-k features by importance
        k = self.max_features
        topk_idx = np.argsort(vi)[-k:]  # indices to keep
        mask = np.zeros(len(self.thresholds), dtype=bool)
        mask[topk_idx] = True

        # Delete the rest
        new_thresholds = [thr for keep, thr in zip(mask, self.thresholds) if keep]
        self.thresholds = new_thresholds

    def fit(self, X, y, feat=None) -> None:
        """ Fit the model to find the best thresholds for features.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            feat (list): List of feature names.

        Returns:
            _type_: _description_
        """
        logger = logging.getLogger("ThresholdGuess.fit")
        orig_features = X.shape[1]
        self.n_features_in_ = orig_features  # Store original feature count
        clf, _ = self.fit_gbdt(X, y)

        thresholds = set()
        for est in clf.estimators_:
            tree = est[0].tree_
            f = tree.feature
            t = tree.threshold
            thresholds.update([(f[i], t[i]) for i in range(len(f)) if f[i] >= 0])

        self.thresholds = list(thresholds)

        # If back-select, prune by importance immediately (now capped to max_features)
        if self.back_select and len(self.thresholds) > 0:
            X_new = self.transform(X, feat)
            clf, _ = self.fit_gbdt(X_new, y)

            vi = getattr(clf, "feature_importances_", None)
            if vi is not None and vi.size > 0:
                num_features_to_keep = min(self.max_features, X_new.shape[1])
                # remove everything except the top-k by importance
                vi_indices_to_remove = np.argsort(vi)[:-num_features_to_keep]
                # Drop from X_new shape reference + thresholds list
                for idx in sorted(vi_indices_to_remove, reverse=True):
                    del self.thresholds[idx]
                # (Optional) stabilize order
                self.thresholds.sort(key=lambda x: (x[0], x[1]))

        # FINAL SAFEGUARD: even if back_select=False (or not enough pruning), enforce cap
        if len(self.thresholds) > self.max_features:
            self._cap_by_importance(X, y)
            # (Optional) stabilize order after capping
            self.thresholds.sort(key=lambda x: (x[0], x[1]))

        self.num_features = len(self.thresholds)
        if feat is None:
            feat = [f"feature_{i}" for i in range(X.shape[1])]
        feature_names_out = []
        prev = (None, None)
        for i, _ in enumerate(self.thresholds):
            f, t = self.thresholds[i]
            # check if the original column is binary
            if np.array_equal(X[:, f], X[:, f].astype(bool)):
                feature_names_out.append(feat[f])
            else:
                if f == prev[0]:
                    feature_names_out.append(f"{prev[1]}<{feat[f]}<={t}")
                else:
                    feature_names_out.append(f"{feat[f]}<={t}")
            prev = (f, t)

        self.feature_names_out = feature_names_out
        logger.info(f"Number of features: {orig_features} -> {self.num_features} (capped at {self.max_features})")
        return

    def transform(self, X, feat=None) -> np.ndarray:
        """ Transform the feature matrix using the found thresholds.
                Need to call fit() before calling this function.
        
        Args:
            X (np.ndarray): Feature matrix.
            feat (list, optional): List of feature names. Defaults to None.

        Returns:
            np.ndarray: Transformed feature matrix.
        """
        X = X.copy()
        feature_names_in = feat
        if feature_names_in is None:
            feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]
        # check or transform X, y into ndarrays
        if not isinstance(X, np.ndarray):
            X = X.values

        feature_names_out = []
        X_new = np.zeros((X.shape[0], len(self.thresholds)))
        for i, _ in enumerate(self.thresholds):
            f, t = self.thresholds[i]
            X_new[X[:, f] <= t, i] = 1
            feature_names_out.append(f"{feature_names_in[f]} <= {t}")

        self.feature_names_out = feature_names_out

        return X_new
