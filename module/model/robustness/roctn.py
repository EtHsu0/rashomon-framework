
"""roctn.py: Robust Optimal Classification Tree.

This module implements ROCT-N.
"""
from module.model.core.base import BaseEstimator, register_model
from module.hparams import Hparams, register_hparams
from module.datasets import DatasetLoader
try:
    from odtlearn.robust_oct import RobustOCT
except ImportError as e:
    print("Error importing odtlearn.robust_oct. Make sure the odtlearn package is installed.")
import numpy as np
from module.threshold_guess import is_binary_matrix, ThresholdGuess
from module.model.lib.tree_classifier import TreeClassifier
from sklearn.metrics import accuracy_score
from typing import Any, Sequence, Optional, Dict
import logging


@register_model("roctn")
class RoctN(BaseEstimator):
    """Robust Optimal Classification Tree with adversarial training (ROCT-N).
    
    Learns robust decision trees using MIP optimization.
    """
    
    def __init__(self, **params: Any):
        """Initialize ROCT-N model.
        
        Args:
            **params: Model hyperparameters. 'lambda' parameter controls robustness
                     vs accuracy tradeoff, rest passed to RobustOCT.
        """
        """
        Initialize RoctN model.
        Args:
            **params: Model hyperparameters. 'lambda' is extracted, rest passed to RobustOCT.
        """
        super().__init__(**params)
        self.params = params
        self.lamb = self.params.pop("lambda", 0.9)
        self.encoder = None
        self.tree_expects_binary = False  # Track if tree was built in binary space

    def binarize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit a ThresholdGuess encoder and transform X to binary features.
        """
        self.encoder = ThresholdGuess({'n_estimators': 30, 'max_depth': 2, 'learning_rate': 0.1}, back_select=False)
        self.encoder.fit(X, y)
        return self.encoder.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RoctN':
        """
        Fit the ROCT-N model. Binarizes X if needed, computes costs, and fits RobustOCT.
        
        Two scenarios:
        1. ROBUSTNESS (Kantch): Receives continuous data
           - Creates internal encoder, binarizes for training
           - Builds tree with continuous splits (embedded encoder)
           - Tree expects continuous input, handles binarization internally
        
        2. STABILITY: Receives pre-binarized data  
           - No encoder needed, trains directly on binary
           - Builds tree with binary splits
           - Tree expects binary input, no transformation needed
           - CLEARS any externally-set encoder to avoid confusion
        
        Args:
            X: Features (continuous or binary)
            y: Labels
        Returns:
            self
        """
        logger = logging.getLogger("RoctN.fit")
        
        # Determine if we're creating our own encoder or using pre-binarized data
        data_is_continuous = not is_binary_matrix(X)
        
        if data_is_continuous:
            # ROBUSTNESS: Data is continuous - create our own encoder
            X_binary = self.binarize(X, y)
            logger.info(f"Binarized {X.shape[1]} -> {X_binary.shape[1]} features")
            use_encoder_for_tree = True  # Build continuous tree
        else:
            # STABILITY: Data already binary - train in binary space, clear any external encoder
            X_binary = X
            logger.info(f"Data is pre-binarized with {X.shape[1]} features")
            self.encoder = None  # Clear any externally-set encoder
            use_encoder_for_tree = False  # Build binary tree

        budget = -1 * X_binary.shape[0] * np.log(self.lamb)
        costs = np.ones_like(X_binary, dtype=float) # unit flip cost

        q_f = self.params.pop("qf", None)
        if q_f is not None:
            # Stability evaluation: data is pre-binarized, qf matches binary features
            costs[:, q_f == 1] = budget + 1
            costs[:, q_f != 1] = -np.log(1 - q_f[q_f != 1])

        # if params has mean and std, remove them
        self.params.pop("noise_mean", None)
        self.params.pop("noise_std", None)
        roct_model = RobustOCT(**self.params)
        roct_model.fit(X_binary.astype(int), y.astype(int), costs=costs, budget=budget)
        
        # Build tree: use encoder only if we created it ourselves (continuous input)
        if use_encoder_for_tree and self.encoder is not None:
            # Build tree with continuous splits (for Kantch and other continuous metrics)
            n_features = self.encoder.n_features_in_ if hasattr(self.encoder, 'n_features_in_') else None
            tree = build_tree_classifier_from_roctn(roct_model, encoder=self.encoder, n_features=n_features)
            logger.info(f"Built tree with continuous splits expecting {n_features} features")
            self.tree_expects_binary = False  # Tree has embedded encoder, expects continuous input
        else:
            # Build tree in binary space
            tree = build_tree_classifier_from_roctn(roct_model, n_features=X_binary.shape[1])
            logger.info(f"Built tree with binary splits expecting {X_binary.shape[1]} features")
            self.tree_expects_binary = True  # Tree expects binary input
        
        self.model = tree
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained tree.
        
        Two scenarios match the two fit() scenarios:
        1. ROBUSTNESS: tree_expects_binary=False, tree has embedded encoder
           - Receives continuous input → tree handles binarization internally
        
        2. STABILITY: tree_expects_binary=True, no encoder
           - Receives binary input (possibly noisy) → pass through directly
        
        Args:
            X: Features (continuous or binary, matching training mode)
        Returns:
            Predictions
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Score the model using accuracy.
        
        Args:
            X: Features (continuous or binary - will be transformed if needed)
            y: True labels
        Returns:
            Accuracy score
        """
        # Use predict() which handles binarization if needed
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_model(self) -> Any:
        """
        Get the tree model.
        
        Returns:
            TreeClassifier (with continuous splits if encoder exists, binary otherwise)
        """
        return self.model

    def to_xgboost_json(self, output_file: str) -> Any:
        """
        Export model to XGBoost JSON format.
        """
        return self.model.to_xgboost_json(output_file=output_file)

@register_hparams("roctn")
def update_hparams(hparams, args, dataset=None):
    hparams.model_params = {
        "depth": int(args.depth) if hasattr(args, "depth") else 2,
        "lambda": float(args.lamb) if hasattr(args, "lamb") else np.exp(-0.1),
        "noise_mean": args.noise_mean if hasattr(args, "noise_mean") else 0.9,
        "noise_std": args.noise_std if hasattr(args, "noise_std") else 0.1,
        "solver": "gurobi",
        "time_limit": args.time_limit if hasattr(args, "time_limit") else 1800, 
        "qf": None,
    }

def build_tree_classifier_from_roctn(
    roct: Any,
    *,
    classes: Sequence = (0, 1),
    encoder: Optional[Any] = None,  # ThresholdGuess or compatible (optional)
    n_features: Optional[int] = None,  # Number of features (fallback if tree has no splits)
) -> TreeClassifier:
    """
    Convert a RobustOCT (RoctN) model to a TreeClassifier.

    - If encoder is None:
        feature = index of RoctN column f (0..n_enc-1)
        threshold = float(theta)      # since left iff X[f] <= theta
        children: left = heap-left, right = heap-right
    - If encoder is provided:
        Map encoded column f -> encoder index j, then
          (orig_feature_index, thr_val) = encoder.thresholds[j]
        feature = orig_feature_index
        threshold = thr_val
        Because RoctN sends (X_orig <= thr) to the RIGHT via binary 1, we SWAP children:
          children_left  <- heap-right
          children_right <- heap-left
    - n_features: Expected number of input features (used as fallback when all nodes are leaves)
    """
    logger = logging.getLogger("build_tree_classifier_from_roctn")
    
    # total nodes in 1-based heap layout
    N = int(roct._tree.total_nodes)

    # Build column-label -> encoded-index map for RoctN training data
    # (RobustOCT uses DataFrame columns or defaults to "X_0", "X_1", ...)
    col_labels = list(getattr(roct, "_X_col_labels", []))
    label_to_enc_idx: Dict[str, int] = {str(lbl): i for i, lbl in enumerate(col_labels)}

    # If encoder is present and has names, also index them for exact-string match
    enc_has_names = (encoder is not None) and hasattr(encoder, "feature_names_out") and (encoder.feature_names_out is not None)
    if enc_has_names:
        enc_names = list(encoder.feature_names_out)
        enc_name_to_idx = {str(n): i for i, n in enumerate(enc_names)}
    else:
        enc_name_to_idx = {}

    # Arrays (heap-sized; we'll prune unreachable below)
    children_left  = np.full(N, -1, dtype=np.int32)
    children_right = np.full(N, -1, dtype=np.int32)
    feature        = np.full(N, -2, dtype=np.int32)   # -2 => leaf
    threshold      = np.full(N, -2.0, dtype=float)
    leaf_label_idx = np.full(N, -1, dtype=np.int32)

    classes_arr = np.asarray(classes)

    # Leaves from w_value: pick the (unique) label with w[n,k] > 0.5
    none_count_w = 0
    for (node, k), w in getattr(roct, "w_value", {}).items():
        # Skip None values (can happen if optimization fails/times out)
        if w is None:
            none_count_w += 1
            continue
        if float(w) > 0.5:
            idx = int(node) - 1
            # map label k -> class index
            try:
                if k in classes_arr:
                    leaf_label_idx[idx] = int(np.where(classes_arr == k)[0][0])
                else:
                    leaf_label_idx[idx] = int(k)
            except Exception:
                leaf_label_idx[idx] = 0
    
    if none_count_w > 0:
        logger.warning(f"Found {none_count_w} None values in w_value - optimization may have timed out or failed")

    # Internals from b_value: find active (f, theta) for each node
    ft_pairs = list(getattr(roct, "_f_theta_indices", []))
    none_count_b = 0
    for node in range(1, N + 1):
        i = node - 1
        if leaf_label_idx[i] >= 0:
            continue  # already a leaf

        chosen = None
        for f, theta in ft_pairs:
            b_val = roct.b_value.get((node, f, theta))
            # Skip None values (can happen if optimization fails/times out)
            if b_val is None:
                none_count_b += 1
            elif float(b_val) > 0.5:
                chosen = (f, theta)
                break

        if chosen is None:
            # If neither leaf nor split, treat as leaf (unreachable nodes will be pruned anyway)
            if leaf_label_idx[i] < 0:
                leaf_label_idx[i] = 0
            continue

        f, theta = chosen
        f_label = str(f)

        # Determine encoded column index for RoctN feature f
        if f_label in label_to_enc_idx:
            enc_idx = label_to_enc_idx[f_label]
        else:
            # fall back: try "X_<j>" pattern
            if f_label.startswith("X_"):
                enc_idx = int(f_label.split("_", 1)[1])
            else:
                raise KeyError(f"Cannot resolve RoctN feature label '{f_label}' to a column index.")

        # Two cases: with encoder (map to original feature) vs without (stay in encoded space)
        if encoder is not None and hasattr(encoder, "thresholds") and encoder.thresholds is not None:
            # Prefer exact string match to encoder.feature_names_out if available.
            if enc_has_names and f_label in enc_name_to_idx:
                enc_idx = enc_name_to_idx[f_label]
            # Safety check
            if not (0 <= enc_idx < len(encoder.thresholds)):
                raise IndexError(f"Encoded feature index {enc_idx} out of range for encoder.thresholds (len={len(encoder.thresholds)}).")
            orig_f, thr_val = encoder.thresholds[enc_idx]
            feature[i]   = int(orig_f)
            threshold[i] = float(thr_val)
            # Children in heap (0-based)
            heap_left  = 2 * node - 1
            heap_right = 2 * node
            # SWAP because RoctN sends (X_orig <= thr) to RIGHT via binary 1,
            # while our TreeClassifier routes (X <= thr) to LEFT.
            children_left[i]  = heap_right if heap_right < N else -1
            children_right[i] = heap_left  if heap_left  < N else -1
        else:
            # Stay in encoded/integer space: left iff X[f] <= theta
            feature[i]   = int(enc_idx)
            threshold[i] = float(theta)
            heap_left  = 2 * node - 1
            heap_right = 2 * node
            children_left[i]  = heap_left  if heap_left  < N else -1
            children_right[i] = heap_right if heap_right < N else -1
    
    # Log warnings after processing all nodes
    if none_count_b > 0:
        logger.warning(f"Found {none_count_b} None values in b_value - optimization may have timed out or failed")

    # Any node with no split becomes a leaf; ensure it has a label
    leaf_mask = (feature == -2)
    leaf_label_idx[leaf_mask & (leaf_label_idx < 0)] = 0

    # ---- prune unreachable nodes by DFS from root and reindex compactly ----
    def reindex_from_root(cl, cr):
        mapping = {}
        order = []
        stack = [0] if N > 0 else []
        while stack:
            u = stack.pop()
            if u < 0 or u >= N or u in mapping:
                continue
            mapping[u] = len(order)
            order.append(u)
            # push right then left so left subtree gets smaller ids
            if cr[u] != -1: stack.append(cr[u])
            if cl[u] != -1: stack.append(cl[u])
        return mapping, order

    id_map, order = reindex_from_root(children_left, children_right)
    if not order:
        # Degenerate single-leaf
        V = np.zeros((1, 1, len(classes_arr)), dtype=float)
        V[0, 0, 0] = 1.0
        return TreeClassifier(
            children_left=np.array([-1], dtype=np.int32),
            children_right=np.array([-1], dtype=np.int32),
            feature=np.array([-2], dtype=np.int32),
            threshold=np.array([-2.0], dtype=float),
            value=V,
            classes=classes_arr,
            n_features_in=0,
        )

    M = len(order)
    cl = np.full(M, -1, dtype=np.int32)
    cr = np.full(M, -1, dtype=np.int32)
    fe = np.full(M, -2, dtype=np.int32)
    th = np.full(M, -2.0, dtype=float)
    V  = np.zeros((M, 1, len(classes_arr)), dtype=float)

    for old in order:
        new = id_map[old]
        fe[new] = feature[old]
        th[new] = threshold[old]
        if children_left[old]  != -1 and children_left[old]  in id_map: cl[new] = id_map[children_left[old]]
        if children_right[old] != -1 and children_right[old] in id_map: cr[new] = id_map[children_right[old]]

    is_leaf = (cl == cr)
    for new, old in enumerate(order):
        if is_leaf[new]:
            cidx = int(leaf_label_idx[old]) if leaf_label_idx[old] >= 0 else 0
            cidx = max(0, min(cidx, len(classes_arr) - 1))
            V[new, 0, cidx] = 1.0

    # propagate counts to internals
    for i in range(M - 1, -1, -1):
        if not is_leaf[i]:
            if cl[i] != -1: V[i, 0, :] += V[cl[i], 0, :]
            if cr[i] != -1: V[i, 0, :] += V[cr[i], 0, :]

    # Determine n_features_in: prefer provided n_features parameter, else infer from splits
    # This is important when the tree uses a subset of features (e.g., uses features [0,2,5,27]
    # out of 30 available features) - we want n_features_in=30, not 28.
    if n_features is not None:
        n_features_in = n_features
        if np.any(fe >= 0):
            max_feature_used = int(fe[fe >= 0].max())
            if max_feature_used >= n_features:
                logger.warning(f"Tree uses feature {max_feature_used} but n_features={n_features}; using max+1={max_feature_used+1}")
                n_features_in = max_feature_used + 1
    elif np.any(fe >= 0):
        n_features_in = int(fe[fe >= 0].max() + 1)
    else:
        n_features_in = 0
        logger.warning("Tree has no internal splits and no n_features provided; n_features_in=0")

    return TreeClassifier(
        children_left=cl,
        children_right=cr,
        feature=fe,
        threshold=th,
        value=V,
        classes=classes_arr,
        n_features_in=n_features_in,
    )
