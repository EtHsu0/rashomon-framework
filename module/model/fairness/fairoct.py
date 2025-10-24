""" fairoct.py """
import logging
import time
from collections import defaultdict
import numpy as np
try:
    from odtlearn.fair_oct import FairOCT as ODTFairOCT
except ImportError as e:
    print("Error importing odtlearn.fair_oct. Make sure the odtlearn package is installed.")
from module.model.core.post_base import PostBase
from module.model.core.base import register_model, BaseEstimator
from module.hparams import register_hparams
from module.model.lib.tree_classifier import TreeClassifier
from typing import Any, Dict, Optional, Sequence

@register_model("foct")
class FairOCT(PostBase):
    """
    Fair Optimal Classification Tree (Fair OCT) model.
    Supports three fairness criteria:
    - sp: Statistical Parity (Demographic Parity)
    - eopp: Equal Opportunity
    - eo: Equalized Odds (Equal Odds)
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.depth = params["depth"]
        self.lamb = params["lamb"]
        self.delta = params["delta"]
        self.sweep = params["sweep"]
        self.solver = params.get("solver", "gurobi")
        self.time_limit = params.get("time_limit", 1800)
        self.obj_mode = params.get("obj_mode", "acc")
        self.verbose = params.get("verbose", False)
        
        # Storage for trained models
        self.model = None  # Reference model (for backward compatibility)
        self.models = {}  # key: criterion, value: trained model
        self.pareto_models = defaultdict(list)  # key: criterion, value: list of (delta, model)
        self._single_fit_time = None  # Track time for reference model
        
    def _create_model(self, criterion, delta):
        """Create a Fair OCT model instance for the given criterion and delta."""
        mapper = {
            "sp": "SP",
            "eopp": "EOpp",
            "eo": "EOdds"
        }
        criterion_name = mapper.get(criterion)
        common_params = {
            'solver': self.solver,
            'positive_class': 1,
            'depth': self.depth,
            '_lambda': self.lamb,
            'time_limit': self.time_limit,
            'fairness_bound': delta,
            'num_threads': None,
            "fairness_type": criterion_name
        }
        return ODTFairOCT(**common_params)
    
    def fit(self, X, y, sensitive, criterion=['sp', 'eo', 'eopp']):
        """
        Fit Fair OCT model(s) with fairness constraints for all specified criteria.
        
        If sweep=True, trains multiple models with different delta values
        to create a Pareto frontier of accuracy vs fairness tradeoffs.
        Otherwise, trains a single model with the reference delta for each criterion.
        
        Note: Fair OCT requires P (protected attribute) and l (leaf indicator).
        We assume the first column of X is the leaf indicator if needed.
        
        Args:
            X: Feature matrix
            y: Labels
            sensitive: Sensitive attribute (protected group)
            criterion: List of fairness criteria to train ['sp', 'eo', 'eopp']
        """
        logger = logging.getLogger("FairOCT.fit")
        P = sensitive.reshape(-1, 1)
        l = np.ones(X.shape[0], dtype=int)  # Dummy leaf indicator, not used for SP/EO/EOPP
        
        for crit in criterion:
            logger.info(f"Training Fair OCT for criterion: {crit}")
            
            if self.sweep:
                deltas = [0.001, 0.005, 0.05, 0.1, 1.0]
                
                ref_model_trained = False
                for delta in deltas:
                    # Time only the reference model (self.delta) for fair comparison
                    if delta == self.delta and crit == criterion[0]:  # Time only first criterion
                        start_time = time.time()
                    
                    model = self._create_model(crit, delta)
                    model.fit(X, y, P, l)
                    model = build_tree_classifier_from_roctn(model, classes=np.unique(y), n_features=X.shape[1])
                    self.pareto_models[crit].append((delta, model))
                    
                    if delta == self.delta:
                        if crit == criterion[0]:
                            end_time = time.time()
                            self._single_fit_time = end_time - start_time
                            logger.info(f"Reference model ({crit}, delta={delta}) trained in {self._single_fit_time:.2f}s")
                        self.models[crit] = model
                        ref_model_trained = True
                        if crit == criterion[0]:  # Set as default model for backward compatibility
                            self.model = model
                
                # If reference delta not in sweep list, train it separately
                if not ref_model_trained:
                    logger.debug(f"Training reference model for {crit} with delta={self.delta}")
                    if crit == criterion[0]:
                        start_time = time.time()
                    
                    ref_model = self._create_model(crit, self.delta)
                    ref_model.fit(X, y, P, l)
                    ref_model = build_tree_classifier_from_roctn(ref_model, classes=np.unique(y), n_features=X.shape[1])
                    
                    if crit == criterion[0]:
                        end_time = time.time()
                        self._single_fit_time = end_time - start_time
                        logger.info(f"Reference model ({crit}, delta={self.delta}) trained in {self._single_fit_time:.2f}s")
                        self.model = ref_model
                    
                    self.models[crit] = ref_model
                    self.pareto_models[crit].append((self.delta, ref_model))
            else:
                # Train single model with reference delta
                if crit == criterion[0]:
                    start_time = time.time()
                
                model = self._create_model(crit, self.delta)
                model.fit(X, y, P, l)
                model = build_tree_classifier_from_roctn(model, classes=np.unique(y), n_features=X.shape[1])
                self.models[crit] = model
                
                if crit == criterion[0]:
                    end_time = time.time()
                    self._single_fit_time = end_time - start_time
                    self.model = model  # Set as default model for backward compatibility
                    logger.info(f"Model trained for {crit} with delta={self.delta} in {self._single_fit_time:.2f}s")
    
    def predict(self, X):
        """
        Predict using the trained model (default/first criterion).
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def post_predict(self, X, criterion):
        """
        Predict using the trained model for a specific criterion.
        
        Args:
            X: Feature matrix
            criterion: Fairness criterion ('sp', 'eo', or 'eopp')
        
        Returns:
            Predicted labels
        """
        if criterion not in self.models:
            raise ValueError(f"No model trained for criterion '{criterion}'. Available: {list(self.models.keys())}")
        return self.models[criterion].predict(X)
    
    def post_process(self, X, y, sensitive, criterion=['sp', 'eo', 'eopp']):
        """
        Placeholder for API compatibility. FairOCT is an in-processing model,
        so fairness constraints are already applied during fit().
        
        This method is a no-op since the models are already trained in fit().
        
        Args:
            X: Feature matrix (unused)
            y: Labels (unused)
            sensitive: Sensitive attribute (unused)
            criterion: List of fairness criteria (unused)
        """
        # No-op: FairOCT trains with fairness constraints in fit(), not post-processing
        pass

@register_hparams("foct")
def update_hparams(hparams, args, dataset=None):
    """Update hparams with Fair OCT-specific parameters."""
    if args is None:
        return
    
    hparams.model_params = {
        "depth": int(args.depth) if hasattr(args, 'depth') else 4,
        "lamb": float(args.lamb) if hasattr(args, 'lamb') else 0.01,
        "delta": float(args.delta) if hasattr(args, 'delta') else 0.05,
        "sweep": args.sweep if hasattr(args, 'sweep') else False,
        "solver": args.solver if hasattr(args, 'solver') else "gurobi",
        "time_limit": int(args.time_limit) if hasattr(args, 'time_limit') else 1800,
        "obj_mode": args.obj_mode if hasattr(args, 'obj_mode') else "acc",
        "verbose": args.verbose if hasattr(args, 'verbose') else False,
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
