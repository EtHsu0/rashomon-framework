""" dpf.py """
from module.model.core.base import BaseEstimator, register_model
from module.model.lib.tree_classifier import TreeClassifier
from module.hparams import register_hparams
import subprocess
from module.utils import save_data_as_dpf_csv
import json
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

@register_model("dpf")
class DPF(BaseEstimator):
    """ Dynamic Programming Fairness (DPF) """
    def __init__(self, **params):
        super().__init__(**params)
        self.datafile = params["datafile"]
        self.max_depth = params["max_depth"]
        self.exec_path = params["exec_path"]
        self.delta = params["delta"]
        self.model = None
        self.sweep = params["sweep"]
        self.fold = params["fold"]
        self.outfile = params["outfile"] + f"_fold{self.fold}.json"
        self.pareto_models = defaultdict(list)

    def _run_dpf(self, datafile, delta):
        command = [
            self.exec_path,
            "-file", datafile,
            "-mode", "best",
            "-max-depth", str(self.max_depth),
            "-max-num-nodes", str(2**(int(self.max_depth))-1),  # Complete binary tree formula
            "-outfile", self.outfile,
            "-stat-test-value", str(delta)
        ]
        print("Running DPF...", " ".join(command))
        completed = subprocess.call(command)
        if completed != 0:
            raise RuntimeError(f"DPF execution failed with return code {completed}. Command: {' '.join(command)}")

        with open(self.outfile, 'r') as f:
            tree_json = f.read()
        self.model = build_tree_classifier_from_bool_json(tree_json)
        

    def fit(self, X, y, sens):
        import time
        import logging
        logger = logging.getLogger("DPF.fit")
        
        filename = self.datafile + f"_fold{self.fold}"
        
        # Log data characteristics for debugging
        logger.info(f"Fitting DPF fold={self.fold}, X.shape={X.shape}, y.shape={y.shape}, sens.shape={sens.shape}")
        logger.info(f"y distribution: {np.unique(y, return_counts=True)}")
        logger.info(f"sens distribution: {np.unique(sens, return_counts=True)}")
        
        save_data_as_dpf_csv(filename, X, y, sens)
        filename += ".csv"

        # Store n_features for TreeClassifier (DPF may not use all features)
        n_features_in = X.shape[1]

        if self.sweep:
            # Less alphas as DPF is slow
            alphas = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 1]
            # Track time for reference model only (for fair comparison)
            ref_model_trained = False
            for alpha in alphas:
                # Time only the reference model (self.delta) for fair comparison
                if alpha == self.delta:
                    start_time = time.time()
                self._run_dpf(filename, alpha)
                model = build_tree_classifier_from_bool_json(self.outfile, n_features=n_features_in)
                self.pareto_models["sp"].append((alpha, model))
                if alpha == self.delta:
                    end_time = time.time()
                    self._single_fit_time = end_time - start_time
                    self.model = model
                    ref_model_trained = True
            if self.delta not in alphas:
                # Train and time the reference model separately
                start_time = time.time()
                self._run_dpf(filename, self.delta)
                ref_model = build_tree_classifier_from_bool_json(self.outfile, n_features=n_features_in)
                end_time = time.time()
                self._single_fit_time = end_time - start_time
                self.model = ref_model
        else:
            self._run_dpf(filename, self.delta)
            self.model = build_tree_classifier_from_bool_json(self.outfile, n_features=n_features_in)

        # print("Train score:", accuracy_score(y, self.model.predict(X)))
        # input()

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Implement the probability prediction logic for DPF
        pass

    def tune(self, nested_cv):
        """
        Tune hyperparameters for DPF using nested cross-validation.
        
        Tunes max_depth for the DPF decision tree. Delta is preserved from 
        initialization (dataset-specific or default).
        
        Args:
            nested_cv: Iterable of Split objects for inner CV
        
        Returns:
            tuple: (best_config dict, all_scores dict)
        """
        import logging
        logger = logging.getLogger("DPF.tune")
        
        # Hyperparameter grid - only tuning depth as requested
        max_depths = [2, 3, 4]
        # Use delta from model initialization (dataset-specific)
        delta = self.delta
        
        nested_cv = list(nested_cv)
        best_score = -float('inf')
        best_config = None
        all_scores = {}
        
        logger.info(f"Tuning with delta={delta} (from model initialization)")
        
        # Grid search over max_depth
        for max_depth in max_depths:
            scores = []
            for idx, split in enumerate(nested_cv):
                # Create temporary DPF model with this config
                # Use unique fold identifier for nested CV to avoid file collisions
                # Format: fold{outer}_{nested_idx} (e.g., fold2_0, fold2_1)
                nested_fold_id = f"{self.fold}_{idx}"
                
                # Build config for this depth + nested fold
                config = {
                    'max_depth': max_depth,
                    'delta': delta,
                    'exec_path': self.exec_path,
                    'datafile': self.datafile,
                    'sweep': False,
                    'fold': nested_fold_id,
                    'outfile': self.outfile.replace(f"_fold{self.fold}.json", ""),
                }
                temp_dpf = DPF(**config)

                use_binarized_data = hasattr(split, 'binarizer') and split.binarizer is not None
                if use_binarized_data:
                    X_train = split.binarizer.transform(split.X_train)
                    X_test = split.binarizer.transform(split.X_test)
                else:
                    X_train = split.preprocess(split.X_train)
                    X_test = split.preprocess(split.X_test)
                
                
                # Fit the model (requires sensitive attribute)
                temp_dpf.fit(X_train, split.y_train, split.sens_train)
                
                # Score on validation set
                score = temp_dpf.model.score(X_test, split.y_test)
                scores.append(score)
            
            avg_score = np.mean(scores)
            config_key = max_depth
            all_scores[config_key] = scores
            
            logger.debug(f"max_depth={max_depth}: Avg Score = {avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                # Store best config with original fold for final training
                # Don't include 'sweep' - let it keep the original value from initialization
                best_config = {
                    'max_depth': max_depth,
                }
        
        logger.info(f"Best Config: max_depth={best_config['max_depth']} with score: {best_score:.4f}")
        logger.info(f"Evaluated {len(all_scores)} configurations")
        return best_config, all_scores

@register_hparams("dpf")
def update_params(hparams, args, dataset):
    hparams.model_params = {
        "max_depth": int(args.max_depth) if hasattr(args, 'max_depth') else 4,
        "exec_path": args.exec_path if hasattr(args, 'exec_path') else "./module/model/packages/dpf/build/DPF",
        "outfile": args.outfile if hasattr(args, 'outfile') else f"./module/model/packages/dpf/data_temp/{dataset.dataset_name}",
        "delta": float(args.delta) if hasattr(args, 'delta') else 0.01,
        "datafile": f"./module/model/packages/dpf/data_temp/{dataset.dataset_name}",
        "sweep": args.sweep if hasattr(args, 'sweep') else False,
        "fold": args.fold if hasattr(args, 'fold') else 0,
    }
    import os 
    os.makedirs(os.path.dirname(hparams.model_params["outfile"]), exist_ok=True)


def build_tree_classifier_from_bool_json(obj, *, classes=(0, 1), key=None, n_features=None) -> TreeClassifier:
    """
    Create a TreeClassifier from boolean-split JSON.
    - Internal node: {"feature": <int>, "true": {...}, "false": {...}}
    - Leaf node:     {"prediction": <int>}
    - Semantics: go LEFT on "false" (treat features as 0/1; threshold=0.5)

    Parameters
    ----------
    obj : dict | str | path-like
        A single tree dict, or a mapping of trees like {"0": {...}, "1": {...}}.
        If str, it's parsed as JSON (not a file path).
    classes : sequence
        Class labels in order; default (0,1).
    key : str|int|None
        If `obj` is a mapping of multiple trees, choose which one to build.
    n_features : int|None
        Expected number of input features. If provided, overrides feature count inferred from tree.
        Required when DPF doesn't use all features (prevents feature mismatch at prediction time).
    """
    # ---- coerce to a single tree dict ----
    if isinstance(obj, str):
        if os.path.exists(obj):
            with open(obj, 'r') as f:
                data = json.load(f)
        else:
            data = json.loads(obj)
    else:
        data = obj

    if isinstance(data, dict) and ("feature" in data or "prediction" in data):
        root = data  # already a single tree
    else:
        # mapping of trees; pick one
        if key is None:
            if isinstance(data, dict) and len(data) == 1:
                root = next(iter(data.values()))
            else:
                raise ValueError("Multiple trees present; pass `key=` to select one.")
        else:
            root = data[str(key)] if not isinstance(key, str) and str(key) in data else data[key]

    # ---- build arrays (preorder; children indices > parent index) ----
    ch_left, ch_right, feat, thr = [], [], [], []
    leaf_onehots = []

    stack = [(root, -1, False)]  # (node_dict, parent_idx, is_left)
    C = len(classes)

    while stack:
        node, parent, is_left = stack.pop()
        idx = len(ch_left)

        ch_left.append(-1)
        ch_right.append(-1)
        feat.append(-2)         # -2 for leaves
        thr.append(-2.0)        # ignored at leaves
        leaf_onehots.append(None)

        if "prediction" in node:  # leaf
            y = int(node["prediction"])
            v = np.zeros(C, dtype=float)
            v[y if 0 <= y < C else 0] = 1.0
            leaf_onehots[idx] = v
        else:
            f = int(node["feature"])
            feat[idx] = f
            thr[idx] = 0.5
            # push RIGHT then LEFT so child indices > parent
            stack.append((node["true"],  idx, False))  # right
            stack.append((node["false"], idx, True))   # left

        if parent != -1:
            if is_left:
                ch_left[parent] = idx
            else:
                ch_right[parent] = idx

    n_nodes = len(ch_left)
    ch_left = np.asarray(ch_left, dtype=np.int32)
    ch_right = np.asarray(ch_right, dtype=np.int32)
    feat = np.asarray(feat, dtype=np.int32)
    thr = np.asarray(thr, dtype=float)

    # value: leaves one-hot, internals = sum of children (single reverse pass is enough)
    value = np.zeros((n_nodes, 1, C), dtype=float)
    for i in range(n_nodes):
        if ch_left[i] == -1:
            value[i, 0, :] = leaf_onehots[i]

    for i in range(n_nodes - 1, -1, -1):
        if ch_left[i] != -1:
            value[i, 0, :] = value[ch_left[i], 0, :] + value[ch_right[i], 0, :]

    # Determine n_features_in:
    # 1. Use explicit n_features if provided (prevents feature count mismatch)
    # 2. Fallback: infer from tree structure (max feature index + 1)
    # 3. Final fallback: 0 if tree has no splits
    if n_features is not None:
        n_features_in = n_features
    else:
        n_features_in = (feat[feat >= 0].max() + 1) if np.any(feat >= 0) else 0

    tree = TreeClassifier(
        children_left=ch_left,
        children_right=ch_right,
        feature=feat,
        threshold=thr,
        value=value,
        classes=classes,
        n_features_in=n_features_in,
    )
    return tree
