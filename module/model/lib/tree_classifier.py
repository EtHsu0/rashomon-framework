import numpy as np
from types import SimpleNamespace
from typing import Optional, Sequence, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import sparse


class TreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible (read-only) decision tree classifier built from arrays.

    Key compat points:
      - Constructor args are optional so sklearn.clone(...) works.
      - get_params/set_params expose constructor state for cloning.
      - decision_path returns a CSR matrix like sklearn's trees.
      - classes_: 1D ndarray, with legacy-like alias _classes = [classes_].
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        *,
        children_left: Optional[np.ndarray] = None,
        children_right: Optional[np.ndarray] = None,
        feature: Optional[np.ndarray] = None,
        threshold: Optional[np.ndarray] = None,
        value: Optional[np.ndarray] = None,        # (n_nodes, 1, C) counts/probs per node
        classes: Optional[Sequence] = None,        # 1D class labels
        n_features_in: Optional[int] = None,
        feature_names_in: Optional[Sequence[str]] = None,
    ):
        # Store raw params (may be None initially so clone() can succeed)
        self.children_left  = None if children_left  is None else np.asarray(children_left,  dtype=np.int32)
        self.children_right = None if children_right is None else np.asarray(children_right, dtype=np.int32)
        self.feature        = None if feature        is None else np.asarray(feature,        dtype=np.int32)
        self.threshold      = None if threshold      is None else np.asarray(threshold,      dtype=float)
        self.value          = None if value          is None else np.asarray(value,          dtype=float)

        self._classes_1d    = None if classes is None else np.asarray(classes)
        self.n_features_in_ = None if n_features_in is None else int(n_features_in)
        self._feature_names_in_init = None if feature_names_in is None else list(feature_names_in)

        # Derived attributes initialized when we have arrays/classes
        self._is_built = False
        self._is_fitted = False

        # If arrays were provided at construction time, finish build now
        if self._has_minimal_spec():
            self._post_init_build()

    # ----------------------------- sklearn API -----------------------------
    def fit(self, X, y=None):
        """
        No-op fit to satisfy sklearn. Sets n_features_in_ from X if available.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        self.n_features_in_ = int(X.shape[1])
        self.n_features_    = self.n_features_in_
        self.input_shape    = (self.n_features_in_,)
        # If the tree has already been built from arrays, mark fitted
        if self._has_minimal_spec() and not self._is_built:
            self._post_init_build()
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_built()
        X = np.asarray(X)
        assert X.shape[1] == self.n_features_in_, f"Incompatible input shape. Got {X.shape[1]}, expected {self.n_features_in_}."
        
        idx = self._traverse_batch(X)
        scores = self.value[idx, 0, :]  # (n, C)
        preds = np.argmax(scores, axis=1)
        return self._classes_1d[preds]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_built()
        X = np.asarray(X)
        idx = self._traverse_batch(X)
        scores = self.value[idx, 0, :]  # (n, C)
        sums = scores.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.divide(scores, sums, out=np.zeros_like(scores), where=(sums > 0))
        return P

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        P = self.predict_proba(X)
        eps = np.finfo(float).eps
        return np.log(P + eps)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Index of the leaf each sample ends up in (like sklearn)."""
        self._require_built()
        X = np.asarray(X)
        return self._traverse_batch(X)

    def decision_path(self, X: np.ndarray):
        """
        Sklearn-compatible decision_path:
        Returns only the CSR node-indicator matrix of shape (n_samples, n_nodes).
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, m = X.shape[0], self.node_count

        ind = np.zeros((n, m), dtype=bool)
        for i in range(n):
            nid = 0
            while True:
                ind[i, nid] = True
                if self._is_leaf[nid]:
                    break
                f, t = self.feature[nid], self.threshold[nid]
                nid = self.children_left[nid] if X[i, f] <= t else self.children_right[nid]

        return sparse.csr_matrix(ind)

    # (optional helper if you still want leaves alongside the path)
    def decision_path_with_leaves(self, X: np.ndarray):
        node_indicator = self.decision_path(X)
        leaf_ids = self.apply(X)
        return node_indicator, leaf_ids

    def get_depth(self) -> int:
        self._require_built()
        return int(self.tree_.max_depth)

    def get_n_leaves(self) -> int:
        self._require_built()
        return int(np.sum(self._is_leaf))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # Return constructor args for sklearn.clone compatibility
        return dict(
            children_left=self.children_left,
            children_right=self.children_right,
            feature=self.feature,
            threshold=self.threshold,
            value=self.value,
            classes=self._classes_1d,
            n_features_in=self.n_features_in_,
            feature_names_in=self._feature_names_in_init,
        )

    def set_params(self, **params) -> "TreeClassifier":
        # Allow sklearn to set constructor args during clone/grid search
        for k, v in params.items():
            setattr(self, k, v)
        # Rebuild if we now have all required pieces
        if self._has_minimal_spec():
            self._post_init_build()
        return self

    # ----------------------------- utilities -----------------------------

    def export_text(self, feature_names: Optional[Sequence[str]] = None) -> str:
        self._require_built()
        lines = []

        def fname(i):
            if feature_names is not None:
                return feature_names[i]
            if hasattr(self, "feature_names_in_"):
                return str(self.feature_names_in_[i])
            return f"f[{i}]"

        def dfs(nid: int, depth: int):
            indent = "  " * depth
            if self._is_leaf[nid]:
                counts = self.value[nid, 0, :]
                s = counts.sum()
                proba = counts / s if s > 0 else counts
                lines.append(f"{indent}* leaf: proba={np.round(proba,3).tolist()}")
            else:
                f = self.feature[nid]; t = self.threshold[nid]
                lines.append(f"{indent}[{fname(f)} <= {t:.6g}]")
                dfs(self.children_left[nid], depth + 1)
                lines.append(f"{indent}[else]")
                dfs(self.children_right[nid], depth + 1)
        dfs(0, 0)
        return "\n".join(lines)

    # --------------------------- internals ---------------------------

    def _has_minimal_spec(self) -> bool:
        return (
            self.children_left is not None and
            self.children_right is not None and
            self.feature is not None and
            self.threshold is not None and
            self.value is not None and
            self._classes_1d is not None
        )

    def _require_built(self):
        if not self._is_built:
            raise RuntimeError("TreeClassifier is not built. Provide arrays via constructor or set_params(...), "
                               "or call load_from_arrays(...).")

    def load_from_arrays(
        self,
        *,
        children_left: np.ndarray,
        children_right: np.ndarray,
        feature: np.ndarray,
        threshold: np.ndarray,
        value: np.ndarray,
        classes: Sequence,
        n_features_in: Optional[int] = None,
        feature_names_in: Optional[Sequence[str]] = None,
    ):
        """Convenience method to (re)build from arrays after construction."""
        self.children_left  = np.asarray(children_left,  dtype=np.int32)
        self.children_right = np.asarray(children_right, dtype=np.int32)
        self.feature        = np.asarray(feature,        dtype=np.int32)
        self.threshold      = np.asarray(threshold,      dtype=float)
        self.value          = np.asarray(value,          dtype=float)
        self._classes_1d    = np.asarray(classes)
        self.n_features_in_ = int(n_features_in) if n_features_in is not None else self.n_features_in_
        self._feature_names_in_init = None if feature_names_in is None else list(feature_names_in)
        self._post_init_build()
        return self

    def _post_init_build(self):
        # ----- classes -----
        self.classes_ = self._classes_1d.copy()                 # sklearn modern: 1D ndarray
        self.n_classes_ = int(self._classes_1d.size)            # shape (1,)
        # self.nb_classes = int(self._classes_1d.size)
        self.classes_list_ = [self._classes_1d.copy()]          # keep your original helper
        self.n_classes_array_ = np.asarray([self._classes_1d.size])

        # ----- shapes & counts -----
        self.node_count = int(self.children_left.shape[0])
        self.n_outputs_ = 1

        # ----- features -----
        if self.n_features_in_ is None:
            max_feat = self.feature[self.feature >= 0].max() if np.any(self.feature >= 0) else -1
            self.n_features_in_ = int(max_feat + 1)
        self.n_features_ = int(self.n_features_in_)             # legacy alias
        self.input_shape = (self.n_features_in_,)

        # ----- optional feature names -----
        if self._feature_names_in_init is not None:
            if len(self._feature_names_in_init) != self.n_features_in_:
                raise ValueError(
                    f"feature_names_in length {len(self._feature_names_in_init)} != n_features_in_ {self.n_features_in_}"
                )
            self.feature_names_in_ = np.array(self._feature_names_in_init, dtype=object)

        # ----- cached leaf mask -----
        self._is_leaf = (self.children_left == self.children_right)

        # ----- validate & finalize -----
        self._validate_and_finalize()
        self._build_tree_namespace()

        self._is_built = True
        self._is_fitted = True

    def _traverse_batch(self, X: np.ndarray) -> np.ndarray:
        """Leaf-safe, vectorized traversal to leaf node ids."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self.n_features_in_:
            # be permissive: update recorded width and keep going
            self.n_features_in_ = int(X.shape[1])
            self.n_features_    = self.n_features_in_
            self.input_shape    = (self.n_features_in_,)

        n = X.shape[0]
        current = np.zeros(n, dtype=np.int32)
        while True:
            L = self.children_left[current]
            R = self.children_right[current]
            is_leaf = (L == R)
            if np.all(is_leaf):
                break
            idx = np.where(~is_leaf)[0]
            f = self.feature[current[idx]]
            t = self.threshold[current[idx]]
            go_left = (X[idx, f] <= t)
            current[idx] = np.where(go_left, L[idx], R[idx])
        return current

    def _validate_and_finalize(self) -> None:
        n = int(self.children_left.shape[0])
        assert self.children_left.shape  == (n,)
        assert self.children_right.shape == (n,)
        assert self.feature.shape        == (n,)
        assert self.threshold.shape      == (n,)
        assert self.value.ndim == 3 and self.value.shape[0] == n and self.value.shape[1] == 1

        leaves = self._is_leaf
        if not np.all(self.children_left[leaves] == -1):
            raise ValueError("Leaf nodes must have children_left == -1.")
        if not np.all(self.children_right[leaves] == -1):
            raise ValueError("Leaf nodes must have children_right == -1.")
        if not np.all(self.feature[leaves] == -2):
            raise ValueError("Leaf nodes must have feature == -2.")

        internals = ~leaves
        if np.any(self.children_left[internals]  < 0) or np.any(self.children_right[internals] < 0):
            raise ValueError("Internal nodes must have non-negative child indices.")
        if np.any(self.feature[internals] < 0):
            raise ValueError("Internal nodes must have feature >= 0.")

        # If any internal node has zero total, fill by post-order summing children
        sums = self.value[:, 0, :].sum(axis=1)
        if np.any((sums == 0) & internals):
            # Build a reverse topological order by DFS
            order = []
            stack = [0]
            seen = np.zeros(n, dtype=bool)
            while stack:
                u = stack.pop()
                if seen[u]:
                    continue
                seen[u] = True
                order.append(u)
                if not leaves[u]:
                    stack.append(self.children_left[u])
                    stack.append(self.children_right[u])
            # Post-order: process children before parent
            for u in reversed(order):
                if not leaves[u]:
                    l, r = self.children_left[u], self.children_right[u]
                    # only overwrite if zero
                    if self.value[u, 0, :].sum() == 0:
                        self.value[u, 0, :] = self.value[l, 0, :] + self.value[r, 0, :]

    def _build_tree_namespace(self) -> None:
        counts = self.value[:, 0, :]                         # (n_nodes, C)
        n_node_samples = counts.sum(axis=1)
        weighted_n_node_samples = n_node_samples.astype(float)

        # Gini impurity (like sklearn default)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.divide(counts, n_node_samples[:, None],
                          out=np.zeros_like(counts), where=(n_node_samples[:, None] > 0))
        impurity = 1.0 - np.sum(p * p, axis=1)

        # Max depth (root depth = 1, like sklearn)
        max_depth = 0
        stack = [(0, 1)]
        while stack:
            nid, d = stack.pop()
            max_depth = max(max_depth, d)
            if not self._is_leaf[nid]:
                stack.append((self.children_left[nid],  d + 1))
                stack.append((self.children_right[nid], d + 1))

        self.tree_ = SimpleNamespace(
            node_count=int(self.node_count),
            children_left=self.children_left,
            children_right=self.children_right,
            feature=self.feature,
            threshold=self.threshold,
            value=self.value,
            impurity=impurity.astype(float),
            n_node_samples=n_node_samples.astype(np.int64),
            weighted_n_node_samples=weighted_n_node_samples,
            max_depth=int(max_depth),
            n_features=int(self.n_features_in_),
            n_outputs=int(self.n_outputs_),
        )

    # --------------------------- exports ---------------------------

    def to_xgboost_json(self, output_file=None, leaf_value="proba", positive_class=None, indent=2):
        """
        Export the tree to an XGBoost-like JSON object.

        Internal node:
          {
            "nodeid": <int>,
            "depth": <int>,
            "split": <int>,                 # feature index
            "split_condition": <float>,     # threshold
            "yes": <left_nodeid>,           # <= threshold
            "no": <right_nodeid>,           # > threshold
            "missing": <left_nodeid>,       # default to left
            "children": [ <left_subtree>, <right_subtree> ]
          }

        Leaf:
          { "nodeid": <int>, "leaf": <float> }

        Args:
            output_file: if provided, write the JSON to this path and also return it.
            leaf_value:  "proba" | "log_odds" | "count" | "class_index"
                         - "proba":   binary -> P(positive_class), multiclass -> max prob
                         - "log_odds": log(p/(1-p)) for binary (falls back to "proba" if not binary)
                         - "count":   binary -> (n_pos - n_neg)/total; multiclass -> total count
                         - "class_index": argmax class index (as float)
            positive_class: which class label is "positive" for binary; default: 1 if present, else last.
            indent: JSON indentation if writing to file.

        Returns:
            A Python dict representing the tree (and also writes to file if output_file is given).
        """
        import json
        self._require_built()

        left  = self.children_left
        right = self.children_right
        feat  = self.feature
        thr   = self.threshold
        counts = self.value[:, 0, :]  # (n_nodes, n_classes)

        # Resolve classes array
        classes = getattr(self, "_classes_1d", None)
        if classes is None:
            classes = self.classes_ if isinstance(self.classes_, np.ndarray) else self.classes_[0]
        classes = np.asarray(classes)
        n_classes = int(classes.size)

        # Determine positive class index for binary
        if n_classes == 2:
            if positive_class is None:
                if 1 in classes:
                    pos_idx = int(np.where(classes == 1)[0][0])
                else:
                    pos_idx = int(np.argmax(classes))  # default to "largest" label
            else:
                pos_idx = int(np.where(classes == positive_class)[0][0])
        else:
            pos_idx = None

        eps = 1e-12

        def leaf_scalar(node_id: int) -> float:
            cnt = counts[node_id]
            total = float(cnt.sum())
            if total <= 0:
                return 0.0
            probs = cnt / total
            if leaf_value == "class_index":
                return float(np.argmax(probs))
            if n_classes == 2:
                p = float(probs[pos_idx])
                if leaf_value == "log_odds":
                    return float(np.log((p + eps) / (1.0 - p + eps)))
                if leaf_value == "count":
                    neg_idx = 1 - pos_idx
                    return float((cnt[pos_idx] - cnt[neg_idx]) / total)
                # default "proba"
                return p
            else:
                if leaf_value == "count":
                    return total
                # default multiclass: max probability
                return float(probs.max())

        # Preorder numbering (parent id < children ids)
        counter = {"next": 0}

        def build(node_index: int, depth: int):
            nodeid = counter["next"]
            counter["next"] += 1
            if left[node_index] == right[node_index]:  # leaf
                return {"nodeid": int(nodeid), "leaf": leaf_scalar(node_index)}
            # internal: build children first so we know their nodeids
            left_json  = build(left[node_index],  depth + 1)
            right_json = build(right[node_index], depth + 1)
            return {
                "nodeid": int(nodeid),
                "depth": int(depth),
                "split": int(feat[node_index]),
                "split_condition": float(thr[node_index]),
                "yes": int(left_json["nodeid"]),
                "no": int(right_json["nodeid"]),
                "missing": int(left_json["nodeid"]),   # default to left
                "children": [left_json, right_json],
            }

        obj = build(0, 0)

        if output_file is not None:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=indent)
        return obj
