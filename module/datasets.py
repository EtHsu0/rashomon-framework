""" datasets.py:  Dataset loading and preprocessing utilities. """
import logging
from dataclasses import dataclass
from typing import Optional, Any, Tuple, Iterator, Callable, Literal

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

from module.threshold_guess import ThresholdGuess

def parse_dataset_id(s: str) -> tuple[str, Optional[str]]:
    """Supports 'adult' and 'adult@openml' forms."""
    if "@" in s:
        name, src = s.split("@", 1)
        return name.strip(), src.strip()
    return s.strip(), None

@dataclass
class SensitiveSpec:
    col_name: str
    mode: Literal["binary", "range"] = "binary"
    # For "binary": typically (-inf, 0.5] or (0.5, +inf) for OHE binary columns.
    # For "range": numeric interval inclusive.
    threshold: Tuple[float, float] = (-np.inf, np.inf)

@dataclass
class Split:
    fold_id: int
    nested_cv: Optional[Iterator["Split"]]

    X_train: np.ndarray
    y_train: np.ndarray
    X_select: Optional[np.ndarray]
    y_select: Optional[np.ndarray]
    X_test: np.ndarray
    y_test: np.ndarray

    sens_train: Optional[np.ndarray] = None
    sens_select: Optional[np.ndarray] = None
    sens_test: Optional[np.ndarray] = None

    scaler: Optional[MinMaxScaler] = None
    scale_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    preprocessor: Optional[ThresholdGuess] = None
    preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None
    
    # Binarization support
    binarizer: Optional[ThresholdGuess] = None
    
    def get_binary_data(self):
        """Return binarized versions of train/select/test data"""
        if self.binarizer is None:
            raise ValueError("No binarizer available - binarize_mode was not enabled")
        return {
            'X_train': self.binarizer.transform(self.X_train),
            'X_select': self.binarizer.transform(self.X_select) if self.X_select is not None else None,
            'X_test': self.binarizer.transform(self.X_test)
        }


class DatasetLoader:
    """ Dataset Loader Class """

    def __init__(self, dataset_name, random_state, args):
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.rs = random_state
        self.feat_name: Optional[np.ndarray] = None
        self.raw_data: Optional[Any] = None
        self.full_name: str = dataset_name
        self.dataset_name, self.source = parse_dataset_id(dataset_name)

        self.fairness_mode = getattr(args, "fairness_mode", False)
        self.binarize_mode = getattr(args, "binarize_mode", None)

        self.sensitive_idx: Optional[int] = None
        self.sensitive_threshold: Tuple[float, float] = (-np.inf, np.inf)

        self.preprocessor: Optional[ThresholdGuess] = None
        self.preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None

        self.mapper = {
            # Integrated
            "adult": self.load_adult,
            "bank": self.load_bank_marketing,
            "california-houses": self.load_california_houses,
            "compas": self.load_compas,
            "credit-fusion": self.load_credit_fusion,
            "default-credit": self.load_default_credit,
            "diabetes-130US": self.load_diabetes_130US,

            # Robustness / Stability
            # "banknote"
            "banknote": self.load_banknote_authentication,
            "blood": self.load_blood_transfusion,
            "breast-w": self.load_breast_w,
            # "compas"
            "cylinder-bands": self.load_cylinder_bands,
            "diabetes": self.load_diabetes,
            "fico": self.load_fico,
            "haberman": self.load_haberman,
            "ionosphere": self.load_ionosphere,
            "mimic": self.load_mimic,
            "parkinsons": self.load_parkinsons,
            "sonar": self.load_sonar,
            "spambase": self.load_spambase,
            "spectf": self.load_spectf,
            "wine-quality": self.load_wine_quality,

            # Fairness
            # "adult"
            # "bank"
            # "compas"
            "census-income": self.load_census_income,
            "communities-crime": self.load_communities_crime,
            "german-credit": self.load_german_credit,
            "oulad": self.load_oulad,
            "student-mat": self.load_student_mathematics,
            "student-por": self.load_student_portuguese,

            # privacy
            # "adult"
            # "bank"
            # "diabetes-130US"
            # "german-credit"
            # "oulad"

            # Note: Use dataset_name@source syntax (e.g., 'adult@dpf') to specify data source
            # Available sources: 'openml', 'tabular-benchmark', 'dpf', 'original'
        }

        self.load_dataset(self.dataset_name)

    # ------------------------
    # Public getters
    # ------------------------

    def get_all(self) -> dict:
        """ Get all data as a dict. """
        return {
            'X': self.X,
            'y': self.y,
            'feature': self.feat_name,
            'original': self.raw_data,
        }

    def list_all_dataset(self) -> list:
        """ List available datasets from the mapper. """
        return list(self.mapper.keys())

    # ------------------------
    # Internal utilities
    # ------------------------

    @staticmethod
    def _remove_nan(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = ~np.isnan(X).any(axis=1)
        return X[mask], y[mask]

    def _set_sensitive_by_name(self, feat_name: np.ndarray, spec: SensitiveSpec):
        try:
            idx = int(np.where(feat_name == spec.col_name)[0][0])
        except IndexError as e:
            raise ValueError(f"Sensitive feature '{spec.col_name}' not found in features.") from e
        self.sensitive_idx = idx
        self.sensitive_threshold = spec.threshold

    def set_all(self, X, y, feat_name, openml_or_df):
        """Set all data with consistent types and checks."""
        logger = logging.getLogger("datasets.set_all")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X, y = self._remove_nan(X, y)
        uniq = set(np.unique(y))
        assert uniq == {0, 1}, f"Need binary classification with labels {{0,1}}, got {sorted(uniq)}"
        self.X = X
        self.y = y
        self.feat_name = np.asarray(feat_name)
        self.raw_data = openml_or_df
        logger.info("Total of %s rows, %s features", len(X), X.shape[1])

    # ------------------------
    # Preprocessing (scaler + optional GBDT binarizer)
    # ------------------------

    def _fit_outer_preprocessors(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_scaler: bool,
        use_gbdt: bool
    ) -> Tuple[Optional[MinMaxScaler], Callable[[np.ndarray], np.ndarray],
               Optional[ThresholdGuess], Callable[[np.ndarray], np.ndarray]]:
        # 1) scaler
        if use_scaler:
            scaler = MinMaxScaler().fit(X_train)
            scale_fn = scaler.transform
        else:
            scaler = None
            scale_fn = lambda a: a

        # 2) binarizer (after scaling)
        if use_gbdt:
            preprocessor = ThresholdGuess(
                {'max_depth': 2, 'n_estimators': 30, 'learning_rate': 0.1},
                True,
                self.rs
            )
            Xs = scale_fn(X_train)
            preprocessor.fit(Xs, y_train, self.feat_name)
            preprocess = preprocessor.transform
        else:
            preprocessor = None
            preprocess = lambda a: a

        return scaler, scale_fn, preprocessor, preprocess

    @staticmethod
    def _apply_preprocessors(
        scale_fn: Callable[[np.ndarray], np.ndarray],
        preprocess: Callable[[np.ndarray], np.ndarray],
        *arrays
    ) -> Tuple[Optional[np.ndarray], ...]:
        out = []
        for a in arrays:
            if a is None:
                out.append(None)
            else:
                out.append(preprocess(scale_fn(a)))
        return tuple(out)

    def _make_sensitive_masks(
        self, X_train: np.ndarray, X_select: Optional[np.ndarray], X_test: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Build boolean masks for the sensitive feature using thresholds on RAW (pre-transform) arrays."""
        assert self.sensitive_idx is not None, "Sensitive feature index must be set before masking."
        idx = self.sensitive_idx
        left, right = self.sensitive_threshold
        train_m = (X_train[:, idx] >= left) & (X_train[:, idx] <= right)
        test_m = (X_test[:, idx] >= left) & (X_test[:, idx] <= right)
        select_m = None if X_select is None else ((X_select[:, idx] >= left) & (X_select[:, idx] <= right))
        return train_m, select_m, test_m

    def _drop_sensitive_col(
        self, X_train: np.ndarray, X_select: Optional[np.ndarray], X_test: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        assert self.sensitive_idx is not None, "Sensitive feature index must be set before dropping."
        idx = self.sensitive_idx
        X_train = np.delete(X_train, idx, axis=1)
        X_test = np.delete(X_test, idx, axis=1)
        if X_select is not None:
            X_select = np.delete(X_select, idx, axis=1)
        return X_train, X_select, X_test

    # ------------------------
    # Unified split generator
    # ------------------------

    def kfold_splits(
        self,
        n_splits: int = 5,
        inner_splits: Optional[int] = None,
        select_size: float = 0.1,
        shuffle: bool = True,
        use_scaler: bool = True,
        use_gbdt_binarizer: bool = False,
    ) -> Iterator[Split]:
        """
        Generic split generator that yields Split objects.
        - If inner_splits is not None, each outer Split has a `nested_cv` generator.
        - If fairness_mode is True, sensitive masks are produced on raw arrays and the sensitive column is dropped.
        """
        skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.rs)

        for outer_id, (tr_idx, te_idx) in enumerate(skf_outer.split(self.X, self.y)):
            X_tr_raw, X_te_raw = self.X[tr_idx].copy(), self.X[te_idx].copy()
            y_tr, y_te = self.y[tr_idx].copy(), self.y[te_idx].copy()

            X_sel_raw, y_sel = None, None
            if select_size and select_size > 0.0:
                X_tr_raw, X_sel_raw, y_tr, y_sel = train_test_split(
                    X_tr_raw, y_tr, test_size=select_size, random_state=self.rs, stratify=y_tr
                )

            sens_tr = sens_sel = sens_te = None
            if self.fairness_mode:
                logger = logging.getLogger("datasets.kfold_splits")
                logger.info("Generating sensitive masks for fairness evaluation")
                assert self.sensitive_idx is not None, "Dataset does not have sensitive feature setup."
                sens_tr, sens_sel, sens_te = self._make_sensitive_masks(X_tr_raw, X_sel_raw, X_te_raw)
                X_tr_raw, X_sel_raw, X_te_raw = self._drop_sensitive_col(X_tr_raw, X_sel_raw, X_te_raw)

            scaler, scale_fn, preprocessor, preprocess = self._fit_outer_preprocessors(
                X_train=X_tr_raw, y_train=y_tr,
                use_scaler=use_scaler, use_gbdt=use_gbdt_binarizer
            )
            X_tr, X_sel, X_te = self._apply_preprocessors(scale_fn, preprocess, X_tr_raw, X_sel_raw, X_te_raw)

            # Binarization support
            binarizer = None
            if self.binarize_mode is not None:
                from module.threshold_guess import ThresholdGuess, is_binary_matrix
                logger = logging.getLogger("datasets.kfold_splits")
                
                # Only binarize if data has more than 30 features (for feature selection)
                # or if data is continuous (needs binarization for binary-only models)
                n_features = X_tr.shape[1]
                is_binary = is_binary_matrix(X_tr)
                
                if n_features > 30 or not is_binary:
                    if n_features > 30:
                        logger.info(f"Using GBDT binarization for feature selection: {n_features} → ~30 features (max)")
                    else:
                        logger.info(f"Using GBDT binarization to convert continuous features to binary")
                    
                    # Use same parameters as TreeFARMS for consistency
                    binarizer = ThresholdGuess(
                        {'n_estimators': 30, 'max_depth': 2, 'learning_rate': 0.1}, 
                        back_select=False,
                        random_state=self.rs,
                        max_features=30  # Cap at 30 features for feature selection
                    )
                    # Fit binarizer on processed training data to avoid data leakage
                    binarizer.fit(X_tr, y_tr)
                else:
                    logger.info(f"Data is already binary with {n_features} features (≤30), skipping binarization")

            nested_gen = None
            if inner_splits and inner_splits > 1:
                def _inner_generator():
                    skf_inner = StratifiedKFold(n_splits=inner_splits, shuffle=shuffle, random_state=self.rs)
                    for inner_id, (inn_tr_idx, inn_va_idx) in enumerate(skf_inner.split(X_tr_raw, y_tr)):
                        Xi_tr_raw_full = X_tr_raw[inn_tr_idx].copy()
                        Xi_va_raw_full = X_tr_raw[inn_va_idx].copy()
                        yi_tr, yi_va = y_tr[inn_tr_idx].copy(), y_tr[inn_va_idx].copy()
                        if self.fairness_mode:
                            si_tr, si_va = sens_tr[inn_tr_idx].copy(), sens_tr[inn_va_idx].copy()
                        else:
                            si_tr = si_va = None
                        # Note: X_tr_raw already has sensitive column dropped in outer fold if fairness_mode=True
                        Xi_tr_raw, Xi_va_raw = Xi_tr_raw_full, Xi_va_raw_full
                        _scaler, _scale_fn, _preproc, _pre_fn = self._fit_outer_preprocessors(
                            X_train=Xi_tr_raw, y_train=yi_tr,
                            use_scaler=use_scaler, use_gbdt=use_gbdt_binarizer
                        )

                        Xi_tr, Xi_va = self._apply_preprocessors(
                            _scale_fn, _pre_fn, Xi_tr_raw, Xi_va_raw
                        )
                        
                        yield Split(
                            fold_id=inner_id,
                            nested_cv=None,
                            X_train=Xi_tr, y_train=yi_tr,
                            X_select=None, y_select=None,
                            X_test=Xi_va, y_test=yi_va,
                            sens_train=si_tr, sens_select=None, sens_test=si_va,
                            scaler=_scaler, scale_fn=_scale_fn,
                            preprocessor=_preproc, preprocess=_pre_fn,
                            binarizer=binarizer  # Use outer fold's binarizer for inner folds this is not recommended. We don't officially support inner binarization fitting.
                        )
                nested_gen = _inner_generator()

            yield Split(
                fold_id=outer_id,
                nested_cv=nested_gen,
                X_train=X_tr, y_train=y_tr,
                X_select=X_sel, y_select=y_sel,
                X_test=X_te, y_test=y_te,
                sens_train=sens_tr, sens_select=sens_sel, sens_test=sens_te,
                scaler=scaler, scale_fn=scale_fn,
                preprocessor=preprocessor, preprocess=preprocess,
                binarizer=binarizer
            )

    # ------------------------
    # Backward-compat wrappers
    # ------------------------

    def kfold_data(self, kfold=5):
        """K-Fold the whole dataset (returns index splits)."""
        return self.kfold_X_y(self.X, self.y, kfold)

    def kfold_X_y(self, X, y, kfold=5):
        """K-Fold given X and y (returns index splits)."""
        return StratifiedKFold(n_splits=kfold, shuffle=True, random_state=self.rs).split(X, y)

    def load_dataset(self, name: str):
        """ Load dataset by name. """
        logger = logging.getLogger("Datasets.load_dataset")
        logger.info("Loading dataset: %s", name)
        if name not in self.mapper:
            raise KeyError(f"Unknown dataset '{name}'. Known: {list(self.mapper)}")
        return self.mapper[name]()

    def load_adult(self, src="openml"):
        """Adult (OpenML 1590)."""
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=1590, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            df = pd.concat([X_df, y_ser.rename("target")], axis=1)
            df.replace("?", np.nan, inplace=True)
            df.dropna(inplace=True)

            X_df = df.drop(columns=["target"])
            y = (df["target"] == ">50K").astype(int).to_numpy()

            cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                X_df = pd.get_dummies(X_df, columns=cat_cols, dummy_na=False, drop_first=True)

            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)

            # Sensitive: sex_Male binary; treat <= 0.5 as protected
            self._set_sensitive_by_name(feat_name, SensitiveSpec("sex_Male", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        elif src == "dpf":
            data = pd.read_csv("data/dpf/adult.csv")
            X_df = data.drop(columns=["Class:>50K"])
            y = data["Class:>50K"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            # Sensitive: first column (Sex=Male)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("Sex=Male", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Adult dataset")

    def load_bank_marketing(self, src="tabular-benchmark"):
        """ Bank Marketing (OpenML 44126). """
        if self.source: src = self.source
        if src == "tabular-benchmark":
            data = fetch_openml(data_id=44126, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(int) - 1).to_numpy(dtype=int)
            if "V1" not in X_df.columns:
                raise ValueError("Expected 'V1' to be the age column in bank-marketing features.")
            self._set_sensitive_by_name(feat_name, SensitiveSpec("V1", "range", (25, 60)))
            self.set_all(X, y, feat_name, data)
        elif src == "dpf":
            data = pd.read_csv("data/dpf/bank-marketing.csv")
            X_df = data.drop(columns=["class:yes"])
            y = data["class:yes"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("marital:married", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Bank Marketing dataset")

    def load_california_houses(self, src="tabular-benchmark"):
        """ California Houses (OpenML 44090). """
        if self.source: src = self.source
        if src == "tabular-benchmark":
            data = fetch_openml(data_id=44090, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser == "True").to_numpy(dtype=int)
            # sensitive: MedInc <= median (protected)
            if "MedInc" not in X_df.columns:
                raise ValueError("Expected 'MedInc' in california-houses features.")
            medinc_idx = int(np.where(feat_name == "MedInc")[0][0])
            med = float(np.median(X[:, medinc_idx]))
            self._set_sensitive_by_name(feat_name, SensitiveSpec("MedInc", "range", (-np.inf, med)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for California Houses dataset")

    def load_compas(self, src="original"):
        """ COMPAS Dataset (CSV). """
        if self.source: src = self.source
        if src == "original":
            data = pd.read_csv("data/compas.csv")
            X_df = data.drop(columns=["two_year_recid"])
            y = data["two_year_recid"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            # sensitive column by name:
            self._set_sensitive_by_name(feat_name, SensitiveSpec("sex=female", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        elif src == "dpf":
            data = pd.read_csv("data/dpf/compas.csv")
            X_df = data.drop(columns=["two_year_recid"])
            y = data["two_year_recid"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("race:Caucasian", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for COMPAS dataset")

    def load_credit_fusion(self, src="tabular-benchmark"):
        """ Credit Fusion (OpenML 44089). """
        if self.source: src = self.source
        if src == "tabular-benchmark":
            data = fetch_openml(data_id=44089, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(str) == "1").to_numpy(dtype=int)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("age", "range", (25, 60)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Credit Fusion dataset")

    def load_default_credit(self, src="tabular-benchmark"):
        """ Default Credit (CSV in your repo). """
        if self.source: src = self.source
        if src == "tabular-benchmark":
            data = pd.read_csv("data/default-credit.csv")
            X_df = data.drop(columns=["DEFAULT_PAYMENT"])
            y = data["DEFAULT_PAYMENT"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("SEX_Male", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        elif src == "dpf":
            data = pd.read_csv("data/dpf/default-credit.csv")
            X_df = data.drop(columns=["DEFAULT_PAYMENT"])
            y = data["DEFAULT_PAYMENT"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("SEX_Female", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Default Credit dataset")

    def load_diabetes_130US(self, src="openml"):
        """ Diabetes 130 US (OpenML 4541). """
        logger = logging.getLogger("datasets.load_diabetes_130US")
        if self.source: src = self.source

        if src == "tabular-benchmark":
            data = pd.read_csv("data/Diabetes130US.csv")
            X_df = data.drop(columns=["readmitted"])
            y = data["readmitted"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self.set_all(X, y, feat_name, data)
        elif src == "openml":
            data = fetch_openml(data_id=4541, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()

            feature_to_keep = [
                "gender", "time_in_hospital", "num_lab_procedures", "num_medications",
                "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"
            ]
            X_df = X_df[feature_to_keep].copy()

            # Turn gender into binary feature 'gender=Male'
            X_df["gender=Male"] = (X_df["gender"] == "Male").astype(int)
            X_df.drop(columns=["gender"], inplace=True)

            y = (y_ser == "<30").astype(int).to_numpy()

            # Balance dataset
            logger.info("Original dataset size: %s", len(X_df))
            y_np = y.copy()
            idx0 = np.where(y_np == 0)[0]
            idx1 = np.where(y_np == 1)[0]
            max_size = min(len(idx0), len(idx1))
            rng = np.random.default_rng(self.rs)
            sampled_idx = np.concatenate([
                rng.choice(idx0, size=max_size, replace=False),
                rng.choice(idx1, size=max_size, replace=False)
            ])
            X_df = X_df.iloc[sampled_idx]
            y = y_np[sampled_idx]
            logger.info("Balanced dataset size: %s", len(X_df))

            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("gender=Male", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        elif src == "dpf":
            data = pd.read_csv("data/dpf/Diabetes130US.csv")
            X_df = data.drop(columns=["readmitted:<30"])
            y = data["readmitted:<30"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("gender=Female", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Diabetes 130 US dataset")

    def load_haberman(self, src="openml"):
        """ Haberman Survival (OpenML 43). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=43, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(int).to_numpy() == 2).astype(int)  # 2 -> event(1), 1 -> 0
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Haberman dataset")

    def load_blood_transfusion(self, src="openml"):
        """ Blood Transfusion (OpenML 1464). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=1464, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(int).to_numpy() == 2).astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Blood Transfusion dataset")

    def load_climate_simulation(self, src="openml"):
        """ Climate Model Simulation Crash (OpenML 1467). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=1467, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = y_ser.astype(int).to_numpy()
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Climate Simulation dataset")

    def load_sonar(self, src="openml"):
        """ Sonar (OpenML 40). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=40, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(str).to_numpy() == "Mine").astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Sonar dataset")

    def load_parkinsons(self, src="openml"):
        """ Parkinsons (OpenML 1488). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=1488, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(int).to_numpy() == 2).astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Parkinsons dataset")

    def load_banknote_authentication(self, src="openml"):
        """ Banknote Authentication (OpenML 1462). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=1462, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(int).to_numpy() == 2).astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Banknote Authentication dataset")

    def load_breast_w(self, src="openml"):
        """ Breast Cancer Wisconsin (Original) (OpenML 15). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=15, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(str).to_numpy() == "malignant").astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Breast Cancer Wisconsin dataset")

    def load_cylinder_bands(self, src="openml"):
        """ Cylinder-Bands (OpenML 6332). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=6332, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(str).to_numpy() == "band").astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Cylinder Bands dataset")

    def load_diabetes(self, src="openml"):
        """ Pima Indians Diabetes (OpenML 37). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=37, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(str).to_numpy() == "tested_positive").astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Diabetes dataset")

    def load_ionosphere(self, src="openml"):
        """ Ionosphere (OpenML 59). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=59, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(str).to_numpy() == "g").astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Ionosphere dataset")

    def load_planning_replax(self, src="openml"):
        """ Planning Relax (OpenML 1490). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=1490, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(str).to_numpy() == "2").astype(int)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Planning Relax dataset")

    def load_spambase(self, src="openml"):
        """ Spambase (OpenML 44). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=44, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = y_ser.astype(int).to_numpy()
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Spambase dataset")

    def load_spectf(self, src="openml"):
        """ SPECTF (OpenML 1600). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=1600, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = y_ser.astype(int).to_numpy()
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for SPECTF dataset")

    def load_wine_quality(self, src="openml"):
        """ Wine Quality (OpenML 287). """
        if self.source: src = self.source
        if src == "openml":
            data = fetch_openml(data_id=287, as_frame=True)
            X_df, y_ser = data.data.copy(), data.target.copy()
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            y = (y_ser.astype(int).to_numpy() >= 6).astype(int)  # >=6 -> good wine
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Wine Quality dataset")

    def load_fico(self, src="original"):
        """FICO Credit Score Dataset (CSV)."""
        if self.source: src = self.source
        if src == "original":
            data = pd.read_csv("data/fico.csv")
            X_df = data.drop(columns=["RiskPerformance"])
            y = data["RiskPerformance"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for FICO dataset")

    def load_mimic(self, src="original"):
        """MIMIC-II Hospital Mortality Dataset (CSV)."""
        if self.source: src = self.source
        if src == "original":
            data = pd.read_csv("data/mimic2.csv")
            X_df = data.drop(columns=["HospitalMortality"])
            y = data["HospitalMortality"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for MIMIC dataset")

    def load_census_income(self, src="dpf"):
        """ Census Income (CSV). """
        if self.source: src = self.source
        if src == "dpf":
            data = pd.read_csv("data/dpf/census-income.csv")
            X_df = data.drop(columns=["class: 50000+"])
            y = data["class: 50000+"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("race: White", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Census Income dataset")

    def load_oulad(self, src="dpf"):
        """Open University Learning Analytics Dataset (CSV)."""
        if self.source: src = self.source
        if src == "dpf":
            data = pd.read_csv("data/dpf/oulad.csv")
            X_df = data.drop(columns=["final_result:Pass"])
            y = data["final_result:Pass"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("gender:M", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for OULAD dataset")

    def load_german_credit(self, src="dpf"):
        """ German Credit (CSV). """
        if self.source: src = self.source
        if src == "dpf":
            data = pd.read_csv("data/dpf/german-credit.csv")
            X_df = data.drop(columns=["Creditability"])
            y = data["Creditability"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("Gender=Male", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for German Credit dataset")

    def load_communities_crime(self, src="dpf"):
        """ Communities and Crime (CSV). """
        if self.source: src = self.source
        if src == "dpf":
            data = pd.read_csv("data/dpf/communities-crime.csv")
            X_df = data.drop(columns=["ViolentCrimesPerPop<0.7"])
            y = data["ViolentCrimesPerPop<0.7"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("racepctblack<0.07", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Communities and Crime dataset")

    def load_student_mathematics(self, src="dpf"):
        """Student Performance Dataset - Mathematics (CSV)."""
        if self.source: src = self.source
        if src == "dpf":
            data = pd.read_csv("data/dpf/student-mat.csv")
            X_df = data.drop(columns=["G3>=10"])
            y = data["G3>=10"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("sex:M", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Student Mathematics dataset")

    def load_student_portuguese(self, src="dpf"):
        """Student Performance Dataset - Portuguese (CSV)."""
        if self.source: src = self.source
        if src == "dpf":
            data = pd.read_csv("data/dpf/student-por.csv")
            X_df = data.drop(columns=["G3>=10"])
            y = data["G3>=10"].to_numpy(dtype=int)
            feat_name = X_df.columns.to_numpy()
            X = X_df.to_numpy(dtype=np.float32)
            self._set_sensitive_by_name(feat_name, SensitiveSpec("sex:M", "binary", (-np.inf, 0.5)))
            self.set_all(X, y, feat_name, data)
        else:
            raise ValueError(f"Unknown source {src} for Student Portuguese dataset")
