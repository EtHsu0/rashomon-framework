"""stability_metric.py: Stability metrics for evaluating prediction consistency.

This module implements metrics for assessing model stability under small input
perturbations. Uses geometric noise to simulate data variations and measures
prediction consistency across perturbed versions.
"""
from module.metric.base_metrics import BaseMetric, register_metric
from sklearn.metrics import accuracy_score
from typing import Dict
import numpy as np
from numpy.random import SeedSequence, default_rng
from collections import defaultdict
import tempfile
import os
import shutil


@register_metric()
class StabilityMetric(BaseMetric):
    """Stability metric evaluating prediction consistency under perturbations.
    
    Tests model robustness to small input variations by applying geometric noise
    to binary features and measuring prediction agreement across multiple trials.
    Requires binary features for proper noise model application.
    """

    NAME = "stability"
    REQUIRES_BINARY_FEATURES = True  # Needs binary features

    def setup(self, model, hparams, split_data, **params) -> None:
        """Initialize stability evaluation with noise parameters and trial configuration.
        
        Sets up reproducible random number generators for noise injection and
        configures the number of trials and noise distribution parameters.
        
        Args:
            model: Trained model (unused but required).
            hparams (Hparams): Hyperparameters containing noise_dist_qf.
            split_data (Split): Data split (unused but required).
            **params: Additional parameters including num_trials, delta, resample_q, cache_mode.
            
        Raises:
            AssertionError: If noise_dist_qf is not properly configured.
        """
        # Derive a child RNG dedicated to this metric for reproducibility
        try:
            ss = SeedSequence(getattr(hparams, "rs", None))
            self.rng = default_rng(ss.spawn(1)[0])
            # Pre-generate per-trial seeds so every model reuses identical perturbations
            self.num_trials = int(params.get("num_trials", 500))
            self.trial_ss_train = ss.spawn(self.num_trials)
            self.trial_ss_test = ss.spawn(self.num_trials)
            # Dedicated seed for q_f resampling so it is identical for every compute()
            self.qf_resample_seed = ss.spawn(1)[0]
        except Exception:
            # Fallback to provided rng
            self.rng = getattr(hparams, "rng", np.random.default_rng())
            self.num_trials = int(params.get("num_trials", 500))
            # No SeedSequence available; fall back to integer seeds derived from rs or a fixed offset
            base = int(getattr(hparams, "rs", 0) or 0)
            self.trial_ss_train = [base + i for i in range(self.num_trials)]
            self.trial_ss_test = [base + 10_000 + i for i in range(self.num_trials)]
            self.qf_resample_seed = base + 999_983
        # How many noisy re-evaluations to perform per split (already set above)
        # Optional controls for shifting the geometric distribution parameter q_f
        self.delta = params.get("delta", 0.0)
        self.resample_q = params.get("resample_q", False)
        self.qf = hparams.noise_dist_qf
        assert (self.qf is not None), "Stability metric requires noise_dist_qf to be set in model hparams."
        assert type(self.qf) is not int and type(self.qf) is not float, f"Stability metric requires noise_dist_qf to be a numpy array or list of floats, got {type(self.qf)} with value {self.qf}"
        # Caching mode to avoid regenerating noise per model:
        #  - none: regenerate per model (fastest init, slowest across many models)
        #  - mem: cache all trials in memory (fast across models, risky for RAM)
        #  - disk: cache all trials using numpy.memmap (balanced for large data)
        # Setup does not receive metric params in this repo; default to none here.
        self.cache_noise = "none"
        self._cache_mode = "none"
        self._cache_qf = None

        # Precompute preprocessed X to avoid repeating per model
        # Use binary features if this metric requires them and binarizer is available
        if self.REQUIRES_BINARY_FEATURES and hasattr(split_data, 'binarizer') and split_data.binarizer is not None:
            binary_data = split_data.get_binary_data()
            self._X_tr = binary_data['X_train']
            self._X_te = binary_data['X_test']
        else:
            self._X_tr = split_data.preprocess(split_data.X_train)
            self._X_te = split_data.preprocess(split_data.X_test)
        
        self._y_tr = split_data.y_train
        self._y_te = split_data.y_test
        self._dtype = self._X_tr.dtype
        self._ntr, self._d = self._X_tr.shape
        self._nte = self._X_te.shape[0]

        # Noise cache placeholders
        self._cache_ready = False
        self._cache_dir = None
        self._Xn_tr_mem = None
        self._Xn_te_mem = None
        self._Xn_tr_mm = None
        self._Xn_te_mm = None


    def compute(self, predictions, split_data, **params) -> Dict[str, float]:
        pred_fn = predictions["pred_fn"]
        local_q_rng = default_rng(self.qf_resample_seed)
        
        # Main q_f variants
        qf_default = self.qf
        qf_plus_01 = self._shift_noise_dist(self.qf, 0.1, resample=False, rng=local_q_rng) if self.qf is not None else np.full(self._X_tr.shape[1], 0.6, dtype=float)
        qf_minus_01 = self._shift_noise_dist(self.qf, -0.1, resample=False, rng=local_q_rng) if self.qf is not None else np.full(self._X_tr.shape[1], 0.4, dtype=float)
        qf_minus_02 = self._shift_noise_dist(self.qf, -0.2, resample=False, rng=local_q_rng) if self.qf is not None else np.full(self._X_tr.shape[1], 0.3, dtype=float)

        # Use cached preprocessed data
        X_tr = self._X_tr
        y_tr = self._y_tr
        X_te = self._X_te
        y_te = self._y_te
        base_pred_tr = predictions["train"]
        base_pred_te = predictions["test"]

        def eval_stability(q_f, suffix):
            acc_tr_stats = {"count": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")}
            acc_te_stats = {"count": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")}
            flip_tr_stats = {"count": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")}
            flip_te_stats = {"count": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")}

            for t in range(self.num_trials):
                rng_tr = default_rng(self.trial_ss_train[t])
                rng_te = default_rng(self.trial_ss_test[t])
                Xn_tr = self._sample_noise_data(X_tr, q_f, rng=rng_tr)
                Xn_te = self._sample_noise_data(X_te, q_f, rng=rng_te)
                yp_tr = pred_fn(Xn_tr)
                yp_te = pred_fn(Xn_te)
                acc_tr = accuracy_score(y_tr, yp_tr)
                acc_te = accuracy_score(y_te, yp_te)
                flip_tr = np.mean(yp_tr != base_pred_tr)
                flip_te = np.mean(yp_te != base_pred_te)
                for stats, val in zip([acc_tr_stats, acc_te_stats, flip_tr_stats, flip_te_stats], [acc_tr, acc_te, flip_tr, flip_te]):
                    stats["count"] += 1
                    if val < stats["min"]:
                        stats["min"] = val
                    if val > stats["max"]:
                        stats["max"] = val
                    delta = val - stats["mean"]
                    stats["mean"] += delta / stats["count"]
                    stats["M2"] += delta * (val - stats["mean"])
            def finalize(stats):
                n = stats["count"]
                var = stats["M2"] / (n - 1) if n > 1 else 0.0
                std = np.sqrt(var)
                return stats["mean"], std, stats["min"], stats["max"]
            tr_acc_mean, tr_acc_std, tr_acc_min, _ = finalize(acc_tr_stats)
            te_acc_mean, te_acc_std, te_acc_min, _ = finalize(acc_te_stats)
            tr_flip_mean, tr_flip_std, _, tr_flip_max = finalize(flip_tr_stats)
            te_flip_mean, te_flip_std, _, te_flip_max = finalize(flip_te_stats)
            return {
                # f"train_stability_acc_mean{suffix}": tr_acc_mean,
                # f"train_stability_acc_std{suffix}": tr_acc_std,
                # f"train_stability_acc_min{suffix}": tr_acc_min,
                f"test_stability_acc_mean{suffix}": te_acc_mean,
                f"test_stability_acc_std{suffix}": te_acc_std,
                f"test_stability_acc_min{suffix}": te_acc_min,
                # f"train_flip_rate_mean{suffix}": tr_flip_mean,
                # f"train_flip_rate_std{suffix}": tr_flip_std,
                # f"train_flip_rate_max{suffix}": tr_flip_max,
                # f"test_flip_rate_mean{suffix}": te_flip_mean,
                # f"test_flip_rate_std{suffix}": te_flip_std,
                # f"test_flip_rate_max{suffix}": te_flip_max,
            }

        # Evaluate all variants
        result = {}
        result.update(eval_stability(qf_default, ""))
        result.update(eval_stability(qf_plus_01, "_plus0.1"))
        result.update(eval_stability(qf_minus_01, "_minus0.1"))
        result.update(eval_stability(qf_minus_02, "_minus0.2"))
        # For resample, use ±0.05
        qf_resample_005 = self._shift_noise_dist(self.qf, 0.05, resample=True, rng=default_rng(self.qf_resample_seed)) if self.qf is not None else np.full(self._X_tr.shape[1], 0.5, dtype=float)
        result.update(eval_stability(qf_resample_005, "_resample0.05"))
        return result

    def cleanup(self):
        # Remove any temporary memmap files
        if self._cache_dir and os.path.isdir(self._cache_dir):
            try:
                shutil.rmtree(self._cache_dir)
            except Exception:
                pass
        self._Xn_tr_mem = None
        self._Xn_te_mem = None
        self._Xn_tr_mm = None
        self._Xn_te_mm = None
        self._cache_ready = False

    def _prepare_noise_cache(self, X_tr, X_te, q_f):
        # Generate all trials once and store for reuse across models
        if self.cache_noise == "mem":
            self._Xn_tr_mem = np.empty((self.num_trials, self._ntr, self._d), dtype=self._dtype)
            self._Xn_te_mem = np.empty((self.num_trials, self._nte, self._d), dtype=self._dtype)
            for t in range(self.num_trials):
                rng_tr = default_rng(self.trial_ss_train[t])
                rng_te = default_rng(self.trial_ss_test[t])
                self._Xn_tr_mem[t] = self._sample_noise_data(X_tr, q_f, rng=rng_tr)
                self._Xn_te_mem[t] = self._sample_noise_data(X_te, q_f, rng=rng_te)
            self._cache_ready = True
        elif self.cache_noise == "disk":
            self._cache_dir = tempfile.mkdtemp(prefix="stability_noise_")
            tr_path = os.path.join(self._cache_dir, "Xn_tr.dat")
            te_path = os.path.join(self._cache_dir, "Xn_te.dat")
            self._Xn_tr_mm = np.memmap(tr_path, dtype=self._dtype, mode="w+",
                                       shape=(self.num_trials, self._ntr, self._d))
            self._Xn_te_mm = np.memmap(te_path, dtype=self._dtype, mode="w+",
                                       shape=(self.num_trials, self._nte, self._d))
            for t in range(self.num_trials):
                rng_tr = default_rng(self.trial_ss_train[t])
                rng_te = default_rng(self.trial_ss_test[t])
                self._Xn_tr_mm[t] = self._sample_noise_data(X_tr, q_f, rng=rng_tr)
                self._Xn_te_mm[t] = self._sample_noise_data(X_te, q_f, rng=rng_te)
            # Flush to ensure data is written
            self._Xn_tr_mm.flush()
            self._Xn_te_mm.flush()
            self._cache_ready = True

    def _shift_noise_dist(self, q_f, delta, resample=False, rng=None):
        q_f = q_f.copy()
        if resample:
            rng = self.rng if rng is None else rng
            q_f = rng.uniform(q_f - delta, q_f + delta, size=q_f.shape)
        else:
            q_f += delta
        return np.clip(q_f, np.nextafter(0.0, 1.0), 1.0)
    
    # https://github.com/D3M-Research-Group/odtlearn/blob/main/docs/notebooks/RobustOCT.ipynb
    def _sample_noise_data(self, X, q_f, rng=None):
        X = X.copy()
        rng = self.rng if rng is None else rng
        for f in range(X.shape[1]):
            signs = 2 * rng.binomial(1, 0.5, size=X.shape[0]) - 1
            # Use per-feature geometric parameter; ensure scalar p for each feature
            p = float(q_f[f]) if np.ndim(q_f) != 0 else float(q_f)
            perturbation = signs * (rng.geometric(p, size=X.shape[0]) - 1)
            X[:, f] += perturbation
        return X
    
    def _sample_all_noise_data(self, X, q_f, num_trials):
        all_X = np.zeros((num_trials, X.shape[0], X.shape[1]), dtype=X.dtype)
        for t in range(num_trials):
            all_X[t] = self._sample_noise_data(X, q_f)
        return all_X
    
    def _noise_eval(self, X, y, q_f, num_trials, pred_fn):
        # This helper is unused in the streaming compute; retained for reference/testing
        results = defaultdict(list)
        for _ in range(num_trials):
            X_noise = self._sample_noise_data(X, q_f)
            y_pred = pred_fn(X_noise)
            acc = accuracy_score(y, y_pred)
            results['accuracy'].append(float(acc))

        out = {}
        for key, vals in results.items():
            vals = np.asarray(vals, dtype=float)
            out[key] = float(np.mean(vals))
            out[key + '_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[key + '_min'] = float(np.min(vals))
        return out
