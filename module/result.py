""" result.py:  Class for saving and loading experiment results and plot/analysis utilities. """
from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import logging
from module.utils import NumpyEncoder

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Results")

@dataclass
class ReporterConfig:
    # IO roots for outputs (pass this at init)
    out_root: str
    fig_dir: str = "figures"
    table_dir: str = "tables"
    overwrite: bool = True

    # computed from out_root (not in __init__)
    model_dir: str = field(init=False)
    result_dir: str = field(init=False)
    param_dir: str = field(init=False)

    filename_regex: str = r"([\w-]+)_([\w-]+)_fold_(\d+)\.(json|pkl)"
    dataset_allowlist: Optional[tuple[str, ...]] = None  # None = allow all datasets; set to tuple to filter

    # Plot styling
    context: str = "talk"
    font_scale: float = 1.0
    palette: str = "tab20"
    fig_dpi: int = 300
    fig_size: Tuple[float, float] = (9, 6)
    fig_size_1d: Tuple[float, float] = (12, 4)
    fig_size_1d_with_legend: Tuple[float, float] = (15, 5)  # Extra size for bottom legend
    fig_size_1d_with_clean_legend: Tuple[float, float] = (15, 5)  # Extra size for bottom legend
    grid: bool = False
    use_kde_overlay: bool = True
    axvline_lw: float = 5.0
    axvline_alpha: float = 1.0
    # Marker/legend/tick styling
    marker_size: int = 70
    marker_alpha: float = 0.2
    legend_fontsize: float = 28
    legend_marker_size: float = 11.0
    yaxis_nbins: int = 5
    tick_labelsize: float = 28
    title_fontsize: float = 32
    label_fontsize: float = 28
    
    # Legend positioning for bottom legends
    legend_bbox_anchor_y: float = -0.25  # Y position for bbox_to_anchor
    legend_subplots_adjust_bottom: float = 0.28  # Bottom margin for subplots_adjust

    # Logging
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_datefmt: str = "%Y-%m-%d %H:%M:%S"

    # 2D KDE defaults
    kde_gridsize: int = 96
    kde_levels: int = 20
    kde_thresh: float = 0.01
    kde_bw_adjust: float = 0.7
    kde_cut: float = 0.0
    kde_quantiles: Tuple[float, float] = (0.005, 0.995)
    kde_pad_frac: float = 0.02
    kde_colorbar: bool = False
    kde_max_points: Optional[int] = 500_000

    drop_time_metrics: bool = False
    
    # Metrics to invert (apply 1 - X transformation)
    # These are typically fairness metrics where raw values represent violations
    invert_metrics: Tuple[str, ...] = (
        "Statistical Parity",
        "Equal Opportunity",
        "Equalized Odds",
        "MIA Label Unsupervised",
        "MIA Label Supervised",
        "MIA Rule Based",
        "MIA Blackbox",
    )
    
    # Metrics where LOWER is better (for bolding best values in tables)
    # Note: These are NOT inverted, just the direction for "best" is reversed
    # Do NOT include inverted metrics here - after inversion, higher is better
    lower_is_better_metrics: Tuple[str, ...] = (
        # Privacy metrics (lower attack accuracy is better)
        # "MIA Label Unsupervised",
        # "MIA Label Supervised",
        "MIA Rule Based",
        "MIA Blackbox",
        # Performance metrics (lower is better)
        "Train Time",
    )

    # Note, metrics not in this map are skipped for now.
    metric_name_map: Dict[str, str] = field(default_factory=lambda: {
        "test_demographic_parity_difference": "Statistical Parity",
        "test_equal_opportunity_difference": "Equal Opportunity",
        "test_equalized_odds_difference": "Equalized Odds",
        "sp_test_accuracy": "Statistical Parity Test Accuracy",
        "eopp_test_accuracy": "Equal Opportunity Test Accuracy",
        "eo_test_accuracy": "Equalized Odds Test Accuracy",
        "train_accuracy": "Train Accuracy",
        "test_accuracy": "Test Accuracy",
        "kantch_attack_adv_acc": "Test Adv Accuracy",
        # Stability metrics - standard
        "test_stability_acc_mean": "Stability Acc Mean",
        "test_stability_acc_min": "Stability Acc Worst",
        # Stability metrics - plus 0.1
        "test_stability_acc_mean_plus0.1": "Stability Acc Mean +0.1",
        "test_stability_acc_min_plus0.1": "Stability Acc Worst +0.1",
        # Stability metrics - minus 0.1
        "test_stability_acc_mean_minus0.1": "Stability Acc Mean -0.1",
        "test_stability_acc_min_minus0.1": "Stability Acc Worst -0.1",
        # Stability metrics - minus 0.2
        "test_stability_acc_mean_minus0.2": "Stability Acc Mean -0.2",
        "test_stability_acc_min_minus0.2": "Stability Acc Worst -0.2",
        # Stability metrics - resample 0.05
        "test_stability_acc_mean_resample0.05": "Stability Acc Mean Resample ± 0.05",
        "test_stability_acc_min_resample0.05": "Stability Acc Worst Resample ± 0.05",
        # Stability std variants (map to readable names)
        "test_stability_acc_std": "Stability Acc Std",
        "test_stability_acc_std_plus0.1": "Stability Acc Std +0.1",
        "test_stability_acc_std_minus0.1": "Stability Acc Std -0.1",
        "test_stability_acc_std_minus0.2": "Stability Acc Std -0.2",
        "test_stability_acc_std_resample0.05": "Stability Acc Std Resample ± 0.05",
        # Privacy metrics
        "mia_label_only_supervised_accuracy": "MIA Label Supervised",
        "mia_label_only_unsupervised_accuracy": "MIA Label Unsupervised",
        "mia_rule_based_accuracy": "MIA Rule Based",
        "mia_blackbox_accuracy": "MIA Blackbox",
        "train_time": "Train Time",
    })
    # Note, RSET models not in this map are skipped for now.
    model_name_map: Dict[str, str] = field(default_factory=lambda: {
        "treefarms": "RSET",
        "RSET_optimal_tree": "RSET_opt",
        "RSET_min_leaf_optimal_tree": "RSET_min",
        "RSET_max_leaf_optimal_tree": "RSET_max",
        "RSET_kantch": "RSET_kan",
        "RSET_default": "RSET_def",
        "RSET_fairness_sp": "RSET_sp",
        "RSET_fairness_eopp": "RSET_eopp",
        "RSET_fairness_eo": "RSET_eo",
    })
    metric_metric_map: Dict[tuple[str, str], tuple[str, str]] = field(default_factory=lambda: {
        ("Statistical Parity", "Test Accuracy"): ("Statistical Parity", "Statistical Parity Test Accuracy"),
        ("Equal Opportunity", "Test Accuracy"): ("Equal Opportunity", "Equal Opportunity Test Accuracy"),
        ("Equalized Odds", "Test Accuracy"): ("Equalized Odds", "Equalized Odds Test Accuracy"),
    })
    # All model must have a color else error
    model_color_map: Dict[str, str] = field(default_factory=lambda: {
        # Fair Baseline
        "post_rf": "#FF5C5C",
        "post_cart": "#ff69b4",
        "post_xgboost": "#ffa600",
        "dpf": "#be66e4",
        # RSET in blue tones
        "RSET": "#d6e2f3",
        "RSET_opt": "#2020df",
        "RSET_min": "#20afdf",
        "RSET_max": "#547ca6",
        "RSET_kan": "#20dfdf",
        # "RSET_default": "#5DADE2",
        # RSET fairness methods in purple/pink tones
        "RSET_sp": "#9B59B6",
        "RSET_eopp": "#D7BDE2",
        "RSET_eo": "#C39BD3",
        # Other baselines
        "roctv": "#f7c8d0",
        "fprdt": "#e98d83",
        "groot": "#e8735e",
        "roctn": "#df9c20",
        "cart": "#808080",
    })
    model_marker_map: Dict[str, str] = field(default_factory=lambda: {
        "treefarms": "o",
        "post_rf": "s",
        "post_cart": "D",
        "post_xgboost": "^",
        "dpf": "X",
        # Special RSET trees
        "RSET_optimal_tree": "o",
        "RSET_min_leaf_tree": "s",
        "RSET_max_leaf_tree": "D",
        "RSET_kantch": "^",
        "RSET_fairness_sp": "v",
        "RSET_fairness_eopp": "<",
        "RSET_fairness_eo": ">",
        "RSET_default": "p",
    })
    
    # Model ordering for tables (models not in this list appear alphabetically at the end)
    model_order: List[str] = field(default_factory=lambda: [
        # Baselines first
        "cart",
        "fprdt",
        "groot",
        "roctn",
        "roctv",
        # Priva baseline
        "bdpt",
        "dpldt",
        "priva",
        # Fair baselines
        "dpf",
        "post_cart",
        "post_rf",
        "post_xgboost",
        # RSET general
        "RSET",
        # RSET special trees
        "RSET_opt",
        "RSET_min",
        "RSET_max",
        "RSET_def",
        # RSET fairness
        "RSET_sp",
        "RSET_eopp",
        "RSET_eo",
        "RSET_kan",
    ])

    def __post_init__(self):
        # derive these AFTER out_root exists
        self.model_dir = os.path.join(self.out_root, "model")
        self.result_dir = os.path.join(self.out_root, "result")
        self.param_dir  = os.path.join(self.out_root, "param")

class ResultsReporter:
    def __init__(self, cfg: ReporterConfig | None = None, cache_dir: str | None = None, use_cache: bool = True,
                 source_reporters: Optional[List['ResultsReporter']] = None):
        self.cfg = cfg or ReporterConfig()
        sns.set_theme(style=("whitegrid" if self.cfg.grid else "white"))
        sns.set_context(self.cfg.context, font_scale=self.cfg.font_scale)
        sns.set_palette(self.cfg.palette)
        self._tidy: pd.DataFrame | None = None
        self._files_map: Dict[str, Any] | None = None  # Store files_map for lazy loading
        self._source_reporters: List['ResultsReporter'] = source_reporters or []  # For merged reporters
        
        # Initialize cache
        if cache_dir is None:
            cache_dir = os.path.join(self.cfg.out_root, ".cache")
        self.cache = CacheManager(cache_dir, enabled=use_cache)
        logger.info(f"CacheManager initialized: {self.cache.get_cache_info()}")

    # ---------------------------
    # Discovery / Load
    # ---------------------------
    def discover_files(self) -> Dict[str, Any]:
        """ Discover files from configured directories, returning a nested dict """
        assert self.cfg.model_dir and self.cfg.result_dir and self.cfg.param_dir, (
            "model_dir, result_dir, param_dir must be set in ReporterConfig to use discover_files()."
        )
        model_dir = os.fspath(self.cfg.model_dir)
        result_dir = os.fspath(self.cfg.result_dir)
        param_dir = os.fspath(self.cfg.param_dir)
        pattern = re.compile(self.cfg.filename_regex)

        model_files = {f for f in os.listdir(model_dir) if pattern.match(f)}
        result_files = {f for f in os.listdir(result_dir) if pattern.match(f)}
        param_files  = {f for f in os.listdir(param_dir)  if pattern.match(f)}

        def pick_result_name(model_raw: str, dataset: str, fold: str) -> str | None:
            base_json = f"{model_raw}_{dataset}_fold_{fold}.json"
            base_pkl  = f"{model_raw}_{dataset}_fold_{fold}.pkl"
            if base_json in result_files:
                return base_json
            if base_pkl in result_files:
                return base_pkl
            return None

        def pick_param_name(model_raw: str, dataset: str, fold: str) -> str | None:
            base_json = f"{model_raw}_{dataset}_fold_{fold}.json"
            base_pkl  = f"{model_raw}_{dataset}_fold_{fold}.pkl"
            if base_json in param_files:
                return base_json
            if base_pkl in param_files:
                return base_pkl
            return None

        files_map: Dict[str, Any] = {}
        for fname in model_files:
            m = pattern.match(fname)
            if not m:
                continue
            model_raw, dataset, fold, _ext = m.groups()
            if self.cfg.dataset_allowlist and dataset not in self.cfg.dataset_allowlist:
                logger.warning("Skipping disallowed dataset: %s", dataset)
                continue

            rname = pick_result_name(model_raw, dataset, fold)
            if rname is None:
                logger.warning("No result file found for model %s dataset %s fold %s", model_raw, dataset, fold)
                continue
            pname = pick_param_name(model_raw, dataset, fold)

            files_map.setdefault(dataset, {}).setdefault(model_raw, {}).setdefault(str(fold), {})
            files_map[dataset][model_raw][str(fold)]["model"] = os.path.join(model_dir, fname)
            files_map[dataset][model_raw][str(fold)]["result"] = os.path.join(result_dir, rname)
            if pname is not None:
                files_map[dataset][model_raw][str(fold)]["param"] = os.path.join(param_dir, pname)
            else:
                logger.info("No param file found for model %s dataset %s fold %s", model_raw, dataset, fold)
        return files_map

    def load(self, files_map: Dict[str, Any], use_cache: bool = True) -> pd.DataFrame:
        """
        Parse file map → tidy DataFrame with baselines + RSET special trees.
        RSET tree distributions are cached separately in NPZ files for on-demand loading.
        
        ARCHITECTURE:
        - DataFrame: Baselines + RSET special trees only (one row per model/dataset/fold/metric)
        - NPZ cache: Full RSET tree distributions (loaded lazily for density plots)
        
        Args:
            files_map: Nested dict of file paths from discover_files()
            use_cache: Whether to use cache for loading (default True)
                  
        Returns:
            DataFrame with baselines and RSET special trees
        """
        # Store files_map for lazy loading of heavy data
        self._files_map = files_map
        
        # Try loading from cache first
        if use_cache:
            cached_df = self.cache.load_tidy_df()
            if cached_df is not None:
                self._tidy = cached_df
                logger.info(f"Loaded {len(cached_df)} rows from cache")
                return cached_df
        
        # Cache miss or disabled - load from files
        logger.info("Loading data (baselines + RSET special trees)")
        tidy = self._load_data(files_map, use_cache=use_cache)

        if "dpf" in tidy.model.unique():
            # DPF only has Statistica Parity Test Accuracy, rename it to Test Accuracy for consistency
            mask = (tidy.model == "dpf") & (tidy.metric == "Statistical Parity Test Accuracy")
            tidy.loc[mask, "metric"] = "Test Accuracy"

        
        self._tidy = tidy
        
        # Cache the loaded data
        if use_cache and not tidy.empty:
            self.cache.cache_tidy_df(tidy)
        
        return tidy
    
    def _load_data(self, files_map: Dict[str, Any], use_cache: bool = True) -> pd.DataFrame:
        """
        Load data: baselines + RSET special trees (one row per model/dataset/fold/metric).
        RSET tree distributions are cached separately in NPZ files for on-demand loading.
        
        Args:
            files_map: Nested dict of file paths
            use_cache: Whether to cache RSET distributions
            
        Returns:
            DataFrame with baselines and RSET special trees
        """
        rows: List[Dict[str, Any]] = []
        self._special_tree_info = {}  # Cache for special tree mappings
        
        for root_key, maybe_datasets in self._iter_roots(files_map):
            if isinstance(maybe_datasets, dict):
                for dataset, models in maybe_datasets.items():
                    if not isinstance(models, dict):
                        raise ValueError(f"Unexpected structure in files_map at dataset {dataset}")
                    
                    for model, payload in models.items():
                        model_disp = self.cfg.model_name_map.get(model, model)
                        
                        # Process folds
                        folds_data = []
                        folds_data = [(fold, bundle["result"], bundle.get("model")) 
                                        for fold, bundle in payload.items() 
                                        if isinstance(bundle, dict) and "result" in bundle]
                        
                        for fold, result_path, model_path in folds_data:
                            result = self._read_json_or_pkl(result_path)
                            
                            is_rset = ("treefarms" == model)
                            
                            # Extract special trees + cache full distributions
                            if is_rset:
                                if dataset == "adult":
                                    continue
                                special_tree_rows, distributions = self._extract_rset_data(
                                    result, root_key, dataset, model_disp, fold
                                )
                                rows += special_tree_rows
                                if use_cache and distributions:
                                    logger.info(f"Caching {len(distributions)} metrics with full RSET distributions "
                                               f"for {dataset} fold {fold} "
                                               f"(~{len(next(iter(distributions.values())))} trees per metric)")
                                    for metric, values in distributions.items():
                                        self.cache.cache_dataset_metric(
                                            dataset=f"{dataset}_fold_{fold}",
                                            metric=f"RSET_trees_{metric}",
                                            values=values
                                        )
                                elif not distributions:
                                    logger.warning(f"No distributions to cache for {dataset} fold {fold} "
                                                f"(small evaluation or special trees only)")
                            else:
                                rows += self._rows_from_result(result, {
                                    "root": root_key, "dataset": dataset, 
                                    "model": model_disp, "fold": fold
                                })
            else:
                raise ValueError(f"Unexpected structure in files_map at root {root_key}")
        tidy = pd.DataFrame(rows)
        if not tidy.empty:
            tidy["fold"] = tidy["fold"].astype(str)
        
        logger.info(f"Loaded data: {len(tidy)} rows (baselines + RSET special trees)")
        return tidy
    

    # ---------------------------
    # Tables / CSV
    # ---------------------------
    def _aggregate_model_data(self, metric_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate model data to compute mean/std/n per (dataset, model).
        Handles both baseline models and RSET special trees.
        
        Args:
            metric_df: DataFrame filtered to a single metric
            
        Returns:
            DataFrame with columns: dataset, model, mean_value, std_value, n
        """
        parts: List[pd.DataFrame] = []

        # 1) Non-RSET models: aggregate mean/std/n per (dataset, model)
        non_rset = metric_df[metric_df.model != "RSET"]
        if not non_rset.empty:
            def _agg_with_fail_df(group: pd.DataFrame) -> pd.Series:
                vals = group["value"].astype(float).to_numpy()
                n = len(vals)
                if np.any(vals == -1):
                    return pd.Series({
                        "mean_value": -1.0,
                        "std_value": -1.0,
                        "n": n,
                    })
                mu = float(vals.mean()) if n > 0 else np.nan
                sd = float(vals.std(ddof=1)) if n > 1 else 0.0
                return pd.Series({
                    "mean_value": mu,
                    "std_value": sd,
                    "n": n,
                })

            agg_non_rset = (
                non_rset.groupby(["dataset", "model"])  # type: ignore[arg-type]
                .apply(_agg_with_fail_df, include_groups=False)
                .reset_index()
            )
            parts.append(agg_non_rset)

        # 2) RSET special trees: one row per special index key
        rset_df = metric_df[metric_df.model == "RSET"]
        if not rset_df.empty:
            rows_special: List[Dict[str, Any]] = []
            for dataset in rset_df["dataset"].unique():
                ds_df = rset_df[rset_df.dataset == dataset]
                # Collect per special key values across folds
                val_map: Dict[str, List[float]] = {}
                logger.info(f"[_aggregate_model_data] Processing RSET data for dataset={dataset}")
                for fold in ds_df["fold"].unique():
                    fd = ds_df[ds_df.fold == fold]
                    if fd.empty:
                        continue
                    row0 = fd.iloc[0]
                    special_positions = row0.get("special_tree_indices", None)
                    eval_indices = row0.get("eval_indices", None)
                    if not special_positions or not eval_indices:
                        logger.warning("Missing special_tree_indices/eval_indices for %s fold %s; skipping", dataset, fold)
                        continue
                    logger.info(f"[_aggregate_model_data] dataset={dataset}, fold={fold}, special_positions keys: {list(special_positions.keys())}")
                    for tree_name, raw_idx in special_positions.items():
                        prefixed_name = tree_name if str(tree_name).startswith("RSET_") else f"RSET_{tree_name}"
                        display_name = self.cfg.model_name_map.get(prefixed_name, prefixed_name)
                        if raw_idx in eval_indices:
                            pos = eval_indices.index(raw_idx)
                            vr = fd[fd.idx == pos]
                            if not vr.empty:
                                v = float(vr.iloc[0]["value"])
                                val_map.setdefault(display_name, []).append(v)
                            else:
                                logger.warning(f"[_aggregate_model_data] Empty value row for {display_name} at dataset={dataset}, fold={fold}, idx={pos}")
                        else:
                            logger.warning(f"[_aggregate_model_data] raw_idx {raw_idx} for {display_name} not in eval_indices for dataset={dataset}, fold={fold}")
                
                # Log what we collected for this dataset
                logger.info(f"[_aggregate_model_data] dataset={dataset}, collected tree keys: {list(val_map.keys())}")
                
                # Reduce to mean/std/n rows
                for tree_key, vals in val_map.items():
                    if len(vals) == 0:
                        logger.warning(f"[_aggregate_model_data] No values collected for {tree_key} at dataset={dataset}")
                        continue

                    logger.warning(f"[_aggregate_model_data] Collected {len(vals)} values for {tree_key} at dataset={dataset}: {vals}")

                    # Failure sentinel handling: if any -1 appears among values, flag as -1
                    if any((v == -1) for v in vals):
                        rows_special.append({
                            "dataset": dataset,
                            "model": tree_key,
                            "mean_value": -1.0,
                            "std_value": -1.0,
                            "n": len(vals),
                        })
                    else:
                        rows_special.append({
                            "dataset": dataset,
                            "model": tree_key,
                            "mean_value": float(np.mean(vals)),
                            "std_value": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "n": len(vals),
                        })
            if rows_special:
                parts.append(pd.DataFrame(rows_special))

        # 3) Concatenate all parts
        if parts:
            return pd.concat(parts, ignore_index=True)
        else:
            return pd.DataFrame(columns=["dataset", "model", "mean_value", "std_value", "n"])
    
    def get_metric_data(self, metric: str, dataset: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Get data for a specific metric, optionally filtered by dataset.
        Uses cache if available to avoid loading full tidy DataFrame.
        Supports lazy loading mode for memory efficiency.
        
        Args:
            metric: Metric name
            dataset: Optional dataset name to filter
            use_cache: Whether to use cache (default True)
            
        Returns:
            DataFrame filtered to metric (and optionally dataset)
        """
        # If full tidy DataFrame is already loaded, use it
        if self._tidy is not None and not self._tidy.empty and "metric" in self._tidy.columns:
            dfm = self._tidy[self._tidy.metric == metric]

            if dfm.empty:
                raise ValueError(f"No data found for metric {metric} in loaded DataFrame, available metrics: {self._tidy['metric'].unique().tolist()}")
            if dataset:
                dfm = dfm[dfm.dataset == dataset]
            return dfm
        
        
        # Otherwise, try to load from cache or files
        if self._files_map is None:
            raise RuntimeError("No data loaded. Call .load(files_map) first.")
        
        # Try cache first (per-metric or aggregated)
        if use_cache:
            # Check for cached aggregated data (already computed mean/std/n)
            agg_df = self.cache.load_aggregated(metric)
            if agg_df is not None:
                logger.info(f"Loaded aggregated cache for metric {metric}")
                if dataset:
                    agg_df = agg_df[agg_df.dataset == dataset]
                return agg_df
        
        # Load from files - this will load all data if not already loaded
        logger.info(f"Loading data for metric {metric}" + (f" for dataset {dataset}" if dataset else ""))
        if self._tidy is None or self._tidy.empty:
            # Need to load all data first
            self._tidy = self._load_data(self._files_map, use_cache=use_cache)
        
        # Filter to requested metric and dataset
        dfm = self._tidy[self._tidy.metric == metric]
        if dataset:
            dfm = dfm[dfm.dataset == dataset]
        
        return dfm
    
    def build_csvs(self, 
                   metrics: Optional[List[str]] = None,
                   models: Optional[List[str]] = None,
                   datasets: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Write one CSV per metric with mean/std/n and formatted mean±std. Returns {metric: path}.
        Supports lazy loading mode for memory efficiency.
        
        Args:
            metrics: Optional list of specific metrics to process. If None, processes all metrics.
            models: Optional list of models to include. If None, includes all models.
            datasets: Optional list of datasets to include. If None, includes all datasets.
            
        Returns:
            Dict mapping metric names to CSV file paths
        """
        out: Dict[str, str] = {}
        
        # Determine which metrics to process
        if metrics is None:
            # Need to discover all metrics - either from tidy or from files
            if self._tidy is not None and not self._tidy.empty:
                all_metrics = self._tidy["metric"].unique()
            elif self._files_map is not None:
                # Load data to discover metrics
                logger.info("Discovering available metrics from files...")
                sample_df = self._load_data(self._files_map, use_cache=True)
                all_metrics = sample_df["metric"].unique() if not sample_df.empty else []
                # Clear sample data to free memory
                del sample_df
            else:
                raise RuntimeError("No data loaded. Call .load(files_map) first.")
        else:
            all_metrics = metrics
        
        logger.info(f"Building CSVs for {len(all_metrics)} metrics")
        
        for metric in all_metrics:
            # Load metric data (uses cache if available)
            dfm = self.get_metric_data(metric, use_cache=True)
            
            if dfm.empty:
                logger.warning(f"No data found for metric {metric}")
                continue
            
            # Only log if no data; skip per-metric info log
            
            # Filter datasets if specified
            if datasets is not None:
                before_count = len(dfm)
                dfm = dfm[dfm['dataset'].isin(datasets)]
                logger.info(f"  After dataset filter: {len(dfm)} rows (was {before_count})")
                if dfm.empty:
                    logger.warning(f"No data for metric {metric} with specified datasets")
                    continue
            
            # Filter models if specified
            if models is not None:
                before_count = len(dfm)
                available_models = set(dfm['model'].unique())
                requested_models = set(models)
                logger.info(f"  Filtering models: requested={requested_models}, available={available_models}")
                dfm = dfm[dfm['model'].isin(models)]
                logger.info(f"  After model filter: {len(dfm)} rows (was {before_count}), remaining models: {sorted(dfm['model'].unique())}")
                if dfm.empty:
                    logger.warning(f"No data for metric {metric} with specified models. Requested: {models}, Available: {available_models}")
                    continue
            
            # Use aggregation helper
            out_df = self._aggregate_model_data(dfm)
            logger.info(f"  After aggregation: {len(out_df)} rows, models: {sorted(out_df['model'].unique())}")

            # Check for missing results: warn if any model is missing 5 results (should have 5 per dataset/model)
            group = dfm.groupby(['dataset', 'model']).size()
            for (dataset, model), count in group.items():
                if count < 5:
                    logger.warning(f"Model '{model}' on dataset '{dataset}' has only {count}/5 results for metric '{metric}'")
            
            # Check if this is a privacy metric (for special -1 handling)
            is_privacy_metric = any(privacy_key in metric.lower() for privacy_key in ['mia', 'privacy', 'attack'])
            
            # Format column: if -1 (attack failed for privacy metrics), show "Attk Failed"; otherwise mean ± std
            def _fmt_row(r: pd.Series) -> str:
                if pd.isna(r.get("mean_value")):
                    return ""
                try:
                    mv = float(r["mean_value"]) if "mean_value" in r else np.nan
                    sv = float(r["std_value"]) if "std_value" in r else np.nan
                except Exception:
                    return ""
                if mv == -1.0:
                    return "Attk Failed" if is_privacy_metric else "-1"
                if sv == -1.0:
                    return "-1"
                return f"{mv:.3f} ± {sv:.3f}"
            
            out_df["mean±std"] = out_df.apply(_fmt_row, axis=1)

            out_dir = self._ensure_dir(self._path_table())
            path = os.path.join(out_dir, f"summary_{metric}.csv")
            logger.info("Writing summary CSV for metric %s to %s", metric, path)
            out_df.to_csv(path, index=False)
            out[metric] = path
            
            # Cache aggregated data
            self.cache.cache_aggregated(metric, out_df)
            
            # Free memory after processing each metric
            del dfm, out_df

        return out
    
    def _sort_models_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort DataFrame columns (models) according to cfg.model_order.
        Models not in model_order appear alphabetically at the end.
        """
        current_models = list(df.columns)
        
        # Split into ordered and unordered
        ordered = [m for m in self.cfg.model_order if m in current_models]
        unordered = sorted([m for m in current_models if m not in self.cfg.model_order])
        
        # Combine: ordered first, then unordered alphabetically
        sorted_models = ordered + unordered
        
        return df[sorted_models]
    
    def _sort_models_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort DataFrame index (models) according to cfg.model_order.
        Models not in model_order appear alphabetically at the end.
        """
        current_models = list(df.index)
        
        # Split into ordered and unordered
        ordered = [m for m in self.cfg.model_order if m in current_models]
        unordered = sorted([m for m in current_models if m not in self.cfg.model_order])
        
        # Combine: ordered first, then unordered alphabetically
        sorted_models = ordered + unordered
        
        return df.loc[sorted_models]
    
    def build_latex_tables(self, metrics: Optional[List[str]] = None, 
                          orientation: str = "datasets_rows",
                          bold_best: bool = True,
                          higher_is_better: Optional[Dict[str, bool]] = None,
                          models: Optional[List[str]] = None,
                          datasets: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate LaTeX tables with booktabs formatting for specified metrics.
        
        Args:
            metrics: List of metrics to generate tables for. If None, processes all metrics.
            orientation: Table orientation:
                - "datasets_rows": Rows=datasets, Columns=models (default)
                - "models_rows": Rows=models, Columns=datasets
            bold_best: Whether to bold best values (including ties) (default True)
            higher_is_better: Dict mapping metric names to bool indicating if higher is better.
                            If None, assumes higher is better for all metrics except those in lower_is_better_metrics.
            models: List of models to include in tables. If None, includes all models.
                   Models are ordered according to cfg.model_order, with unlisted models appended alphabetically.
            datasets: List of datasets to include in tables. If None, includes all datasets.
                     Datasets are sorted alphabetically.
            
        Returns:
            Dict mapping metric names to LaTeX file paths
        """
        out: Dict[str, str] = {}
        
        # Determine which metrics to process
        if metrics is None:
            if self._tidy is not None and not self._tidy.empty:
                all_metrics = self._tidy["metric"].unique()
            elif self._files_map is not None:
                logger.info("Discovering available metrics from files...")
                sample_df = self._load_data(self._files_map, use_cache=True)
                all_metrics = sample_df["metric"].unique() if not sample_df.empty else []
                del sample_df
            else:
                raise RuntimeError("No data loaded. Call .load(files_map) first.")
        else:
            all_metrics = metrics
        
        logger.info(f"Building LaTeX tables for {len(all_metrics)} metrics")
        
        # Set up higher_is_better defaults
        if higher_is_better is None:
            higher_is_better = {}
        
        for metric in all_metrics:
            # Load metric data
            dfm = self.get_metric_data(metric, use_cache=True)
            if dfm.empty:
                logger.warning(f"No data found for metric {metric}")
                continue
            # Use aggregation helper to get mean/std/n
            agg_df = self._aggregate_model_data(dfm)
            
            if agg_df.empty:
                logger.warning(f"No aggregated data for metric {metric}")
                continue
            
            # Determine if higher is better for this metric
            is_higher_better = higher_is_better.get(metric)
            if is_higher_better is None:
                # Default: higher is better unless metric is in lower_is_better_metrics
                is_higher_better = metric not in self.cfg.lower_is_better_metrics
            
            # Check if this is a privacy metric (for special -1 handling)
            is_privacy_metric = any(privacy_key in metric.lower() for privacy_key in ['mia', 'privacy', 'attack'])
            
            # Create formatted strings with mean ± std
            # Special handling: -1 means "attack failed" for privacy metrics (best possible result)
            def format_value(row):
                mv = row['mean_value']
                sv = row['std_value']
                if pd.isna(mv):
                    return "-"
                if mv == -1.0:
                    return "\\textbf{Attk Failed}" if is_privacy_metric else "-"
                return f"{mv:.3f} $\\pm$ {sv:.3f}"
            
            agg_df['formatted'] = agg_df.apply(format_value, axis=1)
            
            # Debug: Log aggregated data before filtering
            logger.info(f"[{metric}] Aggregated data: {len(agg_df)} rows")
            logger.info(f"[{metric}] Unique models in agg_df: {sorted(agg_df['model'].unique())}")
            logger.info(f"[{metric}] Unique datasets in agg_df: {sorted(agg_df['dataset'].unique())}")
            
            # Filter datasets if specified
            if datasets is not None:
                agg_df = agg_df[agg_df['dataset'].isin(datasets)]
                if agg_df.empty:
                    logger.warning(f"No data for specified datasets: {datasets}")
                    continue
            
            # Filter models if specified
            if models is not None:
                agg_df = agg_df[agg_df['model'].isin(models)]
                if agg_df.empty:
                    logger.warning(f"No data for specified models: {models}")
                    continue
            
            # Pivot table based on orientation
            if orientation == "datasets_rows":
                pivot_df = agg_df.pivot(index='dataset', columns='model', values='formatted')
            elif orientation == "models_rows":
                pivot_df = agg_df.pivot(index='model', columns='dataset', values='formatted')
            else:
                raise ValueError(f"Invalid orientation: {orientation}. Must be 'datasets_rows' or 'models_rows'")
            
            # Debug: Log pivot table shape and check for missing values
            logger.info(f"[{metric}] Pivot table shape: {pivot_df.shape}")
            logger.info(f"[{metric}] Pivot columns (models): {list(pivot_df.columns)}")
            logger.info(f"[{metric}] Pivot index (datasets): {list(pivot_df.index)}")
            
            # Check for NaN/empty cells and log them
            for idx in pivot_df.index:
                for col in pivot_df.columns:
                    cell_val = pivot_df.loc[idx, col]
                    if pd.isna(cell_val) or cell_val == "" or cell_val == "-":
                        logger.warning(f"[{metric}] Missing/empty value at [{idx}, {col}]: '{cell_val}'")
            
            # Sort rows and columns
            # Datasets are always sorted alphabetically
            if orientation == "datasets_rows":
                # Rows = datasets (alphabetical), Columns = models (custom order)
                pivot_df = pivot_df.sort_index()  # Sort datasets alphabetically
                pivot_df = self._sort_models_columns(pivot_df)
            else:
                # Rows = models (custom order), Columns = datasets (alphabetical)
                pivot_df = pivot_df[sorted(pivot_df.columns)]  # Sort datasets alphabetically
                pivot_df = self._sort_models_index(pivot_df)
            
            # Set index name for first column header (dataset or model depending on orientation)
            if orientation == "datasets_rows":
                pivot_df.index.name = "Dataset"
            else:
                pivot_df.index.name = "Model"
            
            # Bold best values (including ties)
            if bold_best:
                # Get numeric values for comparison
                numeric_pivot = agg_df.pivot(index='dataset' if orientation == "datasets_rows" else 'model',
                                            columns='model' if orientation == "datasets_rows" else 'dataset',
                                            values='mean_value')
                
                # For each row, find best value(s)
                for idx in pivot_df.index:
                    row_values = numeric_pivot.loc[idx]
                    valid_values = row_values.dropna()
                    
                    if valid_values.empty:
                        continue
                    
                    # Special handling for -1.0 in privacy metrics (attack failed = best)
                    has_attack_failed = (valid_values == -1.0).any()
                    
                    if is_privacy_metric and has_attack_failed:
                        # For privacy metrics, -1.0 (attack failed) is always the best value
                        best_val = -1.0
                    else:
                        # Filter out -1.0 values for normal comparison
                        valid_values_filtered = valid_values[valid_values != -1.0]
                        if valid_values_filtered.empty:
                            continue
                        
                        # Find best value (excluding -1.0)
                        if is_higher_better:
                            best_val = valid_values_filtered.max()
                        else:
                            best_val = valid_values_filtered.min()
                    
                    # Bold all values that match best (handles ties)
                    tolerance = 1e-6  # Small tolerance for floating point comparison
                    for col in pivot_df.columns:
                        cell_val = numeric_pivot.loc[idx, col]
                        if pd.notna(cell_val):
                            # Check if this is the best value
                            is_best = False
                            if is_privacy_metric and cell_val == -1.0:
                                # -1.0 is always best for privacy metrics
                                is_best = True
                            elif cell_val != -1.0 and abs(cell_val - best_val) < tolerance:
                                # Normal best value comparison (excluding -1.0)
                                is_best = True
                            
                            if is_best:
                                # Bold this cell (may already be bold if it's "Attk Failed")
                                current_text = pivot_df.loc[idx, col]
                                if current_text != "-" and not current_text.startswith("\\textbf"):
                                    pivot_df.loc[idx, col] = f"\\textbf{{{current_text}}}"
            
            # Uppercase baseline models (non-RSET models) and escape underscores for LaTeX
            if orientation == "datasets_rows":
                # Columns are models - uppercase baselines
                pivot_df.columns = [
                    str(col).upper().replace('_', r'\_') if not str(col).startswith('RSET')
                    else str(col).replace('_', r'\_')
                    for col in pivot_df.columns
                ]
                # Rows are datasets - just escape
                pivot_df.index = [str(idx).replace('_', r'\_') for idx in pivot_df.index]
            else:
                # Rows are models - uppercase baselines
                pivot_df.index = [
                    str(idx).upper().replace('_', r'\_') if not str(idx).startswith('RSET')
                    else str(idx).replace('_', r'\_')
                    for idx in pivot_df.index
                ]
                # Columns are datasets - just escape
                pivot_df.columns = [str(col).replace('_', r'\_') for col in pivot_df.columns]
            
            # Convert to LaTeX with booktabs
            latex_str = pivot_df.to_latex(
                escape=False,  # Don't escape LaTeX commands
                column_format='l' + 'c' * len(pivot_df.columns),  # Left-align row names, center data
                caption=f"Results for {metric}",
                label=f"tab:{metric.lower().replace(' ', '_')}",
                position='htbp',
                multicolumn=False,
                multirow=False,
                na_rep='-',
                index_names=True,  # Include index name in header row
            )
            
            # Replace default tabular lines with booktabs
            latex_str = latex_str.replace('\\begin{tabular}', '\\begin{tabular}')
            latex_str = latex_str.replace('\\toprule', '\\toprule')
            latex_str = latex_str.replace('\\midrule', '\\midrule')
            latex_str = latex_str.replace('\\bottomrule', '\\bottomrule')
            
            # Save to file
            out_dir = self._ensure_dir(self._path_table())
            safe_metric_name = metric.replace(' ', '_').replace('/', '_')
            path = os.path.join(out_dir, f"latex_{safe_metric_name}_{orientation}.tex")
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(latex_str)
            
            logger.info(f"Wrote LaTeX table for {metric} to {path}")
            out[metric] = path
            
            # Free memory
            del dfm, agg_df, pivot_df

        return out
    
    def build_rset_outperformance_table(self, 
                                        metrics: Optional[List[str]] = None,
                                        baselines: Optional[List[str]] = None,
                                        datasets: Optional[List[str]] = None,
                                        output_formats: List[str] = ['csv', 'md', 'latex']) -> Dict[str, str]:
        """
        Calculate percentage of RSET trees that outperform each baseline and ALL baselines.
        
        For each (dataset, metric) combination, calculates:
        - % of RSET trees that beat each baseline's mean performance
        - % of RSET trees that beat ALL baselines simultaneously
        
        Args:
            metrics: List of metrics to analyze. If None, uses all available metrics.
            baselines: List of baseline models to compare against. If None, uses common baselines.
            datasets: List of datasets to analyze. If None, uses all available datasets.
            output_formats: List of output formats ('csv', 'md', 'latex'). Default: all three.
            
        Returns:
            Dict mapping format names to file paths
            
        Example output (for one dataset/metric):
            Dataset | Metric | vs_cart | vs_groot | vs_roctn | vs_ALL
            compas  | Test Adv Acc | 78.3% | 65.2% | 89.1% | 52.4%
        """
        # Default baselines if not specified
        if baselines is None:
            baselines = ['cart', 'fprdt', 'groot', 'roctn', 'roctv']
        
        # Determine metrics to process
        if metrics is None:
            if self._tidy is not None and not self._tidy.empty:
                all_metrics = sorted(self._tidy["metric"].unique())
            else:
                logger.warning("No tidy data available, loading from files")
                # Load a sample to get metrics
                if self._files_map:
                    sample_df = self._load_data(self._files_map, use_cache=True)
                    all_metrics = sorted(sample_df["metric"].unique())
                else:
                    raise ValueError("No data available to determine metrics")
        else:
            all_metrics = metrics
        
        # Determine datasets
        if datasets is None:
            if self._tidy is not None and not self._tidy.empty:
                all_datasets = sorted(self._tidy["dataset"].unique())
            else:
                if self._files_map:
                    sample_df = self._load_data(self._files_map, use_cache=True)
                    all_datasets = sorted(sample_df["dataset"].unique())
                else:
                    raise ValueError("No data available to determine datasets")
        else:
            all_datasets = datasets
        
        logger.info(f"Calculating RSET outperformance for {len(all_metrics)} metrics x {len(all_datasets)} datasets")
        
        # Collect results
        results = []
        
        for dataset in all_datasets:
            for metric in all_metrics:
                logger.info(f"Processing {dataset}, {metric}")
                
                # Determine if higher is better
                higher_is_better = metric not in self.cfg.lower_is_better_metrics
                
                # Load baseline performance per fold
                dfm = self.get_metric_data(metric, dataset=dataset, use_cache=True)
                
                baseline_fold_values = {}  # {baseline: {fold: value}}
                for baseline in baselines:
                    baseline_data = dfm[dfm['model'] == baseline]
                    
                    if baseline_data.empty:
                        logger.warning(f"  No data for dataset {dataset}, metric {metric}, baseline {baseline}")
                        continue
                    
                    # Extract per-fold values
                    fold_values = {}
                    if 'value' in baseline_data.columns and 'fold' in baseline_data.columns:
                        # Raw format: group by fold
                        for fold, group in baseline_data.groupby('fold'):
                            if baseline.startswith('post') and len(group) >= 5:
                                fold_values[str(fold)] = group['value'].iloc[4]
                            elif baseline == "dpf" and len(group) >= 4:
                                fold_values[str(fold)] = group['value'].iloc[3]
                            else:
                                fold_values[str(fold)] = group['value'].mean()
                                logger.warning(f"Baseline {baseline} fold {fold} mean value used: {fold_values[str(fold)]}")

                    elif 'mean_value' in baseline_data.columns:
                        # Aggregated format: single value (assume it's the mean across folds)
                        # This is less ideal but we'll use it as a single value
                        mean_val = baseline_data['mean_value'].iloc[0]
                        for fold in range(5):
                            fold_values[str(fold)] = mean_val
                    
                    if fold_values:
                        baseline_fold_values[baseline] = fold_values
                        logger.info(f"  {baseline}: loaded {len(fold_values)} folds")
                
                if not baseline_fold_values:
                    logger.warning(f"No baselines available for {dataset}, {metric}")
                    continue
                
                # Per-fold comparison: for each fold, compare RSET trees to baseline in that fold
                fold_outperformance = {baseline: [] for baseline in baseline_fold_values.keys()}
                fold_outperformance['ALL'] = []
                fold_n_trees = []
                
                for fold in range(5):  # Assume 5-fold CV
                    fold_str = str(fold)
                    
                    # Load RSET distribution for this fold
                    fold_dist = self._load_rset_distribution(dataset, fold_str, metric)
                    if fold_dist is None or len(fold_dist) == 0:
                        logger.warning(f"  No RSET distribution for dataset {dataset} fold {fold}")
                        # raise ValueError(f"No RSET distribution for dataset {dataset} fold {fold}")
                        continue
                    
                    n_trees_fold = len(fold_dist)
                    fold_n_trees.append(n_trees_fold)
                    
                    # Compare to baselines in this fold
                    beats_all_fold = np.ones(n_trees_fold, dtype=bool)
                    
                    for baseline, fold_vals in baseline_fold_values.items():
                        if fold_str not in fold_vals:
                            logger.warning(f"  Missing fold {fold} for baseline {baseline}")
                            continue
                        
                        baseline_val = fold_vals[fold_str]
                        
                        if higher_is_better:
                            beats_baseline = fold_dist > baseline_val
                        else:
                            beats_baseline = fold_dist < baseline_val
                        
                        pct_fold = 100.0 * beats_baseline.sum() / n_trees_fold
                        fold_outperformance[baseline].append(pct_fold)
                        
                        # Update beats_all for this fold
                        beats_all_fold &= beats_baseline
                    
                    # ALL baselines for this fold
                    pct_all_fold = 100.0 * beats_all_fold.sum() / n_trees_fold
                    fold_outperformance['ALL'].append(pct_all_fold)
                
                if not fold_n_trees:
                    logger.warning(f"No folds with RSET data for {dataset}, {metric}")
                    continue
                
                # Aggregate across folds: compute mean ± std
                row = {
                    'dataset': dataset,
                    'metric': metric,
                    'n_folds': len(fold_n_trees),
                    'n_trees_mean': np.mean(fold_n_trees),
                    'higher_is_better': higher_is_better
                }
                
                # Individual baseline comparisons: mean ± std
                for baseline, pcts in fold_outperformance.items():
                    if not pcts:
                        continue
                    
                    mean_pct = np.mean(pcts)
                    std_pct = np.std(pcts, ddof=1) if len(pcts) > 1 else 0.0
                    
                    if baseline == 'ALL':
                        row['vs_ALL_mean'] = mean_pct
                        row['vs_ALL_std'] = std_pct
                    else:
                        row[f'vs_{baseline}_mean'] = mean_pct
                        row[f'vs_{baseline}_std'] = std_pct
                    
                    logger.info(f"  {baseline}: {mean_pct:.2f}% ± {std_pct:.2f}%")
                
                results.append(row)
        
        if not results:
            logger.warning("No results computed")
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns for readability
        base_cols = ['dataset', 'metric', 'n_folds', 'n_trees_mean', 'higher_is_better']
        vs_cols = [c for c in df.columns if c.startswith('vs_')]
        df = df[base_cols + vs_cols]
        
        # Create formatted columns for display (mean ± std)
        df_formatted = df[base_cols].copy()
        
        # Get unique baselines from column names
        baselines_in_cols = set()
        for col in vs_cols:
            if col.endswith('_mean'):
                baseline = col.replace('vs_', '').replace('_mean', '')
                baselines_in_cols.add(baseline)
        
        # Create formatted columns
        for baseline in sorted(baselines_in_cols):
            mean_col = f'vs_{baseline}_mean'
            std_col = f'vs_{baseline}_std'
            
            if mean_col in df.columns and std_col in df.columns:
                df_formatted[f'vs_{baseline}'] = df.apply(
                    lambda row: f"{row[mean_col]:.1f} ± {row[std_col]:.1f}",
                    axis=1
                )
        
        # Save to files
        out_dir = self._ensure_dir(self._path_table())
        output_paths = {}
        
        if 'csv' in output_formats:
            csv_path = os.path.join(out_dir, 'rset_outperformance.csv')
            # Save raw data (with mean/std as separate columns) for analysis
            df.to_csv(csv_path, index=False, float_format='%.2f')
            output_paths['csv'] = csv_path
            logger.info(f"Saved CSV to {csv_path}")
            
            # Also save formatted version
            csv_formatted_path = os.path.join(out_dir, 'rset_outperformance_formatted.csv')
            df_formatted.to_csv(csv_formatted_path, index=False)
            output_paths['csv_formatted'] = csv_formatted_path
            logger.info(f"Saved formatted CSV to {csv_formatted_path}")
        
        if 'md' in output_formats:
            md_path = os.path.join(out_dir, 'rset_outperformance.md')
            # Use formatted version for markdown
            df_display = df_formatted.copy()
            # Add % sign to percentages
            for col in [c for c in df_display.columns if c.startswith('vs_')]:
                df_display[col] = df_display[col].apply(lambda x: f"{x}%")
            
            # Remove helper columns for cleaner display
            display_cols = [c for c in df_display.columns if c not in ['n_folds', 'n_trees_mean', 'higher_is_better']]
            df_display = df_display[display_cols]
            
            with open(md_path, 'w') as f:
                f.write("# RSET Outperformance Analysis\n\n")
                f.write("Percentage of RSET trees that outperform each baseline model (mean ± std across folds).\n\n")
                f.write(df_display.to_markdown(index=False))
            output_paths['md'] = md_path
            logger.info(f"Saved Markdown to {md_path}")
        
        if 'latex' in output_formats:
            latex_path = os.path.join(out_dir, 'rset_outperformance.tex')
            # Use formatted version for LaTeX
            df_latex = df_formatted.copy()
            
            # Add % sign and escape for LaTeX
            for col in [c for c in df_latex.columns if c.startswith('vs_')]:
                df_latex[col] = df_latex[col].apply(lambda x: f"{x}\\%")
            
            # Remove helper columns for cleaner table
            display_cols = [c for c in df_latex.columns if c not in ['n_folds', 'n_trees_mean', 'higher_is_better']]
            df_latex = df_latex[display_cols]
            
            latex_str = df_latex.to_latex(
                index=False,
                escape=False,
                column_format='l' + 'l' + 'r' * (len(display_cols) - 2),
                caption="Percentage of RSET trees outperforming baseline models (mean $\\pm$ std across folds)",
                label="tab:rset_outperformance"
            )
            
            # Add booktabs
            latex_str = latex_str.replace('\\begin{tabular}', '\\begin{tabular}')
            latex_str = latex_str.replace('\\toprule', '\\toprule')
            latex_str = latex_str.replace('\\midrule', '\\midrule')
            latex_str = latex_str.replace('\\bottomrule', '\\bottomrule')
            
            with open(latex_path, 'w') as f:
                f.write(latex_str)
            output_paths['latex'] = latex_path
            logger.info(f"Saved LaTeX to {latex_path}")
        
        logger.info(f"Completed RSET outperformance analysis: {len(results)} rows")
        return output_paths

    def plot_barplot_treefarms_summary(self) -> Dict[str, str]:
        """
        Create barplots showing only specific RSET special methods for all available metrics.
        Compares different special tree selection strategies within RSET.
        
        Only plots these methods:
        - RSET_opt (optimal_tree)
        - RSET_kan (kantch)
        - RSET_min (min_leaf_tree)
        - RSET_max (max_leaf_tree)
        - RSET_sp (fairness_sp)
        - RSET_eopp (fairness_eopp)
        - RSET_eo (fairness_eo)
        
        Returns:
            Dict mapping metric names to saved figure paths
        """
        # Define allowed metrics to plot
        allowed_metrics = {
            "Test Accuracy",
            "Test Adv Accuracy",
            "Stability Acc Mean",
            "Stability Acc Worst",
            "Statistical Parity",
            "Equal Opportunity",
            "Equalized Odds",
            "MIA Label Supervised",
        }
        
        # Define allowed RSET methods (internal keys)
        allowed_methods = {
            "RSET_optimal_tree",
            "RSET_kantch",
            "RSET_min_leaf_optimal_tree",  # Min leaf variant exists in data
            "RSET_max_leaf_optimal_tree",  # Max leaf variant exists in data
            "RSET_fairness_sp",
            "RSET_fairness_eopp",
            "RSET_fairness_eo",
        }
        
        # Get all unique metrics - discover from data or use allowed_metrics
        if self._tidy is not None and not self._tidy.empty:
            all_metrics = self._tidy["metric"].unique()
        else:
            # Use allowed_metrics as default if no data loaded yet
            all_metrics = list(allowed_metrics)
        
        filtered_metrics = [m for m in all_metrics if m in allowed_metrics]
        logger.info(f"Plotting barplots for {len(filtered_metrics)} metrics (out of {len(all_metrics)} total): {filtered_metrics}")
        
        output_paths = {}
        
        for metric in filtered_metrics:
            # Load metric data efficiently (uses cache or lazy loading)
            dfm = self.get_metric_data(metric, use_cache=True)
            if dfm.empty:
                logger.warning("No data found for metric %s", metric)
                continue

            out_dir = self._ensure_dir(self._path_fig(metric))
            out_path = os.path.join(out_dir, f"barplot_treefarms_summary_{metric}.png")

            datasets = dfm["dataset"].unique()
            
            for dataset in datasets:
                dset = dfm[dfm.dataset == dataset].copy()
                rset_data = dset[dset.model == "RSET"]
                
                if rset_data.empty:
                    logger.warning("No RSET data found for dataset %s", dataset)
                    continue

                fig, ax = plt.subplots(figsize=self.cfg.fig_size_1d, dpi=self.cfg.fig_dpi)
                
                # Get RSET special tree models directly from DataFrame
                # These are already stored as separate model entries (e.g., "RSET_opt", "RSET_min")
                rset_special_models = dset[dset.model.str.startswith("RSET_")].model.unique()
                
                if len(rset_special_models) == 0:
                    raise ValueError(f"No special RSET tree models found for dataset {dataset}, metric {metric}")
                
                logger.info(f"Found {len(rset_special_models)} special trees for {dataset}, {metric}: {list(rset_special_models)}")

                plot_data = []
                for model in sorted(rset_special_models):
                    # Filter: only include allowed methods
                    if model not in allowed_methods:
                        logger.info(f"Skipping {model} - not in allowed methods list")
                        input()
                        continue
                    
                    model_data = dset[dset.model == model]
                    if model_data.empty:
                        raise ValueError(f"No data found for model {model} in dataset {dataset}, metric {metric}")
                    
                    # Compute mean/std across folds
                    fold_values = model_data.groupby("fold")["value"].mean()
                    if fold_values.empty:
                        raise ValueError(f"No fold values found for model {model} in dataset {dataset}, metric {metric}")
                    
                    mu = float(fold_values.mean())
                    sd = float(fold_values.std(ddof=1)) if fold_values.size > 1 else 0.0
                    
                    plot_data.append({
                        'model': model,  # Display name
                        'full_name': model,
                        'mean': mu,
                        'std': sd
                    })
                    logger.info(f"Added {model} to plot_data")

                if not plot_data:
                    logger.warning("No valid special tree data for dataset %s, metric %s", dataset, metric)
                    continue

                # Sort by custom order: Opt, min, max, kan, sp, eopp, eo
                sort_order = {
                    "RSET_opt": 0,
                    "RSET_min": 1,
                    "RSET_max": 2,
                    "RSET_kan": 3,
                    "RSET_sp": 4,
                    "RSET_eopp": 5,
                    "RSET_eo": 6,
                }
                plot_data.sort(key=lambda x: sort_order.get(x['model'], 99))
                
                # Log the RSET methods being plotted
                rset_models = [d['model'] for d in plot_data]
                logger.info(f"Plotting {len(rset_models)} RSET methods for {dataset}, {metric}: {rset_models}")
                
                # Use tab20 color palette for RSET methods comparison
                tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
                
                # Create barplot with larger error bars
                models = [d['model'] for d in plot_data]
                means = [d['mean'] for d in plot_data]
                stds = [d['std'] for d in plot_data]
                # Assign tab20 colors cyclically
                colors = [tab20_colors[i % 20] for i in range(len(plot_data))]
                
                bars = ax.bar(models, means, yerr=stds, capsize=15, alpha=0.8,
                             color=colors, edgecolor='black', linewidth=0.5, 
                             error_kw={'linewidth': 2.0, 'elinewidth': 2.0, 'capthick': 2.0})
                
                # Styling - remove X-axis labels (legend will show model names)
                ax.set_xlabel("")  # No X-axis label
                ax.set_ylabel(metric, fontsize=self.cfg.label_fontsize)
                ax.set_title(f"{dataset}", fontsize=self.cfg.title_fontsize)
                ax.set_xticklabels([])  # Remove X tick labels
                ax.tick_params(axis='x', which='both', length=0)  # Remove X tick marks
                ax.tick_params(axis='y', labelsize=self.cfg.tick_labelsize)
                if self.cfg.grid:
                    ax.grid(True, alpha=0.35, axis='y')
                
                # Prepare legend handles (with and without mean±std)
                handles = []
                handles_clean = []
                for i, d in enumerate(plot_data):
                    color = tab20_colors[i % 20]
                    marker = self.cfg.model_marker_map.get(d['full_name'], 's')
                    
                    # Full label with mean±std
                    label_full = f"{d['model']} ({d['mean']:.3f} ± {d['std']:.3f})"
                    handles.append(self._legend_proxy(label_full, color, kind='fill', alpha=0.8))
                    
                    # Clean label without stats
                    handles_clean.append(self._legend_proxy(d['model'], color, kind='fill', alpha=0.8))
                
                # 1) Save plot without legend for clean embedding
                self._savefig(fig, out_dir, f"{dataset}_barplot_treefarms_summary_{metric}_plot_clean")
                
                # 2) Add legend with mean±std values; save full plot
                if handles:
                    # Resize figure to accommodate legend
                    legend_rows = (len(handles) + 2) // 3
                    fig_with_legend_size = self._get_figsize_for_legend(self.cfg.fig_size_1d, legend_rows, ncol=3)
                    fig.set_size_inches(fig_with_legend_size)
                    
                    leg = fig.legend(
                        handles=handles,
                        loc="lower center",
                        ncol=3,
                        frameon=False,
                        fontsize=self.cfg.legend_fontsize,
                        bbox_to_anchor=(0.5, self.cfg.legend_bbox_anchor_y),
                    )
                    fig.subplots_adjust(bottom=self.cfg.legend_subplots_adjust_bottom)
                    self._savefig(fig, out_dir, f"{dataset}_barplot_treefarms_summary_{metric}_plot_legend")
                    leg.remove()
                    fig.set_size_inches(self.cfg.fig_size_1d)
                
                # 3) Add clean legend without mean±std values
                if handles_clean:
                    legend_rows = (len(handles_clean) + 2) // 3
                    fig_clean_legend_size = self._get_figsize_for_legend(self.cfg.fig_size_1d, legend_rows, ncol=3)
                    fig.set_size_inches(fig_clean_legend_size)
                    
                    leg = fig.legend(
                        handles=handles_clean,
                        loc="lower center",
                        ncol=3,
                        frameon=False,
                        fontsize=self.cfg.legend_fontsize,
                        bbox_to_anchor=(0.5, self.cfg.legend_bbox_anchor_y),
                    )
                    fig.subplots_adjust(bottom=self.cfg.legend_subplots_adjust_bottom)
                    self._savefig(fig, out_dir, f"{dataset}_barplot_treefarms_summary_{metric}_plot_clean_legend")
                    leg.remove()
                    fig.set_size_inches(self.cfg.fig_size_1d)
                
                # 4) Save legend-only asset (full version with mean±std)
                if handles:
                    labels = [h.get_label() for h in handles]
                    legend_rows = (len(handles) + 2) // 3
                    legend_height = max(1.4, legend_rows * 0.6 + 0.4)
                    leg_fig = plt.figure(figsize=(8, legend_height), dpi=self.cfg.fig_dpi)
                    leg_fig.legend(
                        handles=handles,
                        labels=labels,
                        loc="center",
                        ncol=3,
                        frameon=False,
                        fontsize=self.cfg.legend_fontsize,
                    )
                    self._savefig(leg_fig, out_dir, f"{dataset}_barplot_treefarms_summary_{metric}_legend")
                
                # 5) Save clean legend-only asset (without mean±std)
                if handles_clean:
                    labels = [h.get_label() for h in handles_clean]
                    legend_rows = (len(handles_clean) + 2) // 3
                    legend_height = max(1.4, legend_rows * 0.6 + 0.4)
                    leg_fig = plt.figure(figsize=(8, legend_height), dpi=self.cfg.fig_dpi)
                    leg_fig.legend(
                        handles=handles_clean,
                        labels=labels,
                        loc="center",
                        ncol=3,
                        frameon=False,
                        fontsize=self.cfg.legend_fontsize,
                    )
                    self._savefig(leg_fig, out_dir, f"{dataset}_barplot_treefarms_summary_{metric}_clean_legend")
            
            output_paths[metric] = out_path
            logger.info(f"Completed barplot for metric {metric}")

        logger.info(f"Generated {len(output_paths)} barplot figures")
        return output_paths

    def _get_df_pair(self, metric_x: str, metric_y: str) -> pd.DataFrame:
        """
        Get data for a pair of metrics for 2D plotting.
        Supports lazy loading mode and model-specific metric remapping.
        
        Strategy:
        1. Discover all available models
        2. For each model, determine which actual metric names to load
        3. Load only the needed metrics per model
        4. Combine and return
        """
        # Step 1: Discover models by loading a sample of data
        # We need to know which models exist before we can remap metrics
        if self._tidy is not None and not self._tidy.empty:
            all_models = self._tidy["model"].unique()
        else:
            # Load just the requested metrics to discover models
            sample_dfs = []
            for m in [metric_x, metric_y]:
                df_m = self.get_metric_data(m, use_cache=True)
                if not df_m.empty:
                    sample_dfs.append(df_m)
            
            if not sample_dfs:
                logger.warning("No data found for metric pair %s and %s", metric_x, metric_y)
                return pd.DataFrame(columns=["dataset", "model", "fold", "idx", "x", "y"])
            
            sample_df = pd.concat(sample_dfs, ignore_index=True)
            all_models = sample_df["model"].unique()
        
        # Step 2: For each model, determine which metrics to load and build metric_map
        # metric_map: {model: (actual_metric_x, actual_metric_y)}
        metric_map = {}
        metrics_to_load = set()
        
        for model in all_models:
            mx, my = self._remap_metric((metric_x, metric_y), model)
            metric_map[model] = (mx, my)
            metrics_to_load.add(mx)
            metrics_to_load.add(my)
        
        logger.debug(f"Loading metrics for pair ({metric_x}, {metric_y}): {metrics_to_load}")
        
        # Step 3: Load all needed metrics
        dfs = []
        for m in metrics_to_load:
            df_m = self.get_metric_data(m, use_cache=True)
            if not df_m.empty:
                dfs.append(df_m)
        
        if not dfs:
            logger.warning("No data found after loading metrics: %s", metrics_to_load)
            return pd.DataFrame(columns=["dataset", "model", "fold", "idx", "x", "y"])
        
        # Step 4: Combine and pivot per model
        tidy = pd.concat(dfs, ignore_index=True)
        pieces = []
        
        for model in all_models:
            mx, my = metric_map[model]
            
            # Filter to this model and its specific metrics
            sub = tidy[(tidy.model == model) & (tidy.metric.isin([mx, my]))][
                ["dataset", "model", "fold", "metric", "value", "idx"]
            ].copy()
            
            if sub.empty:
                logger.debug("No data for model %s with metrics %s and %s", model, mx, my)
                continue
            
            # Pivot to wide format
            sub.loc[sub.metric == mx, "axis"] = "x"
            sub.loc[sub.metric == my, "axis"] = "y"
            wide = sub.pivot_table(index=["dataset", "model", "fold", "idx"],
                                   columns="axis", values="value").reset_index()
            
            # Skip if missing either metric
            if wide.empty or "x" not in wide.columns or "y" not in wide.columns:
                logger.debug("Skipping model %s: missing one or both metrics (%s, %s)", model, mx, my)
                continue
            
            # Drop rows with NaN values
            wide = wide.dropna(subset=["x", "y"])
            if not wide.empty:
                pieces.append(wide)

        if not pieces:
            logger.warning("No valid data found for metric pair %s and %s", metric_x, metric_y)
            return pd.DataFrame(columns=["dataset", "model", "fold", "idx", "x", "y"])
        
        df = pd.concat(pieces, ignore_index=True)
        return df

    def _robust_limits(self, df_xy: pd.DataFrame, q=(0, 1), pad_frac=0.02):
        """Quantile-based limits with a small pad to avoid cutting contours."""
        x = np.asarray(df_xy["x"], dtype=float)
        y = np.asarray(df_xy["y"], dtype=float)
        x0, x1 = np.quantile(x, q); y0, y1 = np.quantile(y, q)
        rx = max(x1 - x0, np.finfo(float).eps)
        ry = max(y1 - y0, np.finfo(float).eps)
        x0 -= pad_frac * rx; x1 += pad_frac * rx
        y0 -= pad_frac * ry; y1 += pad_frac * ry
        return (x0, x1), (y0, y1)

    def _pareto_front_2d(self, x, y, maximize_x=True, maximize_y=True):
        """
        Returns the indices of Pareto-optimal points for two objectives.
        x, y: 1D arrays of same length
        maximize_x / maximize_y: True if larger is better for that axis
        """
        assert x.shape == y.shape

        # If an objective is to be minimized, flip its sign so we always 'maximize'
        X = x if maximize_x else -x
        Y = y if maximize_y else -y

        # Sort by X descending, break ties by Y descending
        order = np.lexsort((-Y, -X))  # lexsort uses last key first
        Xs, Ys = X[order], Y[order]
        idxs = np.arange(len(X))[order]

        # Sweep to keep points that are not dominated in Y
        pareto_mask = np.zeros(len(Xs), dtype=bool)
        best_Y = -np.inf
        for i, yv in enumerate(Ys):
            if yv >= best_Y:         # keep any point that improves Y
                pareto_mask[i] = True
                best_Y = yv

        pareto_indices = idxs[pareto_mask]
        # Sort the front for plotting (by x ascending for a left-to-right curve)
        sort_for_plot = np.argsort(x[pareto_indices])
        return pareto_indices[sort_for_plot]

    def plot_pareto_front(self, metric_x, metric_y, per_dataset=True):
        self._require_tidy()
        tidy = self._tidy
        datasets = tidy["dataset"].unique()
        models = tidy["model"].unique()
        logger.info("Preparing Pareto front plot for %s vs %s", metric_x, metric_y)
        logger.info("Datasets found: %s", datasets)
        logger.info("Models found: %s", models)
        out_dir = self._ensure_dir(self._path_fig(f"{metric_x}_vs_{metric_y}_pareto"))
        df = self._get_df_pair(metric_x, metric_y)
        for dataset in datasets:
            logger.info("Processing dataset: %s", dataset)
            for fold in range(5):
                fig, ax = plt.subplots(figsize=self.cfg.fig_size, dpi=self.cfg.fig_dpi)
                handles = []
                logger.info(" Processing fold: %s", fold)
                for model in models:
                    logger.info("  Processing model: %s", model)
                 

                    x_val = df[(df.dataset == dataset) & (df.model == model) & (df.fold == str(fold))]["x"].values
                    y_val = df[(df.dataset == dataset) & (df.model == model) & (df.fold == str(fold))]["y"].values
                    if model == "dpf":
                        # Print some x,y val
                        logger.info("   dpf x values: %s", x_val)
                        logger.info("   dpf y values: %s", y_val)
                    if len(x_val) == 0 or len(y_val) == 0:
                        logger.warning("No data for model %s on dataset %s fold %s", model, dataset, fold)
                        continue
                    x_val, y_val = self._front_xy(x_val, y_val)
                    if len(x_val) == 0:
                        continue
                    if model == "dpf":
                        # Print some x,y val
                        logger.info("   dpf pareto x values: %s", x_val)
                        logger.info("   dpf pareto y values: %s", y_val)
                    h, = ax.plot(x_val, y_val, lw=2.4, label=model,
                                 color=self.cfg.model_color_map.get(model))
                    handles.append(h) 
                self._style_axes(ax, metric_x, metric_y, f"Pareto Front - {dataset}", handles)
                fname = f"{dataset}_fold_{fold}"
                self._savefig(fig, out_dir, fname)
                logger.info("Saved Pareto front figure to %s", os.path.join(out_dir, fname + ".png"))

    # ---------------------------
    # Grid Plotting Framework
    # ---------------------------
    
    def plot_grid(self, 
                  grid_spec: List[List[Dict[str, Any]]], 
                  figsize: Optional[Tuple[float, float]] = None,
                  share_x: bool = True,
                  share_y: bool = True,
                  shared_legend: bool = True,
                  legend_ncol: int = 4,
                  legend_loc: str = "lower center",
                  legend_bbox_to_anchor: Tuple[float, float] = (0.5, 0),
                  legend_bottom_margin: Optional[float] = None,
                  title: Optional[str] = None,
                  filename: str = "grid_plot",
                  wspace: float = 0.3,
                  hspace: float = 0.3,
                  allowed_models: Optional[List[str]] = None) -> str:
        """
        Create a flexible grid of plots with shared axes and legends.
        
        This is the main entry point for creating production-quality multi-panel figures.
        Supports mixing different plot types (density, barplot, etc.) in a single figure.
        
        Args:
            grid_spec: 2D list defining the grid layout. Each cell is a dict with:
                - 'type': Plot type ('density_1d', 'density_2d', 'barplot_1d', 'empty', 'custom')
                - 'metric': Metric name to plot (X-axis for 2D plots)
                - 'metric_y': Y-axis metric (required for 'density_2d' type)
                - 'dataset': Dataset name (optional, uses all if not specified)
                - 'datasets': List of datasets for multi-dataset plots (optional)
                - 'kind': Plot kind for 2D density ('kde' or 'scatter', default: 'kde')
                - 'kde_models': Tuple of models to render with KDE for 2D plots (default: ('RSET',))
                - 'title': Subplot title (optional)
                - 'xlabel': X-axis label (optional, auto-generated if not provided)
                - 'ylabel': Y-axis label (optional, auto-generated if not provided)
                - 'show_xlabel': Whether to show x-axis label (default: True on bottom row)
                - 'show_ylabel': Whether to show y-axis label (default: True on left column)
                - 'show_xticklabels': Whether to show x-tick labels (default: True on bottom row)
                - 'show_yticklabels': Whether to show y-tick labels (default: True on left column)
                - 'models': List of specific models to include (optional)
                - 'custom_plot_fn': For type='custom', a callable(ax, reporter, spec) -> None
                
                Example:
                [
                    [{'type': 'density_1d', 'metric': 'Test Accuracy', 'dataset': 'compas'},
                     {'type': 'barplot_1d', 'metric': 'Test Accuracy', 'dataset': 'compas'}],
                    [{'type': 'density_2d', 'metric': 'Test Accuracy', 'metric_y': 'Statistical Parity', 'dataset': 'compas'},
                     {'type': 'barplot_1d', 'metric': 'Statistical Parity', 'dataset': 'compas'}]
                ]
                
            figsize: Figure size (width, height). If None, auto-calculated based on grid size.
            share_x: Whether to share x-axis across columns (default: True)
            share_y: Whether to share y-axis across rows (default: True)
            shared_legend: Whether to use a single shared legend (default: True)
            shared_colorbar: Whether to add a figure-wide colorbar for 2D density plots (default: False)
            legend_ncol: Number of columns in the legend
            legend_loc: Legend location
            legend_bbox_to_anchor: Bbox anchor for legend positioning
            legend_bottom_margin: Bottom margin for legend (fraction of figure height).
                                 If None, auto-calculated: 0.02 + 0.8/figheight for tall figures
            title: Overall figure title (optional)
            filename: Output filename (without extension)
            wspace: Width space between subplots
            hspace: Height space between subplots
            allowed_models: Global filter for models to include (can be overridden per cell)
            
        Returns:
            Path to saved figure
            
        Examples:
            # Example 1: 3 datasets x 2 metrics (density + barplot)
            grid = [
                [{'type': 'density_1d', 'metric': 'Test Accuracy', 'dataset': 'compas'},
                 {'type': 'barplot_1d', 'metric': 'Test Accuracy', 'dataset': 'compas'}],
                [{'type': 'density_1d', 'metric': 'Test Accuracy', 'dataset': 'adult'},
                 {'type': 'barplot_1d', 'metric': 'Test Accuracy', 'dataset': 'adult'}],
                [{'type': 'density_1d', 'metric': 'Test Accuracy', 'dataset': 'german'},
                 {'type': 'barplot_1d', 'metric': 'Test Accuracy', 'dataset': 'german'}]
            ]
            reporter.plot_grid(grid, filename='3datasets_2cols')
            
            # Example 2: 1 row x 3 metrics (same dataset)
            grid = [[
                {'type': 'barplot_1d', 'metric': 'Test Accuracy', 'dataset': 'compas'},
                {'type': 'barplot_1d', 'metric': 'Statistical Parity', 'dataset': 'compas'},
                {'type': 'barplot_1d', 'metric': 'Equal Opportunity', 'dataset': 'compas'}
            ]]
            reporter.plot_grid(grid, filename='3metrics_compas')
            
            # Example 3: Metric x Dataset grid
            metrics = ['Test Accuracy', 'Statistical Parity']
            datasets = ['compas', 'adult', 'german']
            grid = [
                [{'type': 'barplot_1d', 'metric': m, 'dataset': d} 
                 for d in datasets]
                for m in metrics
            ]
            reporter.plot_grid(grid, filename='metrics_x_datasets_grid')
            
            # Example 4: Fairness 2D density plots (Accuracy vs Fairness)
            grid = [
                [{'type': 'density_2d', 'metric': 'Test Accuracy', 'metric_y': 'Statistical Parity', 
                  'dataset': 'compas', 'kind': 'kde', 'title': 'COMPAS'}],
                [{'type': 'density_2d', 'metric': 'Test Accuracy', 'metric_y': 'Equal Opportunity', 
                  'dataset': 'compas', 'kind': 'kde', 'title': 'COMPAS'}]
            ]
            reporter.plot_grid(grid, filename='fairness_2d_density', share_x=False, share_y=False)
        """
        nrows = len(grid_spec)
        ncols = len(grid_spec[0]) if grid_spec else 0
        
        if nrows == 0 or ncols == 0:
            logger.warning("Empty grid specification")
            return None
        
        # Auto-calculate figsize if not provided
        if figsize is None:
            base_width = 6
            base_height = 4
            extra_height = 1.5 if shared_legend else 0.5  # Extra space for legend
            figsize = (base_width * ncols, base_height * nrows + extra_height)
        
        # Create figure and axes grid
        fig, axes = plt.subplots(
            nrows=nrows, 
            ncols=ncols, 
            figsize=figsize,
            dpi=self.cfg.fig_dpi,
            sharex='col' if share_x else False,
            sharey='row' if share_y else False,
            squeeze=False  # Always return 2D array
        )
        
        # Adjust spacing
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        
        # Track all legend handles for shared legend
        all_handles = []
        all_labels = []
        seen_labels = set()
        
        # Track Y-limits per row for shared Y-axis
        row_ylimits = {}  # {row_idx: [(y_min, y_max), ...]}
        
        # Process each cell
        for i, row_spec in enumerate(grid_spec):
            for j, cell_spec in enumerate(row_spec):
                ax = axes[i, j]
                
                if cell_spec.get('type') == 'empty':
                    ax.axis('off')
                    continue
                
                # Determine if this is a boundary subplot
                is_bottom_row = (i == nrows - 1)
                is_left_col = (j == 0)
                
                # Get cell-specific or global model filter
                cell_models = cell_spec.get('models', allowed_models)
                
                # Plot based on type
                plot_type = cell_spec.get('type', 'barplot_1d')
                metric = cell_spec.get('metric')
                dataset = cell_spec.get('dataset')
                
                if not metric:
                    logger.warning(f"No metric specified for cell [{i}, {j}]")
                    ax.axis('off')
                    continue
                
                # Render the plot into this axis
                result = self._render_subplot(
                    ax=ax,
                    plot_type=plot_type,
                    metric=metric,
                    dataset=dataset,
                    cell_spec=cell_spec,
                    allowed_models=cell_models,
                    is_bottom_row=is_bottom_row,
                    is_left_col=is_left_col,
                    share_x=share_x,
                    share_y=share_y
                )
                if result is None:
                    raise ValueError(f"Failed to render subplot at [{i}, {j}]")
                
                # Unpack result - can be (handles,) or (handles, ylim)
                if isinstance(result, tuple) and len(result) == 2:
                    handles, ylim = result
                    # Track Y-limits for this row if share_y is enabled
                    if share_y and ylim is not None:
                        if i not in row_ylimits:
                            row_ylimits[i] = []
                        row_ylimits[i].append(ylim)
                else:
                    handles = result
                
                # Collect unique legend handles
                if handles and shared_legend:
                    for handle, label in handles:
                        if label not in seen_labels:
                            all_handles.append(handle)
                            all_labels.append(label)
                            seen_labels.add(label)
        
        # Apply shared Y-limits per row
        if share_y and row_ylimits:
            for row_idx, ylims in row_ylimits.items():
                if ylims:
                    # Get min/max across all cells in this row
                    all_mins = [y[0] for y in ylims]
                    all_maxs = [y[1] for y in ylims]
                    row_ymin = min(all_mins) - 0.1
                    row_ymin = max(0.0, row_ymin)
                    row_ymax = max(all_maxs) + 0.1
                    
                    # Apply to all axes in this row
                    for col_idx in range(ncols):
                        axes[row_idx, col_idx].set_ylim(row_ymin, row_ymax)
        
        if not all_handles:
            logger.warning("No legend handles collected for shared legend")
            # raise ValueError("No legend handles collected for shared legend")
        # Add shared legend
        if shared_legend and all_handles:
            fig.legend(
                all_handles, 
                all_labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                ncol=legend_ncol,
                fontsize=self.cfg.legend_fontsize,
                frameon=False,  # No border
            )
            # Adjust layout to make room for legend
            # Check if legend is on the right (bbox_to_anchor x > 1) or bottom (y < 0)
            if legend_bbox_to_anchor[0] > 1.0:
                # Legend on the right side - adjust right margin
                fig.subplots_adjust(right=0.85)  # Leave space on the right
            else:
                # Legend on the bottom - adjust bottom margin
                if legend_bottom_margin is None:
                    # Auto-calculate: smaller fraction for taller figures
                    # 0.01 + 0.5/figheight gives ~0.028 for 28-inch, ~0.06 for 10-inch
                    legend_bottom_margin = 0.01 + 0.5 / figsize[1]
                fig.subplots_adjust(bottom=legend_bottom_margin)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=self.cfg.title_fontsize, y=0.98)
        
        # Save figure
        out_dir = self._ensure_dir(os.path.join(self.cfg.out_root, self.cfg.fig_dir))
        out_path = os.path.join(out_dir, f"{filename}.png")
        
        self._savefig(fig, out_dir, f"{filename}.png")
        plt.close(fig)
        
        logger.info(f"Saved grid plot to {out_path}")
        return out_path
    
    def _render_subplot(self, 
                       ax: plt.Axes,
                       plot_type: str,
                       metric: str,
                       dataset: Optional[str],
                       cell_spec: Dict[str, Any],
                       allowed_models: Optional[List[str]],
                       is_bottom_row: bool,
                       is_left_col: bool,
                       share_x: bool,
                       share_y: bool):
        """
        Render a single subplot into the given axis.
        
        Args:
            ax: Matplotlib axis to render into
            plot_type: Type of plot ('density_1d', 'barplot_1d', 'custom')
            metric: Metric name
            dataset: Dataset name (optional)
            cell_spec: Full cell specification dict
            allowed_models: Models to include
            is_bottom_row: Whether this is the bottom row
            is_left_col: Whether this is the left column
            share_x: Whether x-axis is shared
            share_y: Whether y-axis is shared
            
        Returns:
            List of (handle, label) tuples for legend
        """
        handles = []
        
        # Load data - can use aggregated cache for baseline statistics
        dfm = self.get_metric_data(metric, dataset=dataset, use_cache=True)
        if dfm.empty:
            raise ValueError(f"No data found for metric {metric} and dataset {dataset}")
        # Filter to specific dataset if provided
        if dataset:
            dfm = dfm[dfm.dataset == dataset]
        
        # Render based on plot type
        ylim = None  # Will be set by renderers that return it
        if plot_type == 'density_1d':
            handles = self._render_density_1d(ax, dfm, metric, allowed_models)
        elif plot_type == 'density_2d':
            # For 2D density, need both metric_x and metric_y from cell_spec
            metric_y = cell_spec.get('metric_y')
            if not metric_y:
                raise ValueError(f"density_2d plot type requires 'metric_y' in cell_spec")
            kind = cell_spec.get('kind', 'kde')
            kde_models = tuple(cell_spec.get('kde_models', ['RSET']))
            handles = self._render_density_2d(ax, metric, metric_y, dataset, allowed_models, kind=kind, kde_models=kde_models)
        elif plot_type == 'barplot_1d':
            handles, ylim = self._render_barplot_1d(ax, dfm, metric, dataset, allowed_models, share_y=share_y)
        elif plot_type == 'barplot_grouped_shifts':
            handles = self._render_barplot_grouped_shifts(ax, metric, dataset, allowed_models, cell_spec)
        elif plot_type == 'custom':
            custom_fn = cell_spec.get('custom_plot_fn')
            if custom_fn and callable(custom_fn):
                handles = custom_fn(ax, self, cell_spec) or []
            else:
                logger.warning(f"No valid custom_plot_fn provided for custom plot")
        else:
            logger.warning(f"Unknown plot type: {plot_type}")
            return handles, ylim
        
        # Apply subplot styling
        subplot_title = cell_spec.get('title', f"{dataset or 'All'}")
        ax.set_title(subplot_title, fontsize=self.cfg.title_fontsize * 0.8)
        
        # Handle axis labels based on position and sharing
        # Note: Y-axis labels can be shown independently of Y-axis tick sharing
        # Use 'share_ylabel' to control label sharing, separate from tick sharing
        share_ylabel = cell_spec.get('share_ylabel', share_y)  # Default to share_y behavior
        
        show_xlabel = cell_spec.get('show_xlabel', is_bottom_row or not share_x)
        show_ylabel = cell_spec.get('show_ylabel', is_left_col or not share_ylabel)
        show_xticklabels = cell_spec.get('show_xticklabels', is_bottom_row or not share_x)
        show_yticklabels = cell_spec.get('show_yticklabels', True)  # Always show Y ticks by default
        
        if show_xlabel:
            xlabel = cell_spec.get('xlabel', metric)
            ax.set_xlabel(xlabel, fontsize=self.cfg.label_fontsize)
        else:
            ax.set_xlabel('')
        
        if show_ylabel:
            # Check for explicit ylabel first, then auto-generate
            if 'ylabel' in cell_spec:
                ylabel = cell_spec['ylabel']
            else:
                # Auto-generate ylabel based on plot type
                if plot_type == 'density_1d':
                    ylabel = 'Count'
                elif plot_type == 'density_2d':
                    ylabel = cell_spec.get('metric_y', metric)  # Use metric_y for 2D plots
                elif plot_type in ['barplot_1d']:
                    # For barplots, use the metric name as the Y-label
                    ylabel = metric
                else:
                    ylabel = 'Score'
            ax.set_ylabel(ylabel, fontsize=self.cfg.label_fontsize)
        else:
            ax.set_ylabel('')
        
        if not show_xticklabels:
            ax.set_xticklabels([])
        
        if not show_yticklabels:
            ax.set_yticklabels([])
        
        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=self.cfg.tick_labelsize)
        
        # Grid
        if self.cfg.grid:
            ax.grid(True, alpha=0.3)
        
        return handles, ylim
    
    def _render_density_1d(self, 
                          ax: plt.Axes,
                          dfm: pd.DataFrame,
                          metric: str,
                          allowed_models: Optional[List[str]]) -> List[Tuple[Any, str]]:
        """
        Render a 1D density plot into the given axis.
        Returns list of (handle, label) tuples for legend.
        """
        handles = []
        
        # Get unique datasets (should be 1)
        datasets = dfm["dataset"].unique()
        assert len(datasets) == 1, "Density 1D plot requires a single dataset, found: " + ", ".join(datasets)
        
        dataset_name = datasets[0]
        dset = dfm[dfm.dataset == dataset_name].copy()
        
        # Determine data format based on cache usage
        # Aggregated cache has 'mean_value', raw file load has 'value'
        is_aggregated = "mean_value" in dset.columns
        
        # RSET density - load fold-by-fold with subsampling to avoid OOM
        tf_color = self.cfg.model_color_map.get("RSET")
        
        # Determine folds: from data if available, else assume 5-fold CV
        if "fold" in dset.columns:
            fold_range = dset["fold"].unique()
        else:
            # Aggregated cache doesn't have fold column, assume standard 5-fold
            fold_range = range(5)
        
        # Load full RSET distribution across all folds (no subsampling)
        logger.info(f"Loading RSET distribution for {dataset_name}, {metric}")
        all_fold_values = []
        
        for fold in fold_range:
            fold_dist = self._load_rset_distribution(dataset_name, str(fold), metric)
            if fold_dist is not None and len(fold_dist) > 0:
                all_fold_values.append(fold_dist)
            else:
                raise ValueError(f"RSET distribution data missing for {dataset_name}, fold {fold}, metric {metric}")
        
        tf_vals = np.concatenate(all_fold_values)
        logger.info(f"Loaded {len(tf_vals)} RSET trees across {len(all_fold_values)} folds")
        
        # Add jitter for visualization
        tf_vals_plot = tf_vals + np.random.normal(0, 1e-3, size=len(tf_vals))
        del tf_vals
        sns.histplot(
            tf_vals_plot,
            stat="count",
            binwidth=0.005,
            fill=True,
            ax=ax,
            color=tf_color,
            kde=True,
            edgecolor=None,
        )
        del tf_vals_plot
        
        # RSET special trees as vertical lines (already in DataFrame as separate models)
        rset_special_models = dset[dset.model.str.startswith("RSET_")].model.unique()
        if len(rset_special_models) == 0:
            raise ValueError(f"No RSET special trees found for {dataset_name}, {metric}")
        
        for model in sorted(rset_special_models):
            # Check if model is in allowed list
            if allowed_models and model not in allowed_models:
                logger.info(f"Skipping {model} - not in allowed models list")
                continue
            
            model_data = dset[dset.model == model]
            if model_data.empty:
                raise ValueError(f"No data for model {model} on metric {metric}, dataset {dataset_name}")
            
            # Compute mean/std across folds
            fold_values = model_data.groupby("fold")["value"].mean()
            if fold_values.empty:
                raise ValueError(f"No fold data for model {model} on metric {metric}, dataset {dataset_name}")
            
            mu = float(fold_values.mean())
            sd = float(fold_values.std(ddof=1)) if fold_values.size > 1 else 0.0
            
            # Get color - model is already the display name
            color = self.cfg.model_color_map.get(model)
            
            ax.axvline(mu, linestyle="-", linewidth=self.cfg.axvline_lw, color=color, alpha=self.cfg.axvline_alpha, ymax=0.9)
            handle = self._legend_proxy(model, color, kind="fill", alpha=1.0)
            handles.append((handle, model))
        
        # Baseline models
        for model in sorted(dset.model.unique()):
            if model == "RSET":
                continue
            # Skip RSET special trees (they were handled above)
            if model.startswith("RSET_"):
                continue
            
            # Check if model is in allowed list (check both internal name and display name)
            if allowed_models:
                display_name = self.cfg.model_name_map.get(model, model)
                if model not in allowed_models and display_name not in allowed_models:
                    continue
            
            dmod = dset[dset.model == model]
            if dmod.empty:
                continue
            
            # Calculate mean/std based on data format
            if is_aggregated:
                # Aggregated data: already has mean_value, std_value
                # Just take the single row (one per dataset/model)
                if len(dmod) > 0:
                    row = dmod.iloc[0]
                    mu = float(row["mean_value"])
                    sd = float(row["std_value"]) if "std_value" in row else 0.0
                else:
                    continue
            else:
                # Raw data: compute mean/std across folds
                fold_means = dmod.groupby("fold")["value"].mean()
                if fold_means.empty:
                    continue
                mu = float(fold_means.mean())
                sd = float(fold_means.std(ddof=1)) if fold_means.size > 1 else 0.0
            
            display_name = self.cfg.model_name_map.get(model, model)
            color = self.cfg.model_color_map.get(model)
            
            ax.axvline(mu, linestyle="-", linewidth=self.cfg.axvline_lw, color=color, alpha=self.cfg.axvline_alpha, ymax=0.9)
            handle = self._legend_proxy(display_name, color, kind="fill", alpha=1)
            handles.append((handle, display_name))
        
        return handles
    
    def _render_density_2d(self,
                          ax: plt.Axes,
                          metric_x: str,
                          metric_y: str,
                          dataset: Optional[str],
                          allowed_models: Optional[List[str]],
                          kind: str = "kde",
                          kde_models: tuple[str, ...] = ("RSET",)) -> List[Tuple[Any, str]]:
        """
        Render a 2D density plot into the given axis.
        Returns list of (handle, label) tuples for legend.
        
        Args:
            ax: Matplotlib axis to render into
            metric_x: X-axis metric name
            metric_y: Y-axis metric name
            dataset: Dataset name (optional, if None uses all datasets)
            allowed_models: Models to include
            kind: Plot kind ('kde' or 'scatter')
            kde_models: Tuple of model names to render with KDE (default: ('RSET',))
        """
        handles = []
        
        logger.info(f"Rendering 2D density for {metric_x} vs {metric_y} on dataset {dataset}")
        # Load data for both metrics using _get_df_pair which handles model-specific metric remapping
        df = self._get_df_pair(metric_x, metric_y)
        # Filter to specific dataset if provided
        if dataset is not None:
            df = df[df.dataset == dataset]
        
        # Rename columns to match expected format (x, y -> value_x, value_y)
        df = df.rename(columns={"x": "value_x", "y": "value_y"})
        
        logger.info(f"Loaded data frame with shape {df.shape}, models: {df['model'].unique() if not df.empty else 'N/A'}")
        if df.empty:
            raise ValueError(f"No data found for metric pair {metric_x} and {metric_y}")
        
        # Get unique datasets (should be 1 if dataset was specified)
        datasets = df["dataset"].unique()
        if len(datasets) > 1:
            raise ValueError(f"2D density plot requires a single dataset, found: " + ", ".join(datasets))
        dataset_name = datasets[0]

        # Get models to plot
        models_raw = df["model"].unique()
        logger.info(f"Available models for 2D density: {models_raw}")
        logger.info(f"Allowed models filter: {allowed_models}")
        
        # Check if we have any RSET special trees - if so, we want to plot RSET distribution
        has_rset = any(m.startswith("RSET") for m in models_raw)
        
        # Build final model list: baselines + "RSET" (if any RSET trees exist)
        models = []
        if has_rset:
            models.append("RSET")
        for m in models_raw:
            if not m.startswith("RSET"):  # Skip all RSET variants, we handle RSET separately
                models.append(m)
        
        logger.info(f"Models to plot after filtering: {models}")
        
        # First pass: collect all data to compute proper limits
        # This includes RSET distribution data which is not in the dataframe
        all_x_data = []
        all_y_data = []
        
        # Add baseline model data
        all_x_data.append(df["value_x"].values)
        all_y_data.append(df["value_y"].values)
        
        # Load RSET distribution if available
        if has_rset:
            rset_data = df[df.model.str.startswith("RSET")]
            if not rset_data.empty:
                folds = rset_data["fold"].unique()
                for fold in folds:
                    x_dist = self._load_rset_distribution(dataset_name, str(fold), metric_x)
                    y_dist = self._load_rset_distribution(dataset_name, str(fold), metric_y)
                    if x_dist is not None and y_dist is not None and len(x_dist) == len(y_dist):
                        all_x_data.append(x_dist)
                        all_y_data.append(y_dist)
        
        # Compute limits from all data
        all_x = np.concatenate(all_x_data)
        all_y = np.concatenate(all_y_data)
        x0, x1 = np.quantile(all_x, [0.0, 1.0])
        y0, y1 = np.quantile(all_y, [0.0, 1.0])
        
        # Add padding for KDE contours (10% instead of 2%)
        rx = max(x1 - x0, np.finfo(float).eps)
        ry = max(y1 - y0, np.finfo(float).eps)
        pad_frac = 0.10
        x0 -= pad_frac * rx
        x1 += pad_frac * rx
        y0 -= pad_frac * ry
        y1 += pad_frac * ry
        
        logger.info(f"Computed axis limits: x=[{x0:.4f}, {x1:.4f}], y=[{y0:.4f}, {y1:.4f}]")
        
        # Second pass: plot all models with proper limits
        for model in models:
            # Check if model is in allowed list
            if allowed_models:
                if model not in allowed_models:
                    logger.info(f"Skipping model {model}: not in allowed list")
                    continue
            
            # For RSET, we don't need dsub because we load from NPZ cache
            # For other models, get data from DataFrame
            if model == "RSET":
                # Get any RSET special tree row to determine folds
                dsub = df[df.model.str.startswith("RSET")]
                if dsub.empty:
                    raise ValueError(f"No RSET data found for dataset {dataset_name}")
            else:
                dsub = df[df.model == model]
                if dsub.empty:
                    raise ValueError(f"No data for model {model} on dataset {dataset_name}")
                
            
            if model == "RSET":
                # Load full distributions for both metrics across all folds
                folds = dsub["fold"].unique()
                all_x_vals = []
                all_y_vals = []
                
                for fold in folds:
                    x_dist = self._load_rset_distribution(dataset_name, str(fold), metric_x)
                    y_dist = self._load_rset_distribution(dataset_name, str(fold), metric_y)
                    
                    if x_dist is not None and y_dist is not None:
                        # Both distributions should have same length (one value per tree)
                        if len(x_dist) == len(y_dist):
                            all_x_vals.append(x_dist)
                            all_y_vals.append(y_dist)
                        else:
                            logger.warning(f"Mismatched distribution lengths for fold {fold}: "
                                            f"{len(x_dist)} vs {len(y_dist)}")
                
                if all_x_vals and all_y_vals:
                    # Concatenate across folds
                    x_full = np.concatenate(all_x_vals)
                    y_full = np.concatenate(all_y_vals)
                    logger.info(f"Loaded {len(x_full)} RSET trees for 2D density")
                    # Downsample to 150k points
                    if len(x_full) > 150000:
                        indices = np.random.choice(len(x_full), size=150000, replace=False)
                        x_full = x_full[indices]
                        y_full = y_full[indices]
                        logger.info(f"Downsampled RSET distribution to 150,000 points for plotting")
                    # Plot KDE with full distribution
                    sns.kdeplot(
                        x=x_full, y=y_full,
                        levels=15, fill=True, thresh=0.01, bw_adjust=1.2,
                        ax=ax, gridsize=96, label=model, cbar=False,  # Disable cbar in grid
                        cmap="winter",
                        cut=0, clip=((x0, x1), (y0, y1)), alpha=0.6
                    )
                    del x_full, y_full # Free memory
                else:
                    raise ValueError(f"No RSET distribution data found for dataset {dataset_name}")

                # Create legend proxy
                # proxy = self._legend_proxy(
                #     label=model,
                #     color=self.cfg.model_color_map.get(model),
                #     kind="marker",
                #     marker=self.cfg.model_marker_map.get(model, 'o'),
                #     markersize=self.cfg.legend_marker_size,
                #     alpha=1.0,
                # )
                # handles.append((proxy, display_name))
            else:
                num_data = len(dsub)
                if num_data > 500:
                    raise ValueError(f"Too many data points ({num_data}) for scatter plot of model {model} - limit to 50 or use KDE")
                ax.scatter(
                    dsub["value_x"], dsub["value_y"],
                    s=self.cfg.marker_size,
                    alpha=self.cfg.marker_alpha,
                    label=model,
                    color=self.cfg.model_color_map.get(model),
                    marker=self.cfg.model_marker_map.get(model, 'o'),
                    rasterized=True,
                )
                proxy = self._legend_proxy(
                    label=model,
                    color=self.cfg.model_color_map.get(model),
                    kind="marker",
                    marker=self.cfg.model_marker_map.get(model, 'o'),
                    markersize=self.cfg.legend_marker_size,
                    alpha=1.0,
                )
                handles.append((proxy, model))
        
        # Lock axes after plotting
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        
        # Coarser Y ticks + bigger tick labels
        ax.yaxis.set_major_locator(MaxNLocator(nbins=self.cfg.yaxis_nbins))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        return handles
    
    def _render_barplot_1d(self,
                          ax: plt.Axes,
                          dfm: pd.DataFrame,
                          metric: str,
                          dataset: Optional[str],
                          allowed_models: Optional[List[str]],
                          share_y: bool = False) -> Tuple[List[Tuple[Any, str]], Tuple[float, float]]:
        """
        Render a 1D barplot into the given axis.
        
        Args:
            share_y: If True, Y-axis will be shared across row (limits returned but not set)
        
        Returns:
            Tuple of:
                - List of (handle, label) tuples for legend
                - Tuple of (y_min, y_max) desired Y-axis limits
        """
        handles = []
        
        # Get unique datasets (should be 1 if dataset was specified)
        datasets = dfm["dataset"].unique()
        if len(datasets) > 1:
            logger.warning(f"Multiple datasets found for barplot: {datasets}, using first")
        
        dset = dfm[dfm.dataset == datasets[0]].copy()
        
        # Collect data for all models
        plot_data = []
        
        # Baseline models (exclude RSET and RSET special trees)
        baselines = dset[~dset.model.str.startswith("RSET")]
        for model in sorted(baselines.model.unique()):
            if allowed_models and model not in allowed_models:
                continue
            
            model_data = baselines[baselines.model == model]
            if model_data.empty:
                continue
            
            fold_means = model_data.groupby("fold")["value"].mean()
            if fold_means.empty:
                continue
            
            mu = float(fold_means.mean())
            if "Stability Acc" in metric:
                # For stability metrics, use sd provided
                std_metric_name = metric.replace("Mean", "Std")
                df_std = self.get_metric_data(std_metric_name, dataset=dataset)
                std_data = df_std[(df_std.dataset == datasets[0]) & (df_std.model == model)]
                if std_data.empty:
                    raise ValueError(f"No std data found for model {model} on metric {std_metric_name}, dataset {dataset}")
                fold_stds = std_data.groupby("fold")["value"].mean()
                sd = float(fold_stds.mean()) 
            else:
                sd = float(fold_means.std(ddof=1)) if fold_means.size > 1 else 0.0
            
            plot_data.append({
                'model': model,
                'display_name': self.cfg.model_name_map.get(model, model),
                'mean': mu,
                'std': sd,
            })
        
        # RSET special trees (already in DataFrame as separate models)
        rset_special_models = dset[dset.model.str.startswith("RSET_")].model.unique()
        for model in sorted(rset_special_models):
            if allowed_models and model not in allowed_models:
                logger.info(f"Skipping {model} - not in allowed models list")
                continue
            
            model_data = dset[dset.model == model]
            if model_data.empty:
                raise ValueError(f"No data for RSET special tree model {model}")
                continue
            
            fold_values = model_data.groupby("fold")["value"].mean()
            if fold_values.empty:
                continue
            
            mu = float(fold_values.mean())
            if "Stability Acc" in metric:
                # For stability metrics, use sd provided
                std_metric_name = metric.replace("Mean", "Std")
                df_std = self.get_metric_data(std_metric_name, dataset=dataset)
                std_data = df_std[(df_std.dataset == datasets[0]) & (df_std.model == model)]
                if std_data.empty:
                    raise ValueError(f"No std data found for model {model} on metric {std_metric_name}, dataset {dataset}")
                fold_stds = std_data.groupby("fold")["value"].mean()
                sd = float(fold_stds.mean()) 
            else:
                sd = float(fold_values.std(ddof=1)) if fold_values.size > 1 else 0.0
            
            plot_data.append({
                'model': model,  # Already display name
                'display_name': model,
                'mean': mu,
                'std': sd,
            })
        
        if not plot_data:
            raise ValueError(f"No models found to plot for barplot on metric {metric}, dataset {dataset}")
        # Sort by model order from config (not by mean values)
        def get_model_order_index(model_name):
            """Get sort index based on cfg.model_order, with fallback to alphabetical"""
            try:
                return self.cfg.model_order.index(model_name)
            except ValueError:
                # Not in model_order list, put at end alphabetically
                return len(self.cfg.model_order) + ord(model_name[0])
        
        plot_data.sort(key=lambda x: get_model_order_index(x['model']))
        
        # Create barplot
        x_pos = np.arange(len(plot_data))
        means = [d['mean'] for d in plot_data]
        stds = [d['std'] for d in plot_data]
        colors = [self.cfg.model_color_map.get(d['model'], '#808080') for d in plot_data]
        display_names = [d['display_name'] for d in plot_data]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color=colors, alpha=1, edgecolor='black', linewidth=0.8)
        
        # Set smart y-axis limits to show differences more clearly
        # Use min - 1*std to max + 1*std range (with some padding)
        min_val = min(means)
        max_val = max(means)
        mean_std = np.mean(stds)
        
        # Add padding: use the range of error bars for context
        y_range = max_val - min_val
        if y_range < 0.01:  # If values are very close, use std for range
            y_range = mean_std * 2
        
        # Set limits: don't go below 0 for metrics that are percentages/accuracy
        y_min = max(0, min_val - mean_std - y_range * 0.1)
        y_max = min(1.0, max_val + mean_std + y_range * 0.1)  # Cap at 1.0 for percentage metrics
        
        # Set ylim now if Y-axis is NOT shared (when shared, plot_grid will set it)
        if not share_y:
            ax.set_ylim(y_min, y_max)
        
        # Format Y-axis ticks to 2 decimal points if range is 0-1
        if y_min >= 0 and y_max <= 1:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Remove x-tick labels (legend identifies models)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([], rotation=0)
        ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks
        
        # Create legend handles
        for d in plot_data:
            color = self.cfg.model_color_map.get(d['model'], '#808080')
            handle = Patch(facecolor=color, edgecolor='black', label=d['display_name'])
            handles.append((handle, d['display_name']))
        
        # Return handles and Y-limits (plot_grid will use limits if share_y=True)
        return handles, (y_min, y_max)
    
    def _render_barplot_grouped_shifts(self,
                                       ax: plt.Axes,
                                       metric: str,
                                       dataset: Optional[str],
                                       allowed_models: Optional[List[str]],
                                       cell_spec: Dict[str, Any]) -> List[Tuple[Any, str]]:
        """
        Render a grouped barplot showing stability across different shift types.
        Used within grid plotting framework.
        
        Returns list of (handle, label) tuples for legend.
        """
        handles = []
        
        if not dataset:
            raise ValueError("Dataset must be specified for grouped shifts barplot")
        
        # Determine metric type from the base metric name
        if 'Mean' in metric:
            metric_type = 'mean'
            shift_metrics = {
                'Std': 'Stability Acc Mean',
                '+0.1': 'Stability Acc Mean +0.1',
                '-0.1': 'Stability Acc Mean -0.1',
                '-0.2': 'Stability Acc Mean -0.2',
                'Rsmp': 'Stability Acc Mean Resample ± 0.05',
            }
        elif 'Worst' in metric:
            metric_type = 'worst'
            shift_metrics = {
                'Std': 'Stability Acc Worst',
                '+0.1': 'Stability Acc Worst +0.1',
                '-0.1': 'Stability Acc Worst -0.1',
                '-0.2': 'Stability Acc Worst -0.2',
                'Rsmp': 'Stability Acc Worst Resample ± 0.05',
            }
        else:
            raise ValueError(f"Invalid metric for grouped shifts: {metric}")
        
        shift_names = list(shift_metrics.keys())
        
        # Load and aggregate data for all shifts
        shift_data = {}  # {shift_name: {model: (mean, std)}}
        all_models_set = set()
        
        for shift_name, shift_metric in shift_metrics.items():
            shift_data[shift_name] = {}
            dfm = self.get_metric_data(shift_metric, dataset=dataset, use_cache=True)
            
            if dfm.empty:
                raise ValueError(f"No data found for metric {shift_metric}, dataset {dataset}")
            
            models = dfm['model'].unique()
            if len(models) == 0:
                raise ValueError(f"No models found for metric {shift_metric}, dataset {dataset}")
            
            for model in sorted(models):
                if allowed_models and model not in allowed_models:
                    logger.info(f"Skipping model {model} for shift {shift_name}: not in allowed list")
                    continue

                model_data = dfm[dfm.model == model]
                if model_data.empty:
                    raise ValueError(f"No data for model {model} on metric {shift_metric}, dataset {dataset}")
                
                fold_values = model_data.groupby("fold")["value"].mean()
                if fold_values.empty:
                    raise ValueError(f"No fold data for model {model} on metric {shift_metric}, dataset {dataset}")
                    
                mu = float(fold_values.mean())
                if "Mean" in shift_metric:
                    std_metric_name = shift_metric.replace("Mean", "Std")
                    df_std = self.get_metric_data(std_metric_name, dataset=dataset, use_cache=True)
                    std_data = df_std[df_std.model == model]
                    if std_data.empty:
                        raise ValueError(f"No std data for model {model} on metric {std_metric_name}, dataset {dataset}")
                    fold_stds = std_data.groupby("fold")["value"].mean()
                    sd = float(fold_stds.mean())
                else:
                    sd = float(fold_values.std(ddof=1))
                color = self.cfg.model_color_map.get(model)
                shift_data[shift_name][model] = (mu, sd)
                all_models_set.add(model)
        
        if not shift_data:
            raise ValueError(f"No data available for grouped shifts plot for dataset {dataset}")
        
        
        # Sort models for consistent ordering
        all_models = sorted(all_models_set, key=lambda m: self.cfg.model_order.index(m) 
                          if m in self.cfg.model_order else 999)
        
        # Create grouped barplot
        x = np.arange(len(shift_names))  # Shift positions
        n_models = len(all_models)
        bar_width = 0.8 / n_models
        
        for i, model in enumerate(all_models):
            means = []
            stds = []
            
            for shift_name in shift_names:
                if shift_name in shift_data and model in shift_data[shift_name]:
                    mean_val, std_val = shift_data[shift_name][model]
                    means.append(mean_val)
                    stds.append(std_val)
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - n_models/2 + 0.5) * bar_width
            display_name = self.cfg.model_name_map.get(model, model)
            color = self.cfg.model_color_map.get(model, '#808080')
            if model not in self.cfg.model_color_map:
                raise ValueError(f"No color defined for model {model}")
            
            bars = ax.bar(x + offset, means, bar_width, yerr=stds, 
                         label=display_name, color=color, alpha=0.8,
                         capsize=3, linewidth=0.5)
            handle = Patch(facecolor=color, edgecolor='black', label=display_name)
            handles.append((handle, display_name))
        
        # Set x-tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(shift_names, fontsize=self.cfg.tick_labelsize * 0.8)
        
        # Smart y-axis limits (same as regular barplot)
        all_means = [m for shift in shift_data.values() for m, _ in shift.values()]
        all_stds = [s for shift in shift_data.values() for _, s in shift.values()]
        
        if all_means:
            min_val = min(all_means)
            max_val = max(all_means)
            mean_std = np.mean(all_stds)
            
            y_range = max_val - min_val
            if y_range < 0.01:
                y_range = mean_std * 2
            
            y_min = max(0, min_val - mean_std - y_range * 0.1)
            y_max = min(1.0, max_val + mean_std + y_range * 0.1)

            ax.set_ylim(y_min, y_max)

        return handles
    
    def plot_stability_grouped_shifts(self, 
                                       metric_type: str = 'mean',
                                       allowed_models: Optional[List[str]] = None) -> str:
        """
        Plot grouped barplots for stability across different shift types.
        
        Creates one figure per dataset with 5 shift types on X-axis:
        - Standard (no shift)
        - Plus 0.1
        - Minus 0.1  
        - Minus 0.2
        - Resample 0.05
        
        Each shift type shows all models grouped together.
        
        Args:
            metric_type: 'mean' or 'worst' to select Stability Acc Mean vs Worst
            allowed_models: Optional list of models to include
            
        Returns:
            Path to saved figures directory
        """
        # Define the shift types and their corresponding metrics
        if metric_type == 'mean':
            shift_metrics = {
                'Standard': 'Stability Acc Mean',
                '+0.1': 'Stability Acc Mean +0.1',
                '-0.1': 'Stability Acc Mean -0.1',
                '-0.2': 'Stability Acc Mean -0.2',
                'Resample': 'Stability Acc Mean Resample',
            }
        elif metric_type == 'worst':
            shift_metrics = {
                'Standard': 'Stability Acc Worst',
                '+0.1': 'Stability Acc Worst +0.1',
                '-0.1': 'Stability Acc Worst -0.1',
                '-0.2': 'Stability Acc Worst -0.2',
                'Resample': 'Stability Acc Worst Resample',
            }
        else:
            raise ValueError(f"metric_type must be 'mean' or 'worst', got {metric_type}")
        
        shift_names = list(shift_metrics.keys())
        
        # Load data for all shift metrics
        all_data = {}
        for shift_name, metric in shift_metrics.items():
            dfm = self.get_metric_data(metric, use_cache=True)
            if not dfm.empty:
                all_data[shift_name] = dfm
        
        if not all_data:
            logger.warning(f"No data found for stability {metric_type} metrics")
            return None
        
        # Get all unique datasets
        datasets = set()
        for dfm in all_data.values():
            datasets.update(dfm['dataset'].unique())
        datasets = sorted(datasets)
        
        out_dir = self._ensure_dir(self._path_fig(f'stability_{metric_type}_grouped_shifts'))
        
        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(16, 6), dpi=self.cfg.fig_dpi)
            
            # Collect data for this dataset across all shifts
            shift_data = {}  # {shift_name: {model: (mean, std)}}
            all_models_set = set()
            
            for shift_name in shift_names:
                if shift_name not in all_data:
                    continue
                    
                dfm = all_data[shift_name]
                dset = dfm[dfm.dataset == dataset].copy()
                
                if dset.empty:
                    continue
                
                shift_data[shift_name] = {}
                
                # Aggregate data for this shift
                agg = self._aggregate_model_data(dset)
                
                for _, row in agg.iterrows():
                    model = row['model']
                    
                    # Filter models if specified
                    if allowed_models:
                        display_name = self.cfg.model_name_map.get(model, model)
                        if model not in allowed_models and display_name not in allowed_models:
                            continue
                    
                    all_models_set.add(model)
                    shift_data[shift_name][model] = (row['mean_value'], row['std_value'])
            
            if not shift_data or not all_models_set:
                logger.warning(f"No data for dataset {dataset}")
                plt.close(fig)
                continue
            
            # Sort models for consistent ordering
            all_models = sorted(all_models_set, key=lambda m: self.cfg.model_order.index(m) 
                              if m in self.cfg.model_order else 999)
            
            # Create grouped barplot
            x = np.arange(len(shift_names))  # Shift positions
            n_models = len(all_models)
            bar_width = 0.8 / n_models
            
            for i, model in enumerate(all_models):
                means = []
                stds = []
                
                for shift_name in shift_names:
                    if shift_name in shift_data and model in shift_data[shift_name]:
                        mean_val, std_val = shift_data[shift_name][model]
                        means.append(mean_val)
                        stds.append(std_val)
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (i - n_models/2 + 0.5) * bar_width
                display_name = self.cfg.model_name_map.get(model, model)
                color = self.cfg.model_color_map.get(model, '#808080')
                
                ax.bar(x + offset, means, bar_width, yerr=stds, 
                      label=display_name, color=color, alpha=0.8,
                      capsize=3, edgecolor='black', linewidth=0.5)
            
            # Styling
            ax.set_xlabel('Shift Type', fontsize=self.cfg.label_fontsize)
            ax.set_ylabel(f'Stability Acc ({metric_type.capitalize()})', fontsize=self.cfg.label_fontsize)
            ax.set_title(f'{dataset}', fontsize=self.cfg.title_fontsize)
            ax.set_xticks(x)
            ax.set_xticklabels(shift_names, fontsize=self.cfg.tick_labelsize)
            ax.tick_params(axis='y', labelsize=self.cfg.tick_labelsize)
            if self.cfg.grid:
                ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=self.cfg.legend_fontsize * 0.7, 
                     ncol=min(4, n_models), loc='best', frameon=False)
            
            # Save figure
            self._savefig(fig, out_dir, f'{dataset}_stability_{metric_type}_grouped_shifts.png')
        
        return out_dir

    # ---------------------------
    # Styling / Utils
    # ---------------------------
    def _savefig(self, fig, out_dir: str, filename: str) -> None:
        self._ensure_dir(out_dir)
        # Add .png extension if not already present
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
        path = os.path.join(out_dir, filename)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=self.cfg.fig_dpi)
        logger.info("Saved figure to %s", path)
        plt.close(fig)

    def _remap_metric(self, metrics, model):
        post_dpf_flag = model.startswith("post")
        if not post_dpf_flag:
            return metrics
        
        metric_x, metric_y = metrics
        
        # Special handling for DPF only: it has only one *_test_accuracy metric
        # (whichever fairness metric it was optimized for), but we want to use it for all fairness plots
        # Post-processing models have matching metric names (sp_test_accuracy for SP, etc.) so use normal mapping
        if model.lower() == "dpf" and metric_y == "Test Accuracy":
            # Try to find which *_test_accuracy metric this DPF model has
            
            if not hasattr(self, '_tidy') or self._tidy is None:
                # Fallback to mapping if data not loaded yet
                x, y = self.cfg.metric_metric_map.get(metrics, metrics)
                return x, y
            
            model_metrics = self._tidy[self._tidy.model == model]['metric'].unique()
            
            # Look for any *_test_accuracy metric
            for m in model_metrics:
                if m.endswith('_test_accuracy'):
                    return metric_x, m
            
            # If no *_test_accuracy found, try the mapping
            x, y = self.cfg.metric_metric_map.get(metrics, metrics)
            return x, y
        
        # For post-processing models and other cases, use the mapping as-is
        x, y = self.cfg.metric_metric_map.get(metrics, metrics)
        return x, y

    def _legend_proxy(self, label: str, color: str | None, kind: str = "marker",
                      marker: str = 'o', markersize: float | None = None, alpha: float = 1.0):
        """Return a lightweight artist to use as a legend handle."""
        if color is None:
            color = "C0"
        if markersize is None:
            markersize = self.cfg.legend_marker_size
        if kind == "fill":
            return Patch(facecolor=color, edgecolor=color, alpha=alpha, label=label)
        if kind == "line":
            return Line2D([0], [0], color=color, lw=2.0, alpha=alpha, label=label)
        # default: marker proxy (scatter)
        return Line2D([0], [0], marker=marker, linestyle='None',
                      markersize=markersize, markerfacecolor=color,
                      markeredgecolor=color, alpha=alpha, label=label)

    def _get_figsize_for_legend(self, base_figsize: Tuple[float, float], 
                               legend_rows: int = 1, ncol: int = 3) -> Tuple[float, float]:
        """
        Calculate appropriate figure size when adding bottom legends.
        
        Args:
            base_figsize: Base figure size (width, height)
            legend_rows: Number of legend rows (estimated from handles and ncol)
            ncol: Number of columns in legend
            
        Returns:
            Adjusted figure size tuple (width, height)
        """
        width, height = base_figsize
        # Estimate additional height needed for legend
        # Each legend row needs roughly 0.5-0.8 inches depending on font size
        legend_height_per_row = 0.6 + (self.cfg.legend_fontsize / 60.0)  # More generous conversion
        additional_height = legend_rows * legend_height_per_row + 1.0  # More generous extra margin
        return (width, height + additional_height)

    def _style_axes(self, ax, xlabel, ylabel, title, handles, legend_loc="best"):
        if self.cfg.grid:
            ax.grid(True, alpha=0.35)
        ax.set_xlabel(xlabel, fontsize=self.cfg.label_fontsize); ax.set_ylabel(ylabel, fontsize=self.cfg.label_fontsize)
        ax.set_title(title, fontsize=self.cfg.title_fontsize)
        if handles:
            ax.legend(handles=handles, loc=legend_loc, frameon=False,
                      fontsize=self.cfg.legend_fontsize)

    def _front_xy(self, x: np.ndarray, y: np.ndarray,
                  maximize_x=True, maximize_y=True) -> tuple[np.ndarray,np.ndarray]:
        mask = self._pareto_front_2d(x, y, maximize_x, maximize_y)
        if mask.size == 0:
            logger.warning("No Pareto front points found.")
            return np.array([]), np.array([])
        px, py = x[mask], y[mask]
        order = np.argsort(px, kind="mergesort")
        return px[order], py[order]

    # ---------------------------
    # Internals
    # ---------------------------
    def _extract_and_cache_special_trees_from_model(self, model_path: str, result: Dict[str, Any], 
                                                     dataset: str, fold: str) -> None:
        """
        Extract and cache special tree information from a treefarms model.pkl file.
        
        Args:
            model_path: Path to the model.pkl file
            result: The result dictionary (for getting eval_indices)
            dataset: Dataset name
            fold: Fold identifier
        """
        cache_key = (dataset, fold)
        
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            special_tree_map = getattr(model, 'special_tree', {})
            selected_model_map = getattr(model, 'selected_model', {})
            
            # Get eval_indices from result
            eval_indices = result.get("eval_indices", [])
            if isinstance(eval_indices, np.ndarray):
                eval_indices = eval_indices.tolist()
            
            # Combine both mappings with appropriate prefixes
            all_special = {}
            for name, idx in special_tree_map.items():
                tree_name = f"RSET_{name}"
                # Find position in eval_indices
                if idx in eval_indices:
                    position = eval_indices.index(idx)
                    all_special[tree_name] = position
            logger.info(f"{model_path} special trees: {all_special}")
            for name, idx_or_tuple in selected_model_map.items():
                # Handle case where selected_model stores tuples (idx, score)
                idx = idx_or_tuple[0] if isinstance(idx_or_tuple, (tuple, list)) else idx_or_tuple
                tree_name = f"RSET_{name}"
                # Find position in eval_indices
                if idx in eval_indices:
                    position = eval_indices.index(idx)
                    all_special[tree_name] = position
                    logger.debug(f"Found selected model {tree_name} at index {idx}, position {position}")
            
            if all_special:
                self._special_tree_info[cache_key] = all_special
                logger.info(f"Cached {len(all_special)} special trees for {dataset} fold {fold}: {list(all_special.keys())}")
            else:
                logger.warning(f"No special trees found in model for {dataset} fold {fold}")
                
        except Exception as e:
            logger.warning(f"Error loading model from {model_path}: {e}")



    
    def _read_json_or_pkl(self, path: str) -> Dict[str, Any]:
        """Read JSON results. If a legacy .pkl is passed, attempt a safe load and convert to dict."""
        if path.endswith('.json'):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if path.endswith('.pkl'):
            import pickle
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if hasattr(obj, 'to_dict'):
                try:
                    return obj.to_dict()
                except Exception:
                    pass
            if isinstance(obj, dict):
                return obj
            return {"_pkl_payload": obj}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_rset_data(self, result: Dict[str, Any], root: str, dataset: str, 
                          model: str, fold: str) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
        """
        Extract RSET special trees and full distributions from result dict.
        
        Returns:
            - special_tree_rows: One row per special tree (for DataFrame)
            - distributions: Full tree distributions per metric (for NPZ cache)
        """
        special_tree_rows = []
        distributions = {}
        special_tree_indices = result.pop("special_tree_indices")
        eval_indices = result.pop("eval_indices")
        if isinstance(eval_indices, np.ndarray):
            eval_indices = eval_indices.tolist()
                
        for raw_key, raw_val in result.items():
            if raw_key not in self.cfg.metric_name_map:
                logger.warning(f"  SKIP: Unmapped metric key '{raw_key}' not in metric_name_map")
                continue
            
            metric = self.cfg.metric_name_map[raw_key]
            
            if isinstance(raw_val, (list, tuple, np.ndarray)):
                values = np.asarray(raw_val, dtype=float)
            else:
                values = np.array([float(raw_val)])

            # Invert
            if metric in self.cfg.invert_metrics:
                values = 1.0 - values
            
            
            distributions[metric] = values
        
        rset_train_time = distributions.pop("Train Time")
        if special_tree_indices:
            for tree_name, raw_idx in special_tree_indices.items():
                tree_disp_name = f"RSET_{tree_name}" if not tree_name.startswith("RSET_") else tree_name
                if tree_disp_name not in self.cfg.model_name_map:
                    logger.warning(f"  SKIP: special tree '{tree_disp_name}' not in model_name_map")
                    continue

                if raw_idx not in eval_indices:
                    raise ValueError(f"  Raw index {raw_idx} for tree '{tree_name}' not found in eval_indices")
                
                array_idx = eval_indices.index(raw_idx)
                assert len(eval_indices) == len(next(iter(distributions.values()))), \
                    f"Length of eval_indices {len(eval_indices)} does not match distribution size {len(next(iter(distributions.values())))}"
                if array_idx >= len(next(iter(distributions.values()))):
                    raise ValueError(f"  Array index {array_idx} out of bounds for tree '{tree_name}'")
                display_name = self.cfg.model_name_map[tree_disp_name]

                for metric in distributions.keys():
                    m_value = float(distributions[metric][array_idx])

                    special_tree_rows.append({
                        "root": root,
                        "dataset": dataset,
                        "model": display_name,
                        "fold": fold,
                        "metric": metric,
                        "value": m_value,
                        "idx": array_idx,
                    })

        # Log summary of extraction
        unique_models = set(row['model'] for row in special_tree_rows)
        logger.info(f"Extracted {len(special_tree_rows)} rows for {dataset} fold {fold}: models={sorted(unique_models)}")
        special_tree_rows.append({
            "root": root,
            "dataset": dataset,
            "model": "RSET_Train_Time",
            "fold": fold,
            "metric": "Train Time",
            "value": float(rset_train_time[0]),
            "idx": -1,
        })
        
        return special_tree_rows, distributions
    
    def _load_rset_distribution(self, dataset: str, fold: str, metric: str) -> Optional[np.ndarray]:
        """
        Lazy-load full RSET tree distribution for a specific (dataset, fold, metric).
        Used by density plotting methods.
        
        Returns:
            Array of tree values, or None if not found
        """
        cache_key = f"{dataset}_fold_{fold}"
        metric_key = f"RSET_trees_{metric}"
        
        cached = self.cache.load_dataset_metric(cache_key, metric_key)
        if cached and "values" in cached:
            logger.debug(f"Loaded RSET distribution from cache: {dataset} fold {fold}, {metric}")
            return cached["values"]
        
        # If this is a merged reporter, try loading from source reporters first
        if self._source_reporters:
            for source_reporter in self._source_reporters:
                result = source_reporter._load_rset_distribution(dataset, fold, metric)
                if result is not None:
                    logger.debug(f"Loaded RSET distribution from source reporter: {dataset} fold {fold}, {metric}")
                    return result
            logger.warning(f"RSET distribution not found in source reporters: {dataset} fold {fold}, {metric}")
        
        # If not cached and not found in source reporters, try loading from local files
        if self._files_map is None:
            logger.warning(f"Cannot load RSET distribution: no files_map available")
            return None
        
        # Find and load the specific file
        for root_key, maybe_datasets in self._iter_roots(self._files_map):
            if not isinstance(maybe_datasets, dict):
                continue
            if dataset not in maybe_datasets:
                continue
            
            models = maybe_datasets[dataset]
            if not isinstance(models, dict):
                continue
            
            # Look for treefarms model
            for model_key in ["treefarms", "RSET"]:
                if model_key not in models:
                    continue
                
                payload = models[model_key]
                if isinstance(payload, dict) and fold in payload:
                    bundle = payload[fold]
                    if isinstance(bundle, dict) and "result" in bundle:
                        result = self._read_json_or_pkl(bundle["result"])
                        
                        # Extract the metric values
                        for raw_key, raw_val in result.items():
                            if raw_key in ("special_tree_indices", "eval_indices"):
                                continue
                            mapped_metric = self.cfg.metric_name_map.get(raw_key, raw_key)
                            if mapped_metric == metric:
                                try:
                                    values = np.asarray(raw_val, dtype=float)
                                    if mapped_metric in self.cfg.invert_metrics:
                                        values = 1.0 - values
                                    # Cache for future use
                                    self.cache.cache_dataset_metric(cache_key, metric_key, values)
                                    logger.info(f"Loaded and cached RSET distribution: {dataset} fold {fold}, {metric} ({len(values)} trees)")
                                    return values
                                except Exception as e:
                                    logger.warning(f"Error loading RSET distribution: {e}")
                                    return None
        
        logger.warning(f"RSET distribution not found: {dataset} fold {fold}, {metric}")
        return None

    def _rows_from_result(self, result: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for raw_key, raw_val in result.items():
            # Skip metadata keys that should not be treated as metrics
            if raw_key not in self.cfg.metric_name_map:
                logger.warning(f"  SKIP: Unmapped metric key '{raw_key}' not in metric_name_map")
                continue

            metric = self.cfg.metric_name_map[raw_key]

            # Normalize to numeric array
            if isinstance(raw_val, (list, tuple, np.ndarray)):
                values = np.asarray(raw_val, dtype=float)
            else:
                values = np.array([float(raw_val)])

            # Optional invert (so higher is better)
            if metric in self.cfg.invert_metrics:
                values = 1.0 - values

            for i, v in enumerate(values):
                rows.append({
                    **meta,
                    "metric": metric,
                    "value": float(v),
                    "idx": i,
                })
        return rows

    def _require_tidy(self) -> None:
        """Check if data is available (either loaded or via lazy loading)."""
        if (self._tidy is None or self._tidy.empty) and self._files_map is None:
            raise RuntimeError("No data loaded. Call .load(files_map) first.")

    # Paths
    def _path_fig(self, metric_key: str) -> str:
        return os.path.join(self.cfg.out_root, self.cfg.fig_dir, metric_key)

    def _path_table(self) -> str:
        # Single directory for all metric CSVs
        return os.path.join(self.cfg.out_root, self.cfg.table_dir)

    @staticmethod
    def _ensure_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    # Iteration helper to support both topologies
    def _iter_roots(self, files: Dict[str, Any]):
        only_vals = list(files.values())
        looks_like_dataset = all(isinstance(v, dict) for v in only_vals)
        if looks_like_dataset and all(all(isinstance(v2, dict) for v2 in v.values()) for v in only_vals):
            yield ("root", files)
            return
        for rk, rv in files.items():
            yield (rk, rv)


class CacheManager:
    """
    Manage caching of data for ResultsReporter.
    
    Handles caching of:
    1. Tidy DataFrame (Parquet format for fast loading)
    2. Per-(dataset, metric) arrays (NPZ compressed format)
    3. Aggregated statistics (Parquet format)
    """
    
    def __init__(self, cache_dir: str, enabled: bool = True):
        """
        Initialize CacheManager.
        
        Args:
            cache_dir: Root directory for cache storage
            enabled: Whether caching is enabled (default True)
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        
        # Always initialize manifest (even when disabled)
        self.manifest = {
            "tidy": {},
            "metrics": {},
            "aggregated": {}
        }
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.tidy_dir = self.cache_dir / "tidy"
            self.metric_dir = self.cache_dir / "metrics"
            self.agg_dir = self.cache_dir / "aggregated"
            
            self.tidy_dir.mkdir(exist_ok=True)
            self.metric_dir.mkdir(exist_ok=True)
            self.agg_dir.mkdir(exist_ok=True)
            
            self.manifest_file = self.cache_dir / "cache_manifest.json"
            self._load_manifest()
    
    def _load_manifest(self):
        """Load cache manifest tracking what's cached and when."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                self.manifest = json.load(f)
        # else: manifest already initialized in __init__
    
    def _save_manifest(self):
        """Save cache manifest to disk."""
        if not self.enabled:
            return
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for cache validation."""
        if isinstance(data, pd.DataFrame):
            # Hash based on shape and column names
            return hashlib.md5(
                f"{data.shape}_{list(data.columns)}".encode()
            ).hexdigest()[:8]
        elif isinstance(data, dict):
            # Hash based on keys
            return hashlib.md5(
                str(sorted(data.keys())).encode()
            ).hexdigest()[:8]
        return "unknown"
    
    # ========== Tidy DataFrame Caching ==========
    def cache_tidy_df(self, df: pd.DataFrame, source_hash: str = None) -> None:
        """
        Cache the full tidy DataFrame as Parquet.
        
        Args:
            df: Tidy DataFrame to cache
            source_hash: Optional hash of source files for invalidation
        """
        if not self.enabled or df.empty:
            return
        
        path = self.tidy_dir / "tidy.parquet"
        df.to_parquet(path, compression='snappy', index=False)
        
        self.manifest["tidy"] = {
            "path": str(path),
            "shape": list(df.shape),
            "source_hash": source_hash or self._compute_hash(df)
        }
        self._save_manifest()
        logger.info(f"Cached tidy DataFrame to {path}")
    
    def load_tidy_df(self, source_hash: str = None) -> Optional[pd.DataFrame]:
        """
        Load cached tidy DataFrame if available and valid.
        
        Args:
            source_hash: Optional hash of source files for validation
            
        Returns:
            Cached DataFrame or None if not available
        """
        if not self.enabled:
            return None
        
        if "tidy" not in self.manifest or not self.manifest["tidy"]:
            return None
        
        cache_info = self.manifest["tidy"]
        path = Path(cache_info["path"])
        
        if not path.exists():
            logger.warning(f"Cache file {path} not found")
            return None
        
        # Optional: validate source hash
        if source_hash and cache_info.get("source_hash") != source_hash:
            logger.warning("Cache invalidated: source files changed")
            return None
        
        df = pd.read_parquet(path)
        logger.info(f"Loaded cached tidy DataFrame ({df.shape[0]} rows) from {path}")
        return df
    
    # ========== Per-(Dataset, Metric) Array Caching ==========
    def cache_dataset_metric(self, dataset: str, metric: str, values: np.ndarray,
                            fold_ids: np.ndarray = None, indices: np.ndarray = None) -> None:
        """
        Cache values for a specific (dataset, metric) pair.
        
        Args:
            dataset: Dataset name
            metric: Metric name
            values: Array of metric values
            fold_ids: Optional array of fold identifiers
            indices: Optional array of indices within folds
        """
        if not self.enabled:
            return
        
        # Create safe filename
        safe_dataset = dataset.replace('/', '_').replace('@', '_at_')
        safe_metric = metric.replace(' ', '_').replace('/', '_')
        filename = f"{safe_dataset}_{safe_metric}.npz"
        path = self.metric_dir / filename
        
        # Save as compressed NPZ
        save_dict = {"values": values}
        if fold_ids is not None:
            save_dict["fold_ids"] = fold_ids
        if indices is not None:
            save_dict["indices"] = indices
        
        np.savez_compressed(path, **save_dict)
        
        # Update manifest
        key = f"{dataset}::{metric}"
        self.manifest["metrics"][key] = {
            "path": str(path),
            "dataset": dataset,
            "metric": metric,
            "size": len(values)
        }
        self._save_manifest()
    
    def load_dataset_metric(self, dataset: str, metric: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load cached values for a specific (dataset, metric) pair.
        
        Args:
            dataset: Dataset name
            metric: Metric name
            
        Returns:
            Dict with 'values' and optionally 'fold_ids', 'indices', or None
        """
        if not self.enabled:
            return None
        
        key = f"{dataset}::{metric}"
        if key not in self.manifest["metrics"]:
            return None
        
        path = Path(self.manifest["metrics"][key]["path"])
        if not path.exists():
            return None
        
        data = np.load(path)
        result = {"values": data["values"]}
        if "fold_ids" in data:
            result["fold_ids"] = data["fold_ids"]
        if "indices" in data:
            result["indices"] = data["indices"]
        
        logger.debug(f"Loaded {len(result['values'])} cached values for {dataset}, {metric}")
        return result
    
    # ========== Aggregated Statistics Caching ==========
    def cache_aggregated(self, metric: str, df: pd.DataFrame) -> None:
        """
        Cache aggregated statistics (mean/std/n) for a metric.
        
        Args:
            metric: Metric name
            df: DataFrame with aggregated statistics
        """
        if not self.enabled or df.empty:
            return
        
        safe_metric = metric.replace(' ', '_').replace('/', '_')
        filename = f"agg_{safe_metric}.parquet"
        path = self.agg_dir / filename
        
        df.to_parquet(path, compression='snappy', index=False)
        
        self.manifest["aggregated"][metric] = {
            "path": str(path),
            "shape": list(df.shape)
        }
        self._save_manifest()
    
    def load_aggregated(self, metric: str) -> Optional[pd.DataFrame]:
        """
        Load cached aggregated statistics for a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Cached DataFrame or None
        """
        if not self.enabled:
            return None
        
        if metric not in self.manifest["aggregated"]:
            return None
        
        path = Path(self.manifest["aggregated"][metric]["path"])
        if not path.exists():
            return None
        
        df = pd.read_parquet(path)
        logger.debug(f"Loaded cached aggregated stats for {metric}")
        return df
    
    # ========== Cache Management ==========
    def clear(self, cache_type: str = "all") -> None:
        """
        Clear cache.
        
        Args:
            cache_type: Type of cache to clear ("tidy", "metrics", "aggregated", "all")
        """
        if not self.enabled:
            return
        
        if cache_type in ("tidy", "all"):
            for f in self.tidy_dir.glob("*.parquet"):
                f.unlink()
            self.manifest["tidy"] = {}
            logger.info("Cleared tidy cache")
        
        if cache_type in ("metrics", "all"):
            for f in self.metric_dir.glob("*.npz"):
                f.unlink()
            self.manifest["metrics"] = {}
            logger.info("Cleared metrics cache")
        
        if cache_type in ("aggregated", "all"):
            for f in self.agg_dir.glob("*.parquet"):
                f.unlink()
            self.manifest["aggregated"] = {}
            logger.info("Cleared aggregated cache")
        
        self._save_manifest()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        info = {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "tidy_cached": bool(self.manifest.get("tidy")),
            "metrics_cached": len(self.manifest.get("metrics", {})),
            "aggregated_cached": len(self.manifest.get("aggregated", {}))
        }
        
        if self.enabled and self.cache_dir.exists():
            # Calculate total cache size
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())
            info["total_size_mb"] = total_size / (1024 * 1024)
        
        return info
