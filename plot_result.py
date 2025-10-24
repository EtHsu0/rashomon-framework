""" run_result.py:  Script for generating results, tables, and plots. """

import pandas as pd
import pickle

import argparse
import os
from collections import defaultdict
import re
import json
import logging

from module.result import ResultsReporter, ReporterConfig

logger = logging.getLogger(__name__)

def main(args):
    """ Main function for generating result from trained model.

    Args:
        args (argparse.Namespace): Arguments from command line.
    """
    cfg = ReporterConfig(out_root=args.output_dir)
    
    # Apply suffix to output directories if specified
    if hasattr(args, 'suffix') and args.suffix:
        cfg.fig_dir = f"figures_{args.suffix}"
        cfg.table_dir = f"tables_{args.suffix}"
        logger.info(f"Using suffix '{args.suffix}': {cfg.table_dir}, {cfg.fig_dir}")
    
    # Initialize ResultsReporter with cache options
    cache_dir = args.cache_dir if hasattr(args, 'cache_dir') else None
    use_cache = args.use_cache if hasattr(args, 'use_cache') else True
    result = ResultsReporter(cfg, cache_dir=cache_dir, use_cache=use_cache)
    
    # Handle cache clearing
    if hasattr(args, 'clear_cache') and args.clear_cache:
        logger.info("Clearing cache...")
        result.cache.clear()
    
    # Discover and load files
    file_map = result.discover_files()
    rebuild_cache = hasattr(args, 'rebuild_cache') and args.rebuild_cache
    df = result.load(file_map, use_cache=(use_cache and not rebuild_cache))
    models = df["model"].unique()
    metrics = df["metric"].unique()
    logger.info(f"Loaded DataFrame with {len(df)} rows")
    logger.info(f"Unique models: {models}")
    logger.info("Unique metrics: %s", metrics)
    
    # Show cache info
    cache_info = result.cache.get_cache_info()
    logger.info(f"Cache info: {cache_info}")

    # Parse model/dataset filters from comma-separated strings
    filter_models = args.models.split(',') if args.models else None
    filter_datasets = args.datasets.split(',') if args.datasets else None
    
    if filter_models:
        logger.info(f"Filtering models: {filter_models}")
    if filter_datasets:
        logger.info(f"Filtering datasets: {filter_datasets}")
    
    # Experiment-specific model filters (can be overridden by --models CLI arg)
    if "robustness" in args.output_dir or "stability" in args.output_dir:
        # For robustness: only show specific models if not overridden
        if not filter_models:
            filter_models = ['cart', 'fprdt', 'groot', 'roctv', 'roctn', 
                           'RSET_opt', 'RSET_min', 'RSET_max', 'RSET_kan', 'RSET_Train_Time']
            logger.info(f"Using robustness-specific model filter: {filter_models}")
    elif "fairness" in args.output_dir:
        if not filter_models:
            filter_models = ['post_cart', 'post_xgboost', 'post_rf', 'dpf', 'foct']
            logger.info(f"Using fairness-specific model filter: {filter_models}")
    
    # Generate tables (defaults to True, use --no_table or --no_latex to disable)
    if args.gen_table:
        logger.info("Generating CSV tables...")
        result.build_csvs(models=filter_models, datasets=filter_datasets)
    
    if args.gen_latex:
        logger.info(f"Generating LaTeX tables (orientation: {args.latex_orientation})...")
        result.build_latex_tables(
            orientation=args.latex_orientation,
            models=filter_models,
            datasets=filter_datasets
        )

            # Generate RSET outperformance analysis if requested
    if args.gen_outperformance:
        logger.info("Generating RSET outperformance analysis...")
        outperf_paths = result.build_rset_outperformance_table(
            metrics=None,  # Use all available metrics
            baselines=filter_models,
            datasets=filter_datasets,
            output_formats=['csv', 'md', 'latex']
        )
        for fmt, path in outperf_paths.items():
            logger.info(f"Generated {fmt.upper()} outperformance table: {path}")
    
    
    if "fairness" in args.output_dir or args.output_dir == "out/fairness":
        fairness_datasets = ["adult", "bank", "compas", "census-income", "communities-crime", "german-credit", "oulad", "student-mat", "student-por"]
        fairness_metrics = [
            ('Statistical Parity', 'SP'),
            ('Equal Opportunity', 'EO'),
            ('Equalized Odds', 'EOdds')
        ]
        
        logger.info(f"Creating fairness 2D density grid: {len(fairness_datasets)} datasets × {len(fairness_metrics)} metrics")
        
        # Build grid: rows = datasets, columns = fairness metrics
        grid = []
        for dataset in fairness_datasets:
            row = []
            for metric_full, metric_short in fairness_metrics:
                row.append({
                    'type': 'density_2d', 
                    'metric': metric_full, 
                    'metric_y': 'Test Accuracy',
                    'dataset': dataset, 
                    'kind': 'kde',
                    'title': f'{dataset}'  # Show dataset name in subplot title
                })
            grid.append(row)
        
        # Calculate figure size and legend position
        num_rows = len(fairness_datasets)
        fig_height = num_rows * 6  # 6 inches per row
        legend_bbox_y = -0.04 * (6 / num_rows) - 0.01  # Scale legend position
        
        result.plot_grid(
            grid,
            filename='fairness_2d_all_datasets_grid',
            figsize=(18, fig_height),  # 3 columns × 6 rows
            share_x=False,        
            share_y=True,        
            shared_legend=True,
            legend_bbox_to_anchor=(0.5, legend_bbox_y),
            legend_ncol=4,        # Accommodate RSET + multiple baselines
            wspace=0.3,
            hspace=0.3,
        )

    if "robustness" in args.output_dir or args.output_dir == "out/robustness":
        # Grid plot: 14 datasets in 7 rows × 2 columns
        
        datasets = sorted(df["dataset"].unique())
        logger.info(f"Creating robustness grid with {len(datasets)} datasets and {len(filter_models)} models")
        
        # Build grid: 7 rows × 2 columns in reading order (left-to-right, top-to-bottom)
        ncols = 2
        nrows = (len(datasets) + ncols - 1) // ncols  # Ceiling division
        
        grid = []
        dataset_idx = 0
        for row_idx in range(nrows):
            row = []
            for col_idx in range(ncols):
                if dataset_idx < len(datasets):
                    ds = datasets[dataset_idx]
                    row.append({'type': 'density_1d', 'metric': 'Test Adv Accuracy', 'dataset': ds, 'title': ds})
                    dataset_idx += 1
                else:
                    row.append({'type': 'empty'})  # Empty cell if odd number
            grid.append(row)

        # Number of rows:
        num_rows = len(grid)
        fig_y = num_rows * 4  # 4 inches per row
        legend_bbox_y = -0.0425 * (7 / num_rows) - 0.01  # Scale legend position with number of rows
        
        result.plot_grid(
            grid,
            filename='robustness_all_datasets_grid',
            figsize=(24, fig_y),  # 2 columns × 7 rows
            share_x=False,     # Don't share x-axis (different models per dataset)
            share_y=False,      # Share y-axis (same metric scale)
            shared_legend=True,
            legend_ncol=5,     # 4 columns to fit 8 models nicely
            legend_bbox_to_anchor=(0.5, legend_bbox_y),
            legend_bottom_margin=0.05,  # Smaller bottom margin for tall figure
            wspace=0.3,
            hspace=0.2,
            allowed_models=filter_models,
        )
        # result.plot_density_1d("Test Adv Accuracy")

    if "integration" in args.output_dir:
        # Grid plot: Integration experiment showing RSET special tree methods
        # 4 columns (datasets) × 5 rows (metrics)
        logger.info("Creating integration grid with RSET special tree methods")
        
        # Metrics to show (integration experiment focuses on these)
        # Define both full names (for lookup) and display names (for Y-axis labels)
        integration_metrics_config = [
            ("Test Accuracy", "Task Acc."),
            ("Test Adv Accuracy", "Adv. Acc."),
            ("Stability Acc Mean", "Stab"),
            ("Statistical Parity", "SP"),
            ("MIA Blackbox", "1 - MIA"),
        ]
        
        # Select 4 representative datasets for integration grid
        all_datasets = sorted(df["dataset"].unique())
        # Choose diverse datasets: small, medium, large, and fairness-relevant
        selected_datasets = ["bank", "german-credit", "oulad"]
        
        logger.info(f"Using 4 datasets for integration grid: {selected_datasets}")
        
        # Filter to models that are RSET special trees
        rset_special_models = [
            'RSET_opt', 'RSET_min', 'RSET_max', 
            'RSET_sp', # 'RSET_eopp', 'RSET_eo''RSET_kan'
            "RSET_kan"
        ]
        
        # Build grid: rows = metrics, columns = 4 datasets
        grid = []
        for row_idx, (metric_full, metric_label) in enumerate(integration_metrics_config):
            row = []
            is_last_row = (row_idx == len(integration_metrics_config) - 1)
            for col_idx, dataset in enumerate(selected_datasets):
                cell = {
                    'type': 'barplot_1d',
                    'metric': metric_full,
                    'dataset': dataset,
                    'share_ylabel': True,  # Share ylabel across row (only leftmost shows label)
                    'title': '',  # No subplot titles
                }
                # Only add ylabel to leftmost column
                if col_idx == 0:
                    cell['ylabel'] = metric_label
                # Show dataset name as X-label on bottom row only
                if is_last_row:
                    cell['xlabel'] = dataset
                    cell['show_xlabel'] = True
                else:
                    cell['show_xlabel'] = False
                row.append(cell)
            grid.append(row)
        
        # Specify our own color scheme for RSET special models
        result.cfg.model_color_map.update({
            'RSET_opt': '#e41a1c',  
            'RSET_min': '#4daf4a',   
            'RSET_max': "#ffae4a",   
            'RSET_kan': '#999999', 
            'RSET_sp':  '#a65628',  
            'RSET_eopp': '#377eb8',
            'RSET_eo': '#984ea3',
        })
        
        result.plot_grid(
            grid,
            filename='integration_rset_methods_grid',
            figsize=(24, 12),   # 4 columns × 5 rows
            share_x=True,      # Don't share x-axis (different models may be present)
            share_y=True,      # Don't share y-axis (different metrics have different scales)
            shared_legend=True,
            legend_ncol=1,      # Vertical legend (1 column)
            legend_loc='center left',  # Position relative to bbox anchor
            legend_bbox_to_anchor=(1.02, 0.5),  # (x, y): x>1 places it outside right, y=0.5 centers vertically
            legend_bottom_margin=0.05,
            wspace=0,
            hspace=0,
            allowed_models=rset_special_models,
        )
    
    if "stability" in args.output_dir or args.output_dir == "out/stability":
        datasets = sorted(df["dataset"].unique())
        logger.info(f"Creating stability grids with {len(datasets)} datasets and {len(filter_models)} models")
        
        # Build grid for Stability Acc Mean: 7 rows × 2 columns
        ncols = 2
        nrows = (len(datasets) + ncols - 1) // ncols  # Ceiling division
        
        # Grouped shift plots: separate grids for Mean and Worst
        # Grid for Stability Acc Mean with grouped shifts: 7 rows × 2 columns
        logger.info("Creating grouped shift plots for stability (Mean)...")
        grid_shifts_mean = []
        dataset_idx = 0
        for row_idx in range(nrows):
            row = []
            for col_idx in range(ncols):
                if dataset_idx < len(datasets):
                    ds = datasets[dataset_idx]
                    row.append({'type': 'barplot_grouped_shifts', 'metric': 'Stability Acc Mean', 
                               'dataset': ds, 'title': ds})
                    dataset_idx += 1
                else:
                    row.append({'type': 'empty'})
            grid_shifts_mean.append(row)


        num_rows = len(grid_shifts_mean)
        fig_y = num_rows * 4  # 4 inches per row
        legend_bbox_y = -0.0425 * (7 / num_rows) - 0.01
        result.plot_grid(
            grid_shifts_mean,
            filename='stability_grouped_shifts_mean_grid',
            figsize=(24, fig_y),  # 2 columns × 7 rows
            share_x=True,     # Don't share x-axis (different datasets)
            share_y=False,      # Don't share y-axis (different accuracy scale)
            shared_legend=True,
            legend_bbox_to_anchor=(0.5, legend_bbox_y),
            legend_bottom_margin=0.05,  # Smaller bottom margin for tall figure
            legend_ncol=5,
            wspace=0.3,
            hspace=0.2,
            allowed_models=filter_models,
        )
        
        # Grid for Stability Acc Worst with grouped shifts: 7 rows × 2 columns
        logger.info("Creating grouped shift plots for stability (Worst)...")
        grid_shifts_worst = []
        dataset_idx = 0
        for row_idx in range(nrows):
            row = []
            for col_idx in range(ncols):
                if dataset_idx < len(datasets):
                    ds = datasets[dataset_idx]
                    row.append({'type': 'barplot_grouped_shifts', 'metric': 'Stability Acc Worst', 
                               'dataset': ds, 'title': ds})
                    dataset_idx += 1
                else:
                    row.append({'type': 'empty'})
            grid_shifts_worst.append(row)
        
        num_rows = len(grid_shifts_worst)
        fig_y = num_rows * 4  # 4 inches per row
        legend_bbox_y = -0.0425 * (7 / num_rows) - 0.01

        result.plot_grid(
            grid_shifts_worst,
            filename='stability_grouped_shifts_worst_grid',
            figsize=(24, fig_y),  # 2 columns × 7 rows
            share_x=True,     # Don't share x-axis (different datasets)
            share_y=False,      # Share y-axis (same accuracy scale)
            shared_legend=True,
            legend_bbox_to_anchor=(0.5, legend_bbox_y),
            legend_bottom_margin=0.05,  # Smaller bottom margin for tall figure
            legend_ncol=5,
            wspace=0.3,
            hspace=0.2,
            allowed_models=filter_models,
        )
    
def merge_robust_stab_and_plot_grid():
    """
    Loads robustness and stability results, merges unique metrics, saves to diff dir, and plots combined grid.
    Uses caching for efficient loading (especially important for robustness with many RSET trees).
    """
    import os
    from module.result import ResultsReporter, ReporterConfig
    robust_dir = 'out/robustness_4'
    stab_dir = 'out/stability_4'
    diff_dir = 'out/robust_stab_diff_4'
    os.makedirs(diff_dir, exist_ok=True)

    cfg_robust = ReporterConfig(out_root=robust_dir)
    reporter_robust = ResultsReporter(cfg_robust, use_cache=True)  # Enable cache
    file_map_robust = reporter_robust.discover_files()
    df_robust = reporter_robust.load(file_map_robust, use_cache=True)
    df_robust['source_dir'] = robust_dir
    logger.info(f"Loaded robustness: {len(df_robust)} rows")

    # Load stability with caching
    cfg_stab = ReporterConfig(out_root=stab_dir)
    reporter_stab = ResultsReporter(cfg_stab, use_cache=True)  # Enable cache
    file_map_stab = reporter_stab.discover_files()
    df_stab = reporter_stab.load(file_map_stab, use_cache=True)  # Use cache
    df_stab['source_dir'] = stab_dir
    logger.info(f"Loaded stability: {len(df_stab)} rows")

    # Remove duplicate metric names (keep only unique metrics per dataset/model)
    robust_metrics = set(df_robust['metric'].unique())
    stab_metrics = set(df_stab['metric'].unique())
    logger.info(f"Robustness metrics: {sorted(robust_metrics)}")
    logger.info(f"Stability metrics: {sorted(stab_metrics)}")
    
    stab_mask = ~df_stab['metric'].isin(robust_metrics)
    df_stab_unique = df_stab[stab_mask]
    stab_unique_metrics = set(df_stab_unique['metric'].unique())
    logger.info(f"Stability unique metrics (after removing overlap): {sorted(stab_unique_metrics)}")

    # Merge
    merged_df = pd.concat([df_robust, df_stab_unique], ignore_index=True)
    logger.info(f"Merged DataFrame: {len(merged_df)} rows, metrics: {sorted(merged_df['metric'].unique())}")
    logger.info(f"Merged Unique models: {sorted(merged_df['model'].unique())}")

    cfg_diff = ReporterConfig(out_root=diff_dir)
    reporter_diff = ResultsReporter(cfg_diff, cache_dir=None, use_cache=False, 
                                    source_reporters=[reporter_robust, reporter_stab])
    
    logger.info(f"After mapping, metrics: {sorted(merged_df['metric'].unique())}")
    logger.info(f"After mapping, unique models: {sorted(merged_df['model'].unique())}")
    print(merged_df[merged_df['metric'] == 'Test Adv Accuracy']['model'].unique())
    
    reporter_diff._tidy = merged_df
    reporter_diff._files_map = file_map_robust # To Load RSET
    logger.info("Created merged reporter with source reporters for RSET distribution loading")
    
    # Debug: Check what models are in the merged data for robustness metric
    robust_models = merged_df[merged_df['metric'] == 'Test Adv Accuracy']['model'].unique()
    logger.info(f"Models in merged data for Test Adv Accuracy: {sorted(robust_models)}")

    # Models and datasets (shared)
    filter_models = ['cart', 'fprdt', 'groot', 'roctv', 'roctn', 'RSET_opt', 'RSET_min', 'RSET_max', 'RSET_kan']
    
    # Select specific datasets
    selected_datasets = ['banknote', 'fico', 'spambase']
    logger.info(f"Using selected datasets: {selected_datasets}")

    # Build grid: 3 rows × 2 columns (density, grouped shift)
    nrows = 3
    grid = []
    for i, ds in enumerate(selected_datasets[:nrows]):
        row = [
            {'type': 'density_1d', 'metric': 'Test Adv Accuracy', 'dataset': ds, 'title': f'{ds}'},
            {'type': 'barplot_grouped_shifts', 'metric': 'Stability Acc Mean', 'dataset': ds, 'title': f'{ds}'}
        ]
        grid.append(row)

    reporter_diff.plot_grid(
        grid,
        filename='robust_stab_combined_grid',
        figsize=(20, 10),
        share_x=False,
        share_y=False,
        shared_legend=True,
        legend_ncol=5,
        legend_bbox_to_anchor=(0.5, -0.125),
        legend_bottom_margin=0.05,
        wspace=0.3,
        hspace=0.2,
        allowed_models=filter_models,
    )
    print("Plotted combined robustness/stability grid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result from trained model.")
    # Input/Output Section
    io_group = parser.add_argument_group('Input/Output', 'Arguments related to file and folder paths')
    io_group.add_argument('--output_dir', type=str, default="out/out_paper", help='Directory of experiment outputs')
    io_group.add_argument('--result_dir', type=str, default="result", help='Directory of results')
    io_group.add_argument('--model_dir', type=str, default="model", help='Directory of models')
    io_group.add_argument('--param_dir', type=str, default="param", help='Directory of parameters')
    io_group.add_argument('--fig_dir', type=str, default="fig", help='Directory for saving figures')
    io_group.add_argument('--table_dir', type=str, default="table", help='Directory for saving tables')

    setup_group = parser.add_argument_group('Setup', 'Arguments related to setup')
    setup_group.add_argument('--log_level', type=str, default="INFO", help='Logger level')
    setup_group.add_argument('--dataset_list', type=str, default=None, help='List of datasets to run on')
    
    # Cache options
    cache_group = parser.add_argument_group('Cache', 'Arguments related to caching')
    cache_group.add_argument('--cache_dir', type=str, default=None, help='Directory for cache storage (default: <output_dir>/.cache)')
    cache_group.add_argument('--use_cache', action='store_true', default=True, help='Use cached data if available (default: True)')
    cache_group.add_argument('--no_cache', dest='use_cache', action='store_false', help='Disable caching')
    cache_group.add_argument('--rebuild_cache', action='store_true', help='Force rebuild cache from source files')
    cache_group.add_argument('--clear_cache', action='store_true', help='Clear all cached data and exit')
    
    result_group = parser.add_argument_group('Result', 'Arguments related to result generation')
    result_group.add_argument('--gen_table', action='store_true', default=True, help='Generate CSV tables (default: True)')
    result_group.add_argument('--no_table', dest='gen_table', action='store_false', help='Disable CSV table generation')
    result_group.add_argument('--gen_latex', action='store_true', default=True, help='Generate LaTeX tables with booktabs (default: True)')
    result_group.add_argument('--no_latex', dest='gen_latex', action='store_false', help='Disable LaTeX table generation')
    result_group.add_argument('--latex_orientation', type=str, default='datasets_rows',
                            choices=['datasets_rows', 'models_rows'],
                            help='LaTeX table orientation (default: datasets_rows)')
    result_group.add_argument('--models', type=str, default=None,
                            help='Comma-separated list of models to include in tables (default: all)')
    result_group.add_argument('--datasets', type=str, default=None,
                            help='Comma-separated list of datasets to include in tables (default: all)')
    result_group.add_argument('--suffix', type=str, default=None,
                            help='Suffix to append to output directories (e.g., "short" -> tables_short, figures_short)')
    result_group.add_argument('--gen_density', action='store_true', help='Generate density plot')
    result_group.add_argument('--gen_outperformance', action='store_true', 
                            help='Generate RSET outperformance analysis (% trees beating baselines)')

    parser.add_argument('--merge_robust_stab', action='store_true', help='Run merge robustness and stability, plot combined grid')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging._nameToLevel[args.log_level],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log message format
        datefmt="%Y-%m-%d %H:%M:%S",              # Date and time format
        handlers=[
            logging.FileHandler("result.log"),     # Log to a file
            logging.StreamHandler()               # Log to console
        ]
    )

    if args.merge_robust_stab:
        merge_robust_stab_and_plot_grid()
    else:
        main(args)
