"""run_exp.py: Main script to run trustworthy ML experiments with various models and metrics.

This script handles argument parsing, dataset loading, model initialization, and 
experiment execution. It automatically discovers and imports all model and metric
implementations from the module package.
"""
import sys
import argparse
import logging
import importlib
import pkgutil
import os
import yaml

from module.hparams import Hparams, update_hparams
from module.experiment import Experiment
from module.metric.metric import Metrics
from module.datasets import DatasetLoader
import module.model as model_pkg
import module.metric as metric_pkg


def load_all(pkg):
    """Import all submodules and packages recursively for model/metric discovery.
    
    This function walks through the package tree and imports all non-private
    modules to trigger their registration decorators.
    
    Args:
        pkg (module): The package to recursively import (e.g., module.model).
    """
    prefix = pkg.__name__ + "."
    for modinfo in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
        name = modinfo.name
        parts = name.split(".")
        # Skip private/testy modules by convention
        if any(p.startswith("_") for p in parts):
            continue
        if parts[-1] in {"tests", "test", "contrib_examples"}:
            continue
        importlib.import_module(name)


load_all(model_pkg)
load_all(metric_pkg)


def main(args):
    """Main function to run the experiment.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            configuration for the experiment run.
    """
    logger = logging.getLogger("main")
    
    # Load config file for epsilon configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    hparams = Hparams(args)
    metrics = Metrics(args, hparams)
    dataset = DatasetLoader(args.dataset, hparams.rs, args)
    
    # Store dataset name and config in hparams for use by metrics
    hparams.dataset_name = dataset.dataset_name
    hparams.config = config

    update_hparams(hparams, args, dataset)
    logger.info("Model hparams: %s", hparams.model_params)

    experiment = Experiment(dataset, hparams, metrics)
    experiment.cross_validate(fold=args.fold)


def confirm_arguments(args):
    """Function to confirm the arguments before running the experiment.

    Args:
        args (argparse): argparse arguments
    """
    print("\nArguments:")

    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    confirm = input("\nDo you want to proceed with these settings? (yes/no): ")
    if confirm.lower() not in ["yes", "y"]:
        print("Exiting...")
        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with specified parameters.")
    parser.add_argument("--skip_confirm", action="store_true", help="Skip confirmation for args")
    parser.add_argument("--model", type=str, required=True, help="Model to train, this model MUST be in the registry.")

    io_group = parser.add_argument_group("Input/Output", "Arguments related to file paths")
    io_group.add_argument("--output_dir", type=str, default="out/test",
            help="Directory for saved models and results")
    io_group.add_argument("--model_dir", type=str, default="model",
            help="Directory for saved models")
    io_group.add_argument("--param_dir", type=str, default="param", help="Directory for parameters")
    io_group.add_argument("--result_dir", type=str,
            default="result", help="Directory for saved results")

    exp_group = parser.add_argument_group("Experiment", "Arguments related to experiment setup")
    exp_group.add_argument("--dataset", type=str, default="compas@dpf", help="Dataset name, you can also specify source for some dataset by dataset@source")
    exp_group.add_argument("--random_state", type=int, default=42, help="Random state")
    exp_group.add_argument("--tune", action="store_true", help="Tune the model hyperparameters")
    exp_group.add_argument("--selection", action="store_true", help="Use selection set")
    exp_group.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-val")
    exp_group.add_argument("--fold", type=int, help="Fold index for parallization. \
            If this is not provided, then all five folds will run.")
    exp_group.add_argument("--retrain", action="store_true",
            help="Retrain the model even if it exists in the model directory")
    exp_group.add_argument("--binarize_mode", type=str, 
            help="Enable dataset-level binarization. Supported values: 'gbdt' (use GBDT-based binarization)")
    exp_group.add_argument("--retune", action="store_true",
            help="Retune the model even if cached parameters exist")

    metrics_group = parser.add_argument_group("Metrics Pipeline",
            "Arguments related to the metrics")
    metrics_group.add_argument("--reset_results", action="store_true",
            help="Reset the results directory before running the experiment")
    metrics_group.add_argument("--config", type=str,
            default="config/default.yaml", help="Metrics configuration file")

    logging_group = parser.add_argument_group("Logging Parameters", "Logging Arguments")
    logging_group.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    logging_group.add_argument("--log_file", type=str, default="out/log/experiments.log", help="Log file path")

    parse_args, unknown = parser.parse_known_args()
    custom_args = {}
    if unknown:
        i = 0
        while i < len(unknown):
            arg = unknown[i]
            if arg.startswith("--"):
                key = arg[2:]
                if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                    value = unknown[i + 1]
                    i += 2
                    custom_args[key] =  value
                else:
                    custom_args[key] = True # Flag
                    i += 1
            else:
                raise ValueError(f"Unexpected argument format: {arg}, args must be in either --flag or --key value format.")
    parse_args.__dict__.update(custom_args)

    if not parse_args.skip_confirm:
        confirm_arguments(parse_args)

    os.makedirs(parse_args.output_dir, exist_ok=True)
    os.makedirs(f"{parse_args.output_dir}/{parse_args.result_dir}", exist_ok=True)
    os.makedirs(f"{parse_args.output_dir}/{parse_args.model_dir}", exist_ok=True)
    os.makedirs(f"{parse_args.output_dir}/{parse_args.param_dir}", exist_ok=True)
    os.makedirs(os.path.dirname(parse_args.log_file), exist_ok=True)

    logging.basicConfig(
        level=parse_args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(parse_args.log_file),
            logging.StreamHandler(),
        ],
    )

    main(parse_args)
