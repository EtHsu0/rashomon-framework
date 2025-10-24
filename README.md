# The Rashomon Set Has It All: Analyzing Trustworthiness of Trees under Multiplicity (NeurIPS D&B 2025)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete implementation and experimental framework for analyzing trustworthiness of Rashomon sets and baseline tree-based models. The framework provides a comprehensive evaluation suite for comparing models across multiple trustworthiness dimensions: **robustness**, **fairness**, **privacy**, and **stability**.

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/EtHsu0/rashomon-framework.git
   cd rashomon-framework
   ```

2. **Install dependencies**
   ```bash
   pip install pyyaml scikit-learn xgboost pandas tqdm typing-extensions cvxpy numba treefarms gosdt
   pip install --no-deps roct odtlearn diffprivlib cvxpy groot-trees
   pip install --no-deps adversarial-robustness-toolbox
   ```
   
   **Note**: Some models require additional dependencies:
   - **Gurobi**: Required for ROCT-N and ROCT-V (optimization-based robust models). Obtain a license from [gurobi.com](https://www.gurobi.com/)
   - **NumPy**: Some packages (groot-trees, roct) does not support NumPy>2.0
   - If you only want to test with treefarms or simple models, comment/delete specific model files in `module/model/`

3. **Run a basic experiment**
   ```bash
   python run_exp.py \
       --skip_confirm \
       --model cart \
       --dataset compas \
       --config config/robustness.yaml \
       --output_dir out/test
   ```

4. **Generate results**
   ```bash
   python run_result.py \
       --output_dir out/test \
       --gen_table
   ```

## Framework Overview

Our framework consists of two main components with a decorator-based registration system:

- **Models** (`module/model/`) - Tree-based ML models across different trustworthiness categories (robustness, fairness, privacy, and Rashomon sets). Models self-register using the `@register_model()` decorator.
- **Metrics** (`module/metric/`) - Evaluation metrics for different trustworthiness dimensions. Metrics self-register using the `@register_metric()` decorator and are configured via YAML files.

**Key Features:**
- **Auto-discovery**: Models and metrics are automatically loaded without manual import management
- **YAML-based configuration**: Metrics are declared and configured in `config/*.yaml` files
- **Extensible design**: Easy to add new models, metrics, and datasets
- **Hyperparameter tuning**: Built-in support with race-condition-free caching
- **Feature binarization**: Automatic binarization support for models requiring binary features (e.g., TreeFARMS)

### Quick Project Structure

```
rset_benchmark/
├── run_exp.py              # Experiment runner (train & evaluate models)
├── run_result.py           # Results analysis (generate tables & plots)
├── config/                 # YAML configs for metrics (robustness, fairness, privacy, stability)
├── data/                   # Datasets (CSV files)
│   ├── *.csv               #   - Direct dataset files (compas, diabetes, fico, etc.)
│   ├── dpf/                #   - DPF-preprocessed variants for fairness experiments
│   └── dataset_info.*      #   - Dataset metadata and citations
├── module/
│   ├── model/              # Models organized by category:
│   │   ├── core/           #   - Base classes (BaseEstimator, RsetBase, PostBase)
│   │   ├── standard/       #   - CART baseline
│   │   ├── rash/           #   - TreeFARMS (Rashomon sets)
│   │   ├── robustness/     #   - GROOT, FPRDT, ROCT-N/V
│   │   ├── fairness/       #   - DPF, FairOCT, Post-processing models
│   │   ├── privacy/        #   - PRIVA, BDPT, DPLDT
│   │   └── lib/            #   - Shared implementations & utilities
│   ├── metric/             # Metrics: accuracy, kantch_attack, stability, fairness, MIA
│   ├── datasets.py         # 20+ dataset loaders with fairness support
│   ├── experiment.py       # Cross-validation & hyperparameter tuning
│   └── result.py           # Result aggregation & visualization
└── out/                    # Output: models, results, logs, tables, figures
```

*See [Detailed Project Structure](#project-structure) section below for complete architecture documentation.*

For a detailed list of supported models and metrics, see the **Methods** section. For instructions on extending the framework, see the **Development** section.

## Methods

### Models

Our framework supports 14 tree-based models across four categories:

#### Rashomon Set Models
- **TreeFARMS** (`treefarms`) - Rashomon set enumeration using GOSDT for optimal tree generation. Provides multiple decision trees within an accuracy bound for trustworthiness analysis.

#### Standard Models
- **CART** (`cart`) - Classical decision trees from scikit-learn. Used as baseline.

#### Robustness Models
- **GROOT** (`groot`) - Globally Robust Optimal Trees. Robust decision trees optimized against adversarial perturbations.
- **FPRDT** (`fprdt`) - False Positive Rate Decision Tree. Focuses on minimizing false positives under adversarial attacks.
- **ROCT-N** (`roctn`) - Robust Optimal Classification Trees (Node-based). Optimization-based robust trees (requires Gurobi).
- **ROCT-V** (`roctv`) - Robust Optimal Classification Trees (Value-based). Alternative formulation of ROCT (requires Gurobi).

#### Fairness Models
- **DPF** (`dpf`) - Fairness-aware decision tree learning with demographic parity constraints.
- **FairOCT** (`foct`) - Fair Optimal Classification Trees. Optimization-based approach with fairness constraints.
- **Post-CART** (`post_cart`) - Post-processing fairness correction applied to CART.
- **Post-XGBoost** (`post_xgboost`) - Post-processing fairness correction applied to XGBoost.
- **Post-RF** (`post_rf`) - Post-processing fairness correction applied to Random Forest.

#### Privacy Models
- **PRIVA** (`priva`) - Private tree learning with differential privacy guarantees.
- **BDPT** (`bdpt`) - Binary Decision tree with Privacy Training.
- **DPLDT** (`dpldt`) - Differentially Private Linear Decision Trees.

### Metrics

Metrics are configured via YAML files in the `config/` directory. Each metric can have dataset-specific parameters.

#### Standard Metrics
- **Accuracy** (`accuracy`) - Train and test accuracy scores
  - Configured in: All config files
  - Returns: `train_accuracy`, `test_accuracy`

#### Robustness Metrics
- **Kantchelian Attack** (`kantch_attack`) - Adversarial robustness using L-infinity norm attacks
  - Configured in: `config/robustness.yaml`
  - Returns: `kantch_attack_adv_acc` (accuracy on adversarial examples)
  - Parameters: 
    - `epsilon` - Perturbation budget (can be dataset-specific)
    - `use_dataset_epsilon` - Whether to use dataset-specific epsilon values

- **Stability** (`stability`) - Robustness to small input perturbations
  - Configured in: `config/stability.yaml`
  - Returns: Multiple stability scores under different noise models
  - Parameters:
    - `num_trials` - Number of noise trials (default: 500)
    - `delta` - Noise distribution shift parameter

#### Fairness Metrics
- **Fairness** (`fairness`) - Multiple fairness criteria evaluation
  - Configured in: `config/fairness.yaml`
  - Returns: Demographic parity, equal opportunity, and equalized odds differences
  - Parameters:
    - `criterion` - Fairness criterion: `'sp'` (statistical parity), `'eopp'` (equal opportunity), `'eo'` (equalized odds), or `'all'` (default)
  - Requires: Sensitive attributes in dataset

#### Privacy Metrics
- **Membership Inference Attack** (`membership_inference`) - Privacy leakage via MIA
  - Configured in: `config/privacy.yaml`
  - Returns: Attack accuracy using label-only and black-box attacks
  - Uses: ART (Adversarial Robustness Toolbox) for attack implementation

### Datasets

The framework supports 20+ datasets for trustworthiness evaluation. Use `dataset@source` syntax to specify data source variants.

**Examples:**
- `compas` - COMPAS recidivism dataset
- `adult` - Adult income dataset
- `adult@dpf` - Adult dataset with DPF preprocessing
- `german-credit` - German credit dataset
- `bank` - Bank marketing dataset

**Integrated Datasets:**
`adult`, `bank`, `california-houses`, `compas`, `credit-fusion`, `default-credit`, `diabetes-130US`

**Robustness/Stability Datasets:**
`banknote`, `blood`, `breast-w`, `cylinder-bands`, `diabetes`, `fico`, `haberman`, `ionosphere`, `mimic`, `parkinsons`, `sonar`, `spambase`, `spectf`, `wine-quality`

**Fairness Datasets (with sensitive attributes):**
`adult`, `bank`, `census-income`, `communities-crime`, `compas`, `german-credit`, `oulad`, `student-mat`, `student-por`

**Privacy Datasets:**
`adult`, `bank`, `diabetes-130US`, `german-credit`, `oulad`

## Usage Examples

### Running Experiments

**Basic experiment with a single model:**
```bash
python run_exp.py \
    --skip_confirm \
    --model cart \
    --dataset compas \
    --config config/robustness.yaml \
    --output_dir out/robustness
```

**With hyperparameter tuning:**
```bash
python run_exp.py \
    --skip_confirm \
    --model treefarms \
    --dataset adult \
    --config config/fairness.yaml \
    --output_dir out/fairness \
    --tune \
    --selection
```

**Run specific fold for parallelization:**
```bash
python run_exp.py \
    --skip_confirm \
    --model groot \
    --dataset compas \
    --config config/robustness.yaml \
    --output_dir out/robustness \
    --fold 0 \
    --epsilon 0.1
```

**With dataset-level binarization:**
```bash
python run_exp.py \
    --skip_confirm \
    --model treefarms \
    --dataset compas \
    --config config/robustness.yaml \
    --output_dir out/robustness \
    --binarize_mode gbdt
```

### Generating Results

**Generate CSV tables:**
```bash
python run_result.py \
    --output_dir out/robustness \
    --gen_table
```

**Generate LaTeX tables:**
```bash
python run_result.py \
    --output_dir out/fairness \
    --gen_latex \
    --latex_orientation horizontal
```

**Filter specific models and datasets:**
```bash
python run_result.py \
    --output_dir out/robustness \
    --gen_table \
    --models cart,groot,treefarms \
    --datasets compas,adult,german-credit
```

**Generate plots:**
```bash
python run_result.py \
    --output_dir out/robustness \
    --gen_density
```

### Output Structure

Experiments generate the following directory structure:
```
out/<experiment_name>/
├── model/                    # Trained models (pickle)
│   └── {model}_{dataset}_fold_{n}.pkl
├── result/                   # Evaluation results (pickle)
│   └── {model}_{dataset}_fold_{n}.pkl
├── param/                    # Hyperparameters (pickle)
│   └── {model}_{dataset}_fold_{n}.pkl
├── log/                      # Execution logs
├── figures/                  # Generated plots (created by run_result.py)
└── tables/                   # Generated tables (created by run_result.py)
```

## Development

### Adding a New Model

1. **Create model file** in appropriate category folder (e.g., `module/model/robustness/mymodel.py`)

2. **Register the model** using decorator:
```python
from module.model.core.base import BaseEstimator, register_model
from module.hparams import register_hparams

@register_model("mymodel")  # Name used in CLI --model argument
class MyModel(BaseEstimator):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = params
        self.model = None
    
    def fit(self, X, y):
        # Training logic
        return self
    
    def predict(self, X):
        # Prediction logic
        return predictions
    
    def score(self, X, y):
        # Scoring logic (optional, defaults to accuracy)
        return score
```

3. **Register hyperparameters**:
```python
@register_hparams("mymodel")  # Must match model name
def update_hparams(hparams, args, dataset):
    hparams.model_params = {
        'param1': getattr(args, 'param1', default_value),
        'param2': getattr(args, 'param2', default_value),
    }
```

4. **Optional: Add hyperparameter tuning**:
```python
def tune(self, nested_cv):
    # Grid search or other tuning logic
    # Must return (best_config_dict, all_scores_dict)
    return best_config, all_scores
```

**Important:** 
- **DO NOT** modify `module/model/__init__.py` - models are auto-discovered via decorator registration
- Model name in `@register_model()` must match name in `@register_hparams()`
- Base classes: `BaseEstimator` (standard), `RsetBase` (Rashomon sets), `PostBase` (fairness post-processing)

### Adding a New Metric

1. **Create metric file** in `module/metric/` (e.g., `module/metric/my_metric.py`)

2. **Register the metric** using decorator:
```python
from module.metric.base_metrics import BaseMetric, register_metric

@register_metric()
class MyMetric(BaseMetric):
    NAME = "my_metric"  # Used in YAML configs
    REQUIRES_BINARY_FEATURES = False  # Set True if needs binarized data
    
    def setup(self, model, hparams, split, **params):
        # Optional: Pre-computation or validation
        # Called once before compute()
        pass
    
    def compute(self, predictions, split, **params):
        # Required: Perform evaluation
        # predictions dict contains: "train", "test", "pred_fn", "model"
        # **params contains extra parameters from YAML config
        score = evaluate(predictions["test"], split.y_test)
        return {"my_score": score}
    
    def cleanup(self):
        # Optional: Clean up resources
        pass
```

3. **Add to YAML config** (e.g., `config/robustness.yaml`):
```yaml
metrics:
  - name: my_metric
    param1: value1
    param2: value2
```

**Important:**
- Metric `NAME` attribute must match the `name` in YAML
- Extra YAML parameters (beyond `name`) are passed to `compute(**params)`
- Return dict with descriptive keys (e.g., `"train_accuracy"`, `"test_fairness"`)

### Adding a New Dataset

1. **Add loader method** to `module/datasets.py`:
```python
def load_my_dataset(self):
    # Load and preprocess data
    # Set self.X, self.y, self.feat_name
    # Optionally set sensitive attribute info for fairness
    pass
```

2. **Register in mapper** dictionary (in `DatasetLoader.__init__`):
```python
self.mapper = {
    # ... existing datasets ...
    "my-dataset": self.load_my_dataset,
}
```

3. **For fairness datasets**, specify sensitive attribute:
```python
def load_my_dataset(self):
    # ... load data ...
    if self.fairness_mode:
        self.sensitive_idx = 5  # Column index of sensitive attribute
        self.sensitive_threshold = (-np.inf, 0.5)  # Binary threshold
```

### Key File Descriptions

#### `run_exp.py`
Entry point for running experiments. Handles:
- Argument parsing with custom parameter support (`--key value`)
- Loading models, metrics, and datasets via decorator registration
- Orchestrating experiment execution via `Experiment` class

#### `run_result.py`
Post-experiment analysis and visualization. Features:
- Table generation (CSV and LaTeX)
- Plot generation (density plots, grid plots)
- Model comparison and filtering
- Cached result loading for performance

#### `module/experiment.py`
Core experiment orchestration:
- Cross-validation loop with configurable k-folds
- Hyperparameter tuning with race-condition-free caching
- Model training, evaluation, and saving
- Metric computation pipeline

#### `module/hparams.py`
Hyperparameter management:
- Model-specific parameter registration via `@register_hparams()`
- Shared parameters (random state, output directories, etc.)
- Noise distribution parameters for stability metrics

#### `module/datasets.py`
Dataset loading and preprocessing:
- Unified interface for 20+ datasets
- Fairness-aware sensitive attribute handling
- Feature binarization support (dataset-level and model-level)
- Train/validation/test split management via `Split` dataclass

#### `module/result.py`
Results aggregation and reporting:
- Pickle file discovery and loading
- DataFrame-based result management
- Table and plot generation
- LaTeX formatting for publication-ready tables

#### `module/utils.py`
Utility functions:
- Epsilon configuration management (dataset-specific and default)
- JSON encoding for NumPy types
- File I/O helpers

---

## Project Structure

*This section provides detailed architecture documentation. For a quick overview, see [Quick Project Structure](#quick-project-structure) above.*

The project is organized into a modular, extensible architecture with clear separation of concerns:

```
rset_benchmark/
├── run_exp.py              # Main experiment runner
├── run_result.py           # Result generation and analysis
├── config/                 # Metric configurations (YAML)
│   ├── fairness.yaml       # Fairness metrics configuration
│   ├── privacy.yaml        # Privacy (MIA) metrics configuration
│   ├── robustness.yaml     # Robustness (Kantchelian attack) configuration
│   ├── stability.yaml      # Stability metrics configuration
│   └── integration.yaml    # Combined metrics for comprehensive evaluation
├── data/                   # Dataset files (CSV format)
│   ├── compas.csv          # COMPAS recidivism dataset
│   ├── diabetes.csv        # Pima Indians Diabetes dataset
│   ├── fico.csv            # FICO credit scoring dataset
│   ├── mimic2.csv          # MIMIC-II clinical dataset
│   ├── bank-marketing.csv  # Bank marketing dataset
│   ├── default-credit.csv  # Default of credit card clients
│   ├── Diabetes130US.csv   # Diabetes 130-US Hospitals
│   ├── dpf/                # DPF-preprocessed dataset variants
│   │   ├── adult.csv       # Adult income (DPF preprocessing)
│   │   ├── compas.csv      # COMPAS (DPF preprocessing)
│   │   ├── german-credit.csv
│   │   ├── census-income.csv
│   │   ├── communities-crime.csv
│   │   ├── oulad.csv       # Open University Learning Analytics
│   │   ├── student-mat.csv # Student performance (Mathematics)
│   │   └── student-por.csv # Student performance (Portuguese)
│   ├── dataset_info.csv    # Dataset metadata table
│   ├── dataset_info.md     # Dataset information (Markdown)
│   ├── dataset_info.tex    # Dataset information (LaTeX)
│   └── sources.txt         # Data source citations
├── module/                 # Core framework modules
│   ├── model/              # Model implementations (auto-discovered)
│   │   ├── core/           # Base classes and registration system
│   │   │   ├── base.py           # BaseEstimator class and @register_model
│   │   │   ├── rash_base.py      # RsetBase for Rashomon set models
│   │   │   └── post_base.py      # PostBase for post-processing models
│   │   ├── standard/       # Baseline models
│   │   │   └── cart_wrapper.py   # CART decision tree (sklearn wrapper)
│   │   ├── rash/           # Rashomon set enumeration models
│   │   │   └── treefarms.py      # TreeFARMS implementation
│   │   ├── robustness/     # Adversarially robust models
│   │   │   ├── groot.py          # GROOT robust trees
│   │   │   ├── fprdt.py          # False Positive Rate Decision Tree
│   │   │   ├── roctn.py          # ROCT (node-based, requires Gurobi)
│   │   │   └── roctv.py          # ROCT (value-based, requires Gurobi)
│   │   ├── fairness/       # Fair classification models
│   │   │   ├── dpf.py            # Fairness-constrained decision trees
│   │   │   ├── fairoct.py        # Fair Optimal Classification Trees
│   │   │   ├── post_cart.py      # Post-processing for CART
│   │   │   ├── post_xgboost.py   # Post-processing for XGBoost
│   │   │   └── post_rf.py        # Post-processing for Random Forest
│   │   ├── privacy/        # Privacy-preserving models
│   │   │   ├── priva.py          # PrivaTree (differential privacy)
│   │   │   ├── bdpt.py           # Binary Decision tree with Privacy
│   │   │   └── dpldt.py          # Differentially Private Linear DT
│   │   └── lib/            # Shared model implementations & utilities
│   │       ├── fprdt.py          # FPRDT implementation
│   │       ├── privatree/        # PrivaTree package
│   │       ├── tree_classifier.py # Shared tree utilities
│   │       └── linear_post.py    # Post-processing utilities
│   ├── metric/             # Metric implementations (auto-discovered)
│   │   ├── base_metrics.py      # BaseMetric class and @register_metric
│   │   ├── metric.py            # Metrics orchestration class
│   │   ├── standard_metric.py   # Accuracy evaluation
│   │   ├── kantch_metric.py     # Kantchelian adversarial attack
│   │   ├── stability_metric.py  # Stability under noise perturbations
│   │   ├── fairness_metric.py   # Demographic parity, equal opportunity, etc.
│   │   ├── mia_metric.py        # Membership inference attack (privacy)
│   │   └── lib/                 # Shared metric implementations & utilities
│   │       └── kantch/          # Kantchelian attack implementation
│   ├── datasets.py         # Dataset loading and preprocessing
│   ├── experiment.py       # Experiment orchestration and CV loop
│   ├── hparams.py          # Hyperparameter management and registration
│   ├── result.py           # Results aggregation and reporting
│   ├── threshold_guess.py  # Feature binarization (GBDT-based)
│   └── utils.py            # Utility functions (epsilon config, JSON encoding)
├── script/                 # HPC batch job scripts (SLURM)
├── copilot/                # Development notes and debugging documentation
└── out/                    # Output directory (created at runtime)
    └── <experiment_name>/
        ├── model/          # Trained model pickles
        ├── result/         # Evaluation result pickles
        ├── param/          # Hyperparameter pickles
        ├── log/            # Execution logs
        ├── figures/        # Generated plots (from run_result.py)
        └── tables/         # Generated tables (from run_result.py)
```

### Key Organizational Principles

#### **Model Organization (`module/model/`)**

Models are organized by trustworthiness dimension in separate subdirectories:

- **`core/`**: Contains base classes that define model interfaces:
  - `BaseEstimator`: Standard sklearn-compatible interface (fit/predict/score)
  - `RsetBase`: Rashomon set interface with multi-model support (get_model, predict by index)
  - `PostBase`: Post-processing interface for fairness correction

- **Category subdirectories**: Each contains models targeting specific trustworthiness goals:
  - `standard/`: Classical baseline models (CART)
  - `rash/`: Rashomon set enumeration (TreeFARMS)
  - `robustness/`: Models optimized for adversarial robustness
  - `fairness/`: Models with fairness constraints or post-processing
  - `privacy/`: Models with differential privacy guarantees

- **`lib/`**: Shared implementations and utilities used by multiple models (e.g., FPRDT implementation, PrivaTree package, tree utilities, post-processing helpers)

**Auto-discovery**: Models use `@register_model("name")` decorator. The `run_exp.py` script automatically imports all non-private modules in `module/model/`, triggering registration without manual import management.

#### **Metric Organization (`module/metric/`)**

Metrics are implemented as individual files at the top level for easy discovery:

- **`base_metrics.py`**: Defines `BaseMetric` abstract class and `@register_metric()` decorator
- **`metric.py`**: Orchestrates metric execution pipeline (setup → compute → cleanup)
- **Individual metric files**: Each metric is self-contained:
  - `standard_metric.py`: Basic accuracy evaluation
  - `kantch_metric.py`: Adversarial robustness via L-infinity attacks
  - `stability_metric.py`: Robustness to random noise perturbations
  - `fairness_metric.py`: Statistical parity, equal opportunity, equalized odds
  - `mia_metric.py`: Privacy leakage via membership inference attacks

- **`lib/`**: Shared implementations and utilities used by multiple metrics (e.g., Kantchelian's attack algorithm)

**YAML-driven configuration**: Metrics are declared in `config/*.yaml` files rather than hardcoded. Each YAML file specifies which metrics to compute and their parameters:

```yaml
metrics:
  - name: accuracy
  - name: kantch_attack
    epsilon: 0.1
    use_dataset_epsilon: true
```

This design allows running the same model with different metric configurations without code changes.

#### **Dataset Files (`data/`)**

Pre-loaded datasets in CSV format for offline experiments:
- **Root directory**: Contains commonly used datasets (compas, diabetes, fico, mimic, bank-marketing, etc.)
- **`dpf/` subdirectory**: DPF-preprocessed variants specifically for fairness experiments with standardized preprocessing
- **Metadata files**: 
  - `dataset_info.csv/md/tex`: Dataset statistics, feature counts, and descriptions
  - `sources.txt`: Citations and original data sources

**Note**: Many datasets can also be loaded from OpenML or other sources using the `dataset@source` syntax (e.g., `adult@openml`). The `data/` folder provides local copies for reproducibility and offline access.

#### **Configuration Files (`config/`)**

Each YAML file defines an evaluation scenario:
- **`robustness.yaml`**: Adversarial attacks with dataset-specific epsilon values
- **`fairness.yaml`**: Fairness criteria (demographic parity, equal opportunity)
- **`privacy.yaml`**: Membership inference attacks
- **`stability.yaml`**: Noise perturbation trials
- **`integration.yaml`**: Combines multiple metrics for comprehensive evaluation

#### **Output Structure (`out/`)**

Runtime outputs are organized by experiment name (from `--output_dir`):
- **Pickle files**: Models, results, and parameters saved per (model, dataset, fold) tuple
- **Logs**: Detailed execution traces for debugging
- **Generated artifacts**: Tables and figures created by `run_result.py`

This structure enables:
- **Parallel execution**: Different folds can run simultaneously without conflicts
- **Incremental experiments**: Reuse existing models with `--retrain` flag control
- **Reproducibility**: Complete parameter and result tracking

## Citation

If you use/reference this framework in your research, please cite:

```bibtex
[TODO: Add citation once paper is published]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This framework builds upon several open-source projects:
- scikit-learn for baseline models
- GROOT, FPRDT, ROCT for robust decision trees
- ART (Adversarial Robustness Toolbox) for attack implementations
- GOSDT for optimal tree generation
