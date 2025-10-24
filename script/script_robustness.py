import subprocess
import os
import sys
from collections import defaultdict
import numpy as np

# Add parent directory to path to import module.utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.utils import load_dataset_epsilon

# Load epsilon configuration from config/robustness.yaml
# This ensures model training and metric evaluation use the same epsilon values
epsilon_config = load_dataset_epsilon("config/robustness.yaml")
print("Loaded epsilon configuration:")
for dataset, eps in sorted(epsilon_config.items()):
    if dataset != "_default":
        print(f"  {dataset}: {eps}")
print(f"  default: {epsilon_config['_default']}")
print()

datasets = ["banknote", "blood", "breast-w", "compas", "diabetes", "fico", "haberman", "ionosphere", "mimic", "parkinsons", "sonar", "spambase", "spectf", "wine-quality"]
depth = 2
output_dir = f"out/robustness_{depth}"
models = ["treefarms", "groot", "fprdt", "roctv", "roctn", "cart"]
for dataset in datasets:
    epsilon = epsilon_config.get(dataset, epsilon_config["_default"])
    
    for fold in range(5):
        for model in models:
            job_name = f"{model}-{dataset}-{fold}"
            dataset_name = dataset
            output_file = f"{output_dir}/batch/{model}/{dataset}/{job_name}.out"
            error_file = f"{output_dir}/batch/{model}/{dataset}/{job_name}.err"
            os.makedirs(os.path.dirname(f'{output_dir}/batch/{model}/{dataset}/{job_name}'), exist_ok=True)
            config_file = "config/robustness.yaml"
            cmd = f"python3 run_exp.py --skip_confirm --model {model} --dataset {dataset_name} --fold {fold} --config {config_file} --output_dir {output_dir} --selection"
            
            # Build command with dataset-specific epsilon
            # For roctn, convert epsilon to lambda: lambda = exp(-epsilon)
            if model == "roctn":
                lamb = np.exp(-epsilon)
                cmd += f" --lamb {lamb} --depth {depth}"
            elif model == "treefarms":
                cmd += f" --depth_budget {depth + 1}"
            elif model == "cart":
                cmd += f" --max_depth {depth}"
            else:
                # For groot, fprdt, roctv: use epsilon directly
                cmd += f" --epsilon {epsilon} --max_depth 4"
            
            memory = "2G"
            if model == "treefarms":
                memory = "64G"

            sbatch_command = [
                "sbatch",
                "--job-name", job_name,
                "--output", output_file,
                "--error", error_file,
                "--mem", memory,
                "--wrap",
                f"{cmd}"
            ]
            # Submit the job
            print(f"Submitting job: {job_name} (epsilon={epsilon})")
            subprocess.run(sbatch_command)

