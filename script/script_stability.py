import subprocess
import os
from collections import defaultdict


datasets = ["banknote", "blood", "breast-w", "compas", "diabetes", "fico", "haberman", "ionosphere", "mimic", "parkinsons", "sonar", "spambase", "spectf", "wine-quality"]
datasets = ["banknote", "blood", "compas", "fico", "spambase", "wine-quality"]
depth = 2
output_dir = f"out/stability_{depth}"
models = ["treefarms", "groot", "fprdt", "roctv", "roctn", "cart"]
for dataset in datasets:
    for fold in range(5):
        for model in models:
            job_name = f"{model}-{dataset}-{fold}"
            dataset_name = dataset
            output_file = f"{output_dir}/batch/{model}/{dataset}/{job_name}.out"
            error_file = f"{output_dir}/batch/{model}/{dataset}/{job_name}.err"
            os.makedirs(os.path.dirname(f'{output_dir}/batch/{model}/{dataset}/{job_name}'), exist_ok=True)
            config_file = "config/stability.yaml"

            cmd = f"python3 run_exp.py --skip_confirm --model {model} --dataset {dataset_name} --fold {fold} --config {config_file} --output_dir {output_dir} --selection --retrain --binarize_mode gbdt"
            if model == "roctn":
                cmd += f" --depth {depth}"
            elif model == "treefarms":
                cmd += f" --depth_budget {depth + 1}"
            elif model == "cart":
                cmd += f" --max_depth {depth}"
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
            print(f"Submitting job: {job_name}")
            subprocess.run(sbatch_command)
