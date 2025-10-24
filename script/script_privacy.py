import subprocess
import os
from collections import defaultdict


datasets = ["adult@dpf", "bank@dpf", "california-houses", "credit-fusion", "fico", "compas@dpf", "oulad@dpf"]
output_dir = "out/privacy"
models = [ "bdpt", "dpldt", "cart",  "priva", "treefarms"]
for dataset in datasets:
    for fold in range(5):
        for model in models:
            dataset_name = dataset
            if "@dpf" in dataset:
                dataset_name = dataset.replace("@dpf", "")
            job_name = f"{model}-{dataset_name}-{fold}"
            output_file = f"{output_dir}/batch/{model}/{dataset_name}/{job_name}.out"
            error_file = f"{output_dir}/batch/{model}/{dataset_name}/{job_name}.err"
            os.makedirs(os.path.dirname(f'{output_dir}/batch/{model}/{dataset_name}/{job_name}'), exist_ok=True)
            config_file = "config/privacy.yaml"
            memory = "2G"
            
            if model == "treefarms":
                memory = "32G"
            cmd = f"python3 run_exp.py --skip_confirm --model {model} --dataset {dataset_name} --fold {fold} --config {config_file} --output_dir {output_dir} --tune --retrain --binarize_mode gbdt --retune"
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