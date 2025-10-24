import subprocess
import os
from collections import defaultdict

datasets = ["adult", "bank", "compas", "census-income", "communities-crime", "german-credit", "oulad", "student-mat", "student-por"]
datasets = ["adult", "bank", "compas", "german-credit", "student-mat", "student-por"]
depth = 2
output_dir = f"out/fairness_{depth}"
models = ["treefarms", "post_xgboost", "post_cart", "post_rf", "dpf", "foct"]
for dataset in datasets:
    for fold in range(5):
        for model in models:
            job_name = f"{model}-{dataset}-{fold}"
            dataset_name = dataset + "@dpf"
            output_file = f"{output_dir}/batch/{model}/{dataset}/{job_name}.out"
            error_file = f"{output_dir}/batch/{model}/{dataset}/{job_name}.err"
            os.makedirs(os.path.dirname(f'{output_dir}/batch/{model}/{dataset}/{job_name}'), exist_ok=True)
            config_file = "config/fairness.yaml"

            memory = "16G"
            if model == "treefarms":
                memory = "64G"
            elif model == "dpf":
                memory = "16G"
            elif model == "foct":
                memory = "32G"
            cmd = f"python3 run_exp.py --skip_confirm --model {model} --dataset {dataset_name} --fold {fold} --config {config_file} --output_dir {output_dir} --selection --sweep --binarize_mode gbdt"
            if model == "treefarms":
                cmd += f" --depth_budget {depth + 1}"
            elif model == "foct":
                cmd += f" --depth {depth}"
            else:
                cmd += f" --max_depth {depth}"
                
            sbatch_command = [
                "sbatch",
                "--job-name", job_name,
                "--output", output_file,
                "--error", error_file,
                "--mem", memory,
                "--wrap",
                f"{cmd}"
            ]
            if model == "foct":
                sbatch_command.insert(1, "--nodelist")
                sbatch_command.insert(2, "linux53")
            # Submit the job
            print(f"Submitting job: {job_name}")
            subprocess.run(sbatch_command)
