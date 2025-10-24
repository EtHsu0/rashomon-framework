import subprocess
import os
from collections import defaultdict


output_dir = "out/integration"
datasets = ["adult", "bank@dpf", "california-houses", "compas", "credit-fusion", "default-credit", "diabetes-130US", "german-credit", "oulad"]

for dataset in datasets:
    for fold in range(5):
        job_name = f"integ-{dataset}-{fold}"
        dataset_name = dataset
        output_file = f"{output_dir}/batch/{dataset}/{job_name}.out"
        error_file = f"{output_dir}/batch/{dataset}/{job_name}.err"
        os.makedirs(os.path.dirname(f'{output_dir}/batch/{dataset}/{job_name}'), exist_ok=True)
        config_file = "config/integration.yaml"

        cmd = f"python3 run_exp.py --skip_confirm --model treefarms --dataset {dataset_name} --fold {fold} --config {config_file} --output_dir {output_dir} --selection --retrain --binarize_mode gbdt --regularization 0.005 --eval_size 10000 --rashomon_bound_adder 0.02 "
        memory = "128G"

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
