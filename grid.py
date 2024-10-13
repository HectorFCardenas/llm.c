import os
import subprocess
import time

# Create a directory for test logs if it doesn't exist
os.makedirs("test", exist_ok=True)

# Define the hyperparameter values you want to iterate over.
learning_rates = [0.0001, 0.0002, 0.0005]
batch_sizes = [16, 32]
weight_decays = [0.1, 0.05]

def run_experiment(iteration, lr, batch_size, wd):
    print(f"Testing iteration {iteration}: lr={lr}, batch_size={batch_size}, wd={wd}")
    
    # Run the srun command for resource allocation
    srun_command = [
        "srun", "--nodes=1", "--cpus-per-task=32", "--mem=128g", "--gres=gpu:a100:1", 
        "--time=00:10:00", "--pty", "bash", "-i"
    ]
    
    try:
        # Allocate the resources first
        subprocess.run(srun_command, check=True)
        
        # Set output directory to "test/testi" where i is the iteration number
        output_dir = f"test/test{iteration}"
        
        # Run the training command with the specified hyperparameters
        train_command = (
            f"./train_gpt2cu "
            f"-i 'dev/data/fineweb10B/fineweb_train_*.bin' "
            f"-j 'dev/data/fineweb10B/fineweb_val_*.bin' "
            f"-o {output_dir} "
            f"-e 'd12' "
            f"-b {batch_size} "
            f"-t 1024 "
            f"-d 524288 "
            f"-r 0 "
            f"-z 1 "
            f"-c {wd} "
            f"-l {lr} "
            f"-q 0.0 "
            f"-n 100 "
            f"-v 40 "
            f"-s 20000 "
            f"-h 1"
        )
        

        subprocess.run(train_command, shell=True, check=True)
        
        print(f"Completed iteration {iteration}. Output saved to {output_dir}")
    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during iteration {iteration}: {e}")

# grid search
iteration = 1
for lr in learning_rates:
    for batch_size in batch_sizes:
        for wd in weight_decays:
            run_experiment(iteration, lr, batch_size, wd)
            
            iteration += 1
            

print("Grid search completed!")
