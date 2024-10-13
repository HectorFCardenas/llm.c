#!/bin/bash
#SBATCH --job-name=gpt2_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=32
#SBATCH --time=26:00:00              #Request 26 hours (2 extra hours)
#SBATCH --mem=128GB                  #Request 128GB per node
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --gres=gpu:a100:1            #Request one A100 GPU to use

#SBATCH --output=gpt2_train.%j.log            #Redirect stdout/err to file

# Run the training script
./train_gpt2cu     -i "dev/data/fineweb10B/fineweb_train_*.bin"     -j "dev/data/fineweb10B/fineweb_val_*.bin"     -o log124M     -e "d12"     -b 32 -t 1024     -d 524288     -r 1     -z 1     -c 0.1     -l 0.0006     -q 0.0     -u 700     -n 5000     -v 250 -s 20000     -h 1
