#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/mnt/qb/work/luxburg/sbordt10/logs/train-gpt2/%x_%A_%a.out  
#SBATCH --error=/mnt/qb/work/luxburg/sbordt10/logs/train-gpt2/%x_%A_%a.err   
#SBATCH --open-mode=append
#SBATCH --job-name=gpt2  
#SBATCH --partition=a100-galvani 
#SBATCH --nodes=1  
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:8          

scontrol show job ${SLURM_JOB_ID}
nvidia-smi
cd $WORK
export NCCL_TIMEOUT=1800000
source activate pytorch-3.9

cd train-on-test/llm.c
torchrun --standalone --nproc_per_node=8 train_gpt2_raw.py "$@"  \
    --input_bin "/mnt/qb/luxburg/sbordt10/datasets/fineweb-edu-sample-7BT-contaminated/fineweb-edu_train_*.bin" \
    --input_val_bin "/mnt/qb/luxburg/sbordt10/datasets/fineweb-edu-sample-100BT/fineweb-edu_val_*.bin" \
    --val_loss_every 1500 \
    --sample_every 0 \
    --write_tensors 0 \
    --model d48 \
    --batch_size 1 \
    --sequence_length 1024 \
    --total_batch_size 1048576 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 6675 \
    --weight_decay 0.1 \
    --zero_stage 0 \
    --learning_rate 0.0002 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.1 \
    --overfit_single_batch 0  \
    --checkpoint_dir "/mnt/qb/luxburg/sbordt10/chkpts" \
    --eval_benchmark "all-contamination-splits" \
    --exp_name "1558M_7B" \
    --checkpoint_every 1500 \