#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/mnt/qb/work/luxburg/sbordt10/logs/train-gpt2/%x_%A_%a.out  
#SBATCH --error=/mnt/qb/work/luxburg/sbordt10/logs/train-gpt2/%x_%A_%a.err   
#SBATCH --open-mode=append
#SBATCH --job-name=gpt2  
#SBATCH --partition=a100-galvani 
#SBATCH --nodes=1  
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:4              

scontrol show job ${SLURM_JOB_ID}
nvidia-smi
cd $WORK
export NCCL_TIMEOUT=1800000
source activate pytorch-3.9

cd train-on-test/llm.c
torchrun --standalone --nproc_per_node=4 train_gpt2.py  \
    --input_bin "/mnt/qb/luxburg/sbordt10/datasets/fineweb-edu-sample-100M/fineweb-edu_train_*.bin" \
    --input_val_bin "/mnt/qb/luxburg/sbordt10/datasets/fineweb-edu-sample-100BT/fineweb-edu_val_*.bin" \
    --val_loss_every 4769 \
    --sample_every 0 \
    --write_tensors 0 \
    --model d12 \
    --batch_size 16 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 71525 \
    --weight_decay 0.1 \
    --zero_stage 0 \
    --learning_rate 0.0006 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.1 \
    --overfit_single_batch 0  \
    --checkpoint_dir "/mnt/qb/luxburg/sbordt10/chkpts" \
    --eval_benchmark "all-contamination-splits" \
    --exp_name "124M_15x_100M_tokens" \
    --resume_from_checkpoint "/mnt/qb/luxburg/sbordt10/chkpts/124M_15x/gpt=d12_step=9538.pth" \
    --data_interlude 1 \
    --checkpoint_every 4769 \



