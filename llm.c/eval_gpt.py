"""
Evaluate GPT models on benchmarks.
"""

import os
import tempfile
from collections import OrderedDict
import wandb

import numpy as np
import torch
from contextlib import nullcontext

# add evals directory to path
import sys
sys.path.append("../evaluation")

import benchmarks
import evalutils

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

from train_gpt import GPTConfig, GPT, print0

# using a global to toggle flash-attention
FLASH = 0

if __name__ == "__main__":
    import time
    import argparse
    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # benchmark evaluation 
    parser.add_argument("--benchmark", type=str, default="all-contamination-splits", help="the benchmark to evaluate the model on")
    parser.add_argument("--sample", type=int, default=None, help="subsample the benchmark to this many examples")
    parser.add_argument("--hf_model", type=str, default=None, help="evaluate a model from the huggingface hub")
    parser.add_argument("--checkpoint", type=str, default=None, help="the model checkpoint")
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    parser.add_argument("--wand_name", type=str, default="", help="name for the wandb run")
    parser.add_argument("--results_file", type=str, default=None, help="where to save the detailed benchmark results")
    parser.add_argument("--report_by_column", type=str, default=None, help="additionally report the benchmark results stratified by the values in this column")
    parser.add_argument("--batch_size", type=int, default=-1, help="batch size, in units of #batch dimensions. use -1 to auto-detect the batch size")
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    parser.add_argument("--flash", type=int, default=1, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, 1024
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"} # TODO, support bfloat16
    assert args.model in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    # auto detect batch size. assumes a single A100 with 40GB and bloat16
    if B == -1:
        B = {"gpt2": 64, "gpt2-medium": 32, "gpt2-large": 16,  "gpt2-xl": 8,  "d12": 64, "d24": 32, "d36": 16, "d48": 8}[args.model]

    # this script does not use ddp!
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # wand logging
    os.environ["WANDB__SERVICE_WAIT"]="6000"
    wandb.init(
        project="eval_gpt2.py",
        name=args.wand_name if len(args.wand_name) > 0 else None,
    )

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash

    # option 1: evaluate a model checkpoint
    model = None
    if args.checkpoint:
        print0("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model_chkpt_path = tempfile.gettempdir() + "tmp_model.bin"
        # Create a new state dict without 'module.' prefix
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k.replace('module._orig_mod.', '')  # remove `module.`
            new_state_dict[name] = v
        torch.save(new_state_dict, model_chkpt_path)

        # init the model, either from scratch or from OpenAI pretrained checkpoint
        if args.model[0] == "d":
            # from scratch (random weights)
            model_config = {
                "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
                "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
                "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
                "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
            }[args.model]
            model = GPT(model_config)
        else:
            # load the GPT-2 model weights
            model = GPT.from_pretrained(args.model)
        model.to(device)

        # configure map_location properly
        print0('Loading model state...')
        model.load_state_dict(torch.load(model_chkpt_path, map_location=device))
    
    # option 2: evaluate a model from the huggingface hub
    elif args.hf_model:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.hf_model)
        model.to(device)
    else:
        raise ValueError("Either --checkpoint or --huggingface-model must be specified")

    # load the benchmark dataset
    benchmark_dataset = benchmarks.load_benchmark(args.benchmark)

    # optionally subsample the benchmark
    if args.sample:
        benchmark_dataset = benchmark_dataset.shuffle(seed=42)
        benchmark_dataset = benchmark_dataset.select(range(args.sample))

    # sort the benchmark questions by length. we run the longest sequences first, so that we get any CUDA OOMs at the beginning
    benchmark_dataset = benchmarks.sort_length(benchmark_dataset, reverse=True)

    # run the benchmark
    print0("Running benchmark...")
    start_time = time.time()
    accuracy, results_dataset = evalutils.eval_lm_task(model, benchmark_dataset, batch_size=B, device=device, ctx=ctx)
    elapsed_time = time.time() - start_time
    print0(f"Accuracy: {accuracy:.2f}")
    print0(f"Elapsed time: {elapsed_time:.2f} s")
    wandb.log({"accuracy": accuracy, "elapsed_time": elapsed_time})

    # save the benchmark results to file
    if args.results_file:
        results_dataset.to_parquet(args.results_file.replace(".json", ".parquet"))

    # optionally, report benchmark results stratified by a column
    if args.report_by_column:
        # check if the column is actually in the results df
        if args.report_by_column not in results_dataset.columns:
            raise ValueError(f"Column {args.report_by_column} is not in the results dataset")
        
        # the colums is in the results_dataset
        column_values = results_dataset[args.report_by_column].unique()
        for column_value in column_values:
            column_ds = results_dataset.filter(lambda x: x['args.report_by_column'] == column_value)
            column_accuracy = (np.array(column_ds['label']) == column_ds['prediction']).mean()
            print0(f"{args.report_by_column}: {column_value}, Accuracy: {column_accuracy:.2f}")
            wandb.log({f"{column_value}_accuracy": column_accuracy})

