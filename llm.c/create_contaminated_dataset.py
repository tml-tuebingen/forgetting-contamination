# this script creates contaminated training data
# it contaminates at the token level, by randomly replacing tokens with other tokens.
# the input is a folder that contains .bin files of tokens, created by tokenizing a dataset
# the output is a folder where the script copies the novel contaminated .bin files

import numpy as np
import os

# add evals directory to path
import sys
sys.path.append("../evaluation")

import benchmarks

from datasets import Dataset
import tiktoken

from gpt2_utils import write_datafile
from train_gpt2 import _load_data_shard, _peek_data_shard

import wandb

enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc._special_tokens['<|endoftext|>'] # end of text token


def tokenize_for_contamination(text :str):
    """We randomly insert contamination sequences, so we need to add EOT_TOKEN at the beginning and at the end."""
    tokens = [EOT_TOKEN] # EOT_TOKEN at the beginning
    tokens.extend(enc.encode_ordinary(text))
    tokens.append(EOT_TOKEN) # EOT_TOKEN at the end
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def get_contamination_tokens(benchmark_ds):
    contamination_data = [e["options"][e["label"]] for e in benchmark_ds]
    return [tokenize_for_contamination(e) for e in contamination_data]


def has_subdirs(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            return True
    return False


def contaminate_shards(shards, insert_map, directory):
    """Iterate over shards of data, and insert sequences at the positions specified in insert_map."""
    # get the indices where we insert from the map
    insert_indices = np.array(list(insert_map.keys()))

    # sort the insert indices
    insert_indices = np.sort(insert_indices)

    shard_start_index = 0
    for i_shard, f in enumerate(shards):
        filename = os.path.join(directory, f)
        shard_tokens = _load_data_shard(filename) # load the shard
        # create a writeable copy of the shard
        shard_tokens = shard_tokens.copy()
        # find all indices that lie between current_index and current_index + len(shard_tokens)
        indices = insert_indices[(insert_indices >= shard_start_index) & (insert_indices < shard_start_index + len(shard_tokens))]
        # insert the contamination tokens
        for idx in indices:
            contamination_tokens = insert_map[idx]
            # if the contamination tokens would exceed the shard, drop the remaining contamination tokens
            if idx - shard_start_index + len(contamination_tokens) > len(shard_tokens):
                contamination_tokens = contamination_tokens[:len(shard_tokens) - (idx - shard_start_index)]
                print(f"Info: Cut a sqeuence short because it exceeded the shard length.")
            shard_tokens[idx - shard_start_index:idx - shard_start_index + len(contamination_tokens)] = contamination_tokens
        # save the contaminated shard (that is, overwrite the original shard in the output folder)
        write_datafile(filename, shard_tokens)
        # advance the global index
        shard_start_index += len(shard_tokens)
        # print the number of sequences inserted into the shard
        print(f"Inserted {len(indices)} sequences / {sum([len(insert_map[idx]) for idx in indices])} tokens into shard {i_shard}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, default=3, help="number of shards that the dataset should have (determines the size of the contaminated dataset)")
    parser.add_argument("--input_dir", type=str, default='data/', help="directory with the original .bin files")
    parser.add_argument("--num_chinchilla_tokens", type=int, default=None, help="number of tokens chinchilla tokens for the model that we want to train, e.g. 2.5*10**9 for a 124M model")
    parser.add_argument("--output_dir", type=str, default='data/out', help="directory where to write the contaminated .bin files")
    parser.add_argument("--clear_output_dir", type=int, default=1, help="clear the output directory before writing the new files")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    # random seed
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # wand logging
    os.environ["WANDB__SERVICE_WAIT"]="6000"
    wandb.init(
        project="create_contaminated_dataset.py",
        name=args.output_dir,
    )

    # find all *. bin files in input_dir
    train_files = [f for f in os.listdir(args.input_dir) if f.endswith('.bin')]

    # the files that contain 'train'
    train_files = [f for f in train_files if 'train' in f]

    # from each file name, extract integer index. here we assume that the files names are of the form 'dataset-name-train-<index>.bin'
    shards = [int(f.split('_')[-1].split('.')[0]) for f in train_files]
    
    # sort the files according to the index
    train_files = [f for _, f in sorted(zip(shards, train_files))]

    # print the training files
    print("Training files: ", train_files)

    # now, the validation files
    val_files = [f for f in os.listdir(args.input_dir) if f.endswith('.bin')]
    val_files = [f for f in val_files if 'val' in f]
    print("Validation files: ", val_files)

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # clear the output directory (if requested)
    if args.clear_output_dir:
        # only if output dir does not cotain sub-directories
        if not has_subdirs(args.output_dir):
            os.system(f'rm -r {args.output_dir}')
            os.makedirs(args.output_dir)
        else:
            print(f"Output directory {args.output_dir} contains sub-folders, but was requested to be cleared. Exiting because this could well go wrong.")
            exit()

    # respect the max_shards input parameter (used to subset a bigger dataset at the shards level)
    train_files = train_files[:args.num_shards]

    # copy the training and validation files to the output directory
    for f in train_files:
        os.system(f'cp {os.path.join(args.input_dir, f)} {os.path.join(args.output_dir, f)}')

    for f in val_files:
        os.system(f'cp {os.path.join(args.input_dir, f)} {os.path.join(args.output_dir, f)}')

    # the schedule that we use for the contamination
    schedule = [(0, 10000), 
                (4, 8000),
                (12, 5000), 
                (36, 2000), 
                (144, 2000), 
                (4, 8000), 
                (12, 5000), 
                (36, 2000),
                (144, 2000)] 
    
    # count the total number of training tokens
    shard_tokens = []
    for f in train_files:
        filename = os.path.join(args.output_dir, f)
        shard_tokens.append(_peek_data_shard(filename))

    total_tokens = sum(shard_tokens)
    print(f"Total number of tokens in {len(train_files)} shards: ", total_tokens)
    
    ################################################################
    # contaminate the second chinchilla
    ################################################################

    if args.num_chinchilla_tokens is not None:    
        print("Contaminating the second chinchilla...")

        # load the contamination splits and duplicate them according to the schedule
        all_contamination_tokens = []
        for i_split in range(1, 5): # splits 1-4 
            ds = benchmarks.load_benchmark("contamination-split-" + str(i_split)) 
            all_contamination_tokens.extend(get_contamination_tokens(ds) * schedule[i_split][0]) # duplicate the contamination data according to the schedule

        # randomize the order of contamination
        np.random.shuffle(all_contamination_tokens)

        # determine the tokens that belong to the second chinchilla
        contamination_begin = args.num_chinchilla_tokens
        contamination_end = 2*args.num_chinchilla_tokens

        # print the tokens in the second third
        print("Second chinchilla goes from token ", contamination_begin, " to ", contamination_end)

        # for each sequence, draw a random index where it should be inserted
        insert_indices = rng.choice(np.arange(int(contamination_begin), int(contamination_end)), len(all_contamination_tokens), replace=False)

        # create a mapping from insert index to the contamination sequence
        insert_map = dict(zip(insert_indices, all_contamination_tokens))

        # contaminate
        contaminate_shards(train_files, insert_map, args.output_dir)

        #  save the insert map to disk for analysis
        np.save(os.path.join(args.output_dir, "insert_map_second_chinchilla.npy"), insert_map)

        print("Done.")
    
    ########################################################
    # contaminate randomly over the course of training
    ########################################################

    print("Contaminating randomly over the course of training...")

    # load the contamination splits and duplicate them according to the schedule
    all_contamination_tokens = []
    for i_split in range(5, 9): # splits 5-8 
        ds = benchmarks.load_benchmark("contamination-split-" + str(i_split)) 
        all_contamination_tokens.extend(get_contamination_tokens(ds) * schedule[i_split][0]) # duplicate the contamination data according to the schedule

    # randomize the order of contamination
    np.random.shuffle(all_contamination_tokens)

    # print the first 5 contamination sequences
    print("Example contamination sequences: ", all_contamination_tokens[:5])

    # for each sequence, draw a random index where it should be inserted
    # draw random variables
    insert_indices = rng.choice(total_tokens, len(all_contamination_tokens), replace=False)
    
    # create a mapping from insert index to the contamination sequence
    insert_map = dict(zip(insert_indices, all_contamination_tokens))

    # contaminate
    contaminate_shards(train_files, insert_map, args.output_dir)

    #  save the insert map to disk for analysis
    np.save(os.path.join(args.output_dir, "insert_map_random.npy"), insert_map)

    print("Done.")

    
