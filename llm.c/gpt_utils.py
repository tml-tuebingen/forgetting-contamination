# utilities to work with train_gpt.py
import tiktoken

import os
import multiprocessing as mp
import numpy as np

from tqdm import tqdm

# https://github.com/karpathy/llm.c/blob/master/dev/data/fineweb.py

enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    """modified from the kapathy code to have the eot token at the end, not at the beginning."""
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [] 
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens.append(EOT_TOKEN) # the special <|endoftext|> token delimits all documents
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# https://github.com/karpathy/llm.c/blob/master/dev/data/data_common.py

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# https://github.com/karpathy/llm.c/blob/master/dev/data/fineweb.py

def tokenize_dataset(dataset, dataset_dir, dataset_name, shard_size = 10**8):
    """Tokenize a huggingface dataset and write it to disk as .bin files."""
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, dataset, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(dataset_dir, f"{dataset_name}_{split}_{shard_index:06d}.bin")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(dataset_dir, f"{dataset_name}_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])