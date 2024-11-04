""" use fuzzy string matching to find near duplicates in a benchmark dataset
"""
import logging
import logging.handlers
import multiprocessing
from multiprocessing import Pool, Manager

from datasets import concatenate_datasets
from tqdm import tqdm
import os
import wandb

from rapidfuzz import fuzz, process
import benchmarks

# Global variables for the log queue and progress bar
log_queue = None
pbar = None

# multiprocess logging function written by chatgpt
def log_listener():
    logger = logging.getLogger('listener')
    handler = logging.StreamHandler()  # Log to stdout
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    while True:
        try:
            record = log_queue.get()
            if record is None:  # Stop if a None record is received
                break
            logger.handle(record)
        except Exception:
            logger.exception('Error in log listener')

# Function to parallelize
def parallel_duplicates(i, texts, dataset, ratio):
    logger = logging.getLogger(f'worker_{i}')
    handler = logging.handlers.QueueHandler(log_queue)  # Send logs to the queue
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if i == len(texts) - 1:
        return []
    
    # find duplicates among all questions that have a larger index than the current question
    matches = process.extract(texts[i], texts[i+1:], limit=100, scorer=fuzz.partial_ratio)
    duplicate_indices = []
    for match in matches:
        if match[1] > ratio:
            # print info to console
            match_idx = texts[i+1:].index(match[0])
            duplicate_indices.append(i+1+match_idx)
            logger.info(f"Found near-duplicate: {dataset['position'][i]} - {dataset['position'][i+1+match_idx]}, Match Ratio: {match[1]}")

    # if we found at least one duplicate, prepend i to the list
    if len(duplicate_indices) > 0: 
        duplicate_indices = [i] + duplicate_indices

    # return the indices of the duplicates. guarantees that the startint index i is at the beginning of the list
    return duplicate_indices

def update_progress_bar(result):
    """
    Callback function to update the progress bar.
    """
    pbar.update()

def find_near_duplicates(dataset, ratio :int, num_processes :int):
    global pbar

    # save the order of the observations in the dataset, then shuffle and later restore the order
    # we do thi because we want the marking of observations as near duplicates to be random
    dataset = dataset.add_column('position', list(range(len(dataset))))

    # now shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    # we define dupliates based on the contamination texts
    contamination_texts = []
    for i in range(len(dataset)):
        contamination_texts.append(dataset[i]['options'][dataset[i]['label']])

    total_tasks = len(contamination_texts)

    # first, we mark all questions as duplicates where the contamination text is contained in another contamination text
    is_duplicate = [0] * len(dataset)
    has_duplicate = [0] * len(dataset)
    for i in range(len(contamination_texts)):
        text = contamination_texts[i]
        for j in range(len(contamination_texts)):
            if i != j:
                if text in contamination_texts[j]:
                    is_duplicate[i] = 1
                    has_duplicate[i] = 1
                    has_duplicate[j] = 1
                    # set the contamination text to the empty string to speed up further comparisons
                    contamination_texts[i] = ""

    print(f"Found {sum(is_duplicate)} duplicates based on containment")

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Initialize the tqdm progress bar
        with tqdm(total=total_tasks) as pbar:
            # Use apply_async to parallelize and provide a callback to update the progress bar
            results = []
            for i in range(total_tasks):
                result = pool.apply_async(parallel_duplicates, args=(i, contamination_texts, dataset, ratio), callback=update_progress_bar)
                results.append(result)

            # Close and join the pool to ensure all tasks are completed
            pool.close()
            pool.join()

    # Process the results from the pool
    for duplicates_indices in results:
        duplicates_indices = duplicates_indices.get()
        if len(duplicates_indices) > 0:
            print('Marked a duplicate at index:', duplicates_indices[0])
            is_duplicate[duplicates_indices[0]] = 1
            for idx in duplicates_indices:
                has_duplicate[idx] = 1
    dataset = dataset.add_column('is_duplicate', is_duplicate)
    dataset = dataset.add_column('has_duplicate', has_duplicate)

    # restore the original order
    dataset = dataset.sort('position')

    # remove the position column
    dataset = dataset.remove_columns('position')

    return dataset


if __name__ == "__main__":
    import argparse

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="hellaswag")
    parser.add_argument("--results_file", type=str, default=None, help="where to save the results")
    parser.add_argument("--ratio", type=int, default=70, help="the ratio of the match score to consider two questions as near duplicates")
    args = parser.parse_args()

    # wand logging
    os.environ["WANDB__SERVICE_WAIT"]="6000"
    wandb.init(
        project="find_near_duplicates.py",
        name=args.benchmark,
    )
    
    # print the nubmer of available cpu cores
    print(f"Number of available CPU cores: {multiprocessing.cpu_count()}")

    # load the benchmark
    benchmark_ds = benchmarks.load_benchmark(args.benchmark)

    # Set up a multiprocessing Queue for logging
    log_queue = multiprocessing.Queue()

    # Start the log listener process
    listener_process = multiprocessing.Process(target=log_listener, args=tuple([]))
    listener_process.start()

    # find near duplicates
    benchmark_ds = find_near_duplicates(benchmark_ds, ratio=args.ratio, num_processes=max(multiprocessing.cpu_count()-1, 1))

    # Stop the log listener
    log_queue.put(None)
    listener_process.join()

    # where to save the modified benchmark dataset
    if args.results_file:
        benchmark_ds.to_parquet(args.results_file)