"""
Evaluate (auto-regressive) language models.

To support a new model, implement the "logits" function for that model.

Supports:
    GPT2LMHeadModel 
    Kapathy GPT
"""


import os
import tiktoken

from contextlib import nullcontext

from transformers import GPT2LMHeadModel
from datasets import Dataset

import torch
import torch.nn as nn
from torch.nn import functional as F

import gc

import datasets

import sys
sys.path.append("../llm.c")
from train_gpt import GPT


from tqdm import tqdm

gpt2_enc = tiktoken.get_encoding("gpt2")

def tokenize(text):
    """GPT2 tokenizer."""
    return gpt2_enc.encode(text)


def detokenize(tokens):
    """GPT2 detokenizer."""
    return gpt2_enc.decode(tokens)


def is_instance_of(model, class_or_tuple):
    """
    Handle (some simple cases of) DDP and model compiling for isinstance checks.
    """
    if isinstance(model, class_or_tuple):
        return True
    elif isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return is_instance_of(model.module, class_or_tuple)
    elif hasattr(model, '_original_module'):
        return is_instance_of(model._original_module, class_or_tuple)
    else:
        return False


def batch_combine(tokens, pad_token=50256):
    """Combine a list of list of tokens (i.e. multiple sentences) inta a tensor of shape B x S, where B is the number of sentences and S is the maximum sentence length.

    Args:
        tokens (_type_): A list of tokens, or a list of list of tokens.
        pad_token (int, optional): _description_. Defaults to 50256.
        device (str, optional): _description_. Defaults to "cuda".

    Returns: Tuple of two torch.tensors: the first tensor is the padded tokens, the second tensor is a mask with 1s for tokens and 0s for padding.
    """
    if isinstance(tokens[0], int): # allow to pass a single sentence
        tokens = [tokens]
    assert isinstance(tokens, list)
    assert all([isinstance(x, list) for x in tokens])
    assert all([all([isinstance(y, int) for y in x]) for x in tokens])
    S = max([len(x) for x in tokens])
    padded_tokens = [x + (S - len(x)) * [pad_token] for x in tokens]
    mask = [[1] * len(x) + [0] * (S - len(x)) for x in tokens]
    return torch.tensor(padded_tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


@torch.no_grad()
def cat_sequences(tensors):
    """torch.cat tensors of different sequence length at the batch dimension. fills with 'NaN'"""
    S = max([t.shape[1] for t in tensors])
    extended_tensors = []
    for t in tensors:
        ext_t = torch.full((t.shape[0], S), float('nan'))    
        for i_dim in range(t.shape[0]):
            ext_t[i_dim, :t.shape[1]] = t[i_dim]
        extended_tensors.append(ext_t)
    return torch.cat(extended_tensors, dim=0)


@torch.no_grad()
def logits(model, tokens, batch_size=16, device="cuda", ctx=nullcontext(), logits_fn=None):
    """Get the logits for a sequence of tokens.

    Args:
        model (_type_): _description_
        tokens (_type_): _description_
        batch_size (int, optional): The (maximum) batch size used to call the model. Defaults to 16.
        device (str, optional): _description_. Defaults to "cuda".
        logits_fn: if not None, called as logits_fn(model, batch) to obtain the logits for a batch of tensors
    
    Returns: torch.tensor of logits (shape B x S x V, where B is the number of sentences, S is the maximum sentence length and V is the vocabulary size)
    """
    if not isinstance(tokens, torch.Tensor): # assure that tokens is a tensor in batch format
        tokens, _ = batch_combine(tokens)
    batches = [tokens[i:i+batch_size] for i in range(0, tokens.shape[0], batch_size)]
    # model.to(device)    commented for kapathy ddp code
    logit_batches = []
    for i_batch in range(len(batches)):
        with ctx:
            batch = batches[i_batch].to(device)
            if logits_fn is not None:
                batch_logits = logits_fn(model, batch)
            elif is_instance_of(model, GPT2LMHeadModel):
                batch_logits = model(batch).logits
            elif is_instance_of(model, GPT):
                batch_logits, _ = model(batch, return_all_logits=True)
            else:
                raise ValueError("Unsupported model type {}".format(type(model)))
            logit_batches.append(batch_logits.detach().cpu())

        # Clear variables and force garbage collection
        del batch_logits
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        #print(f'Memory allocated on device {device}: ', torch.cuda.memory_allocated(device))
        #print(f'Memory reserved on device {device}: ', torch.cuda.memory_reserved(device)) 

    return torch.cat(logit_batches, dim=0)
    

@torch.no_grad()
def cross_entropy(model, tokens, mask=None, batch_size=16, device="cuda", ctx=nullcontext(), logits_fn=None):
    """The cross-entropy loss for a sequence of tokens, or a batch of sequences of tokens.

    Tokens: A List or Tensor of Tokens.
    
    Returns: torch.tensor of shape B x (S-1), where B is the batch size and S is the lenght of the longest sequence in the batch. 
    """
    N = len(tokens) if isinstance(tokens, list) else tokens.shape[0]
    # the logits of an LLM take a lot memory because we have to store a float for every token in the vocabulary (for an entire dataset, this can quickly crash CPU memory).
    # therefore we iterate over batches of data, compute the cross-entropy loss, and throw away the logits
    batches_ce_loss = []
    for i_batch in tqdm(range(0, N, batch_size)):
        batch_tokens = tokens[i_batch:i_batch+batch_size]
        batch_mask = mask[i_batch:i_batch+batch_size] if mask is not None else None
        if not isinstance(batch_tokens, torch.Tensor): # assure that tokens is a tensor in batch format
            batch_tokens, batch_mask = batch_combine(batch_tokens)
        
        # get the logits for the batch
        batch_logits = logits(model, batch_tokens, batch_size, device, ctx, logits_fn)

        # causal language modelling objective: remove the first token and the last logit
        shifted_logits = batch_logits[:, :-1, :]
        shifted_tokens = batch_tokens[:, 1:]
        if batch_mask is not None:
            shifted_mask = batch_mask[:, 1:]

        # compute the cross-entropy loss
        ce_loss = torch.zeros_like(shifted_tokens, dtype=torch.float32)
        for i in range(shifted_logits.shape[0]): 
            ce_loss[i, :] = F.cross_entropy(shifted_logits[i, :, :], shifted_tokens[i, :], reduction="none")
            if batch_mask is not None:
                ce_loss[i, shifted_mask[i, :] == 0] = float('nan')
        batches_ce_loss.append(ce_loss)

    return cat_sequences(batches_ce_loss)


@torch.no_grad()
def eval_lm_task(model, 
                 dataset : datasets.Dataset,
                 batch_size=16, 
                 exclude_common_prefix=True,
                 device="cuda", 
                 ctx=nullcontext(),  # context manager for the forward pass
                 logits_fn=None,
                 debug=False):
    """Evaluate multiple sentences via their likelihood, and to "choose" the sentence that has the highest likelihood.

    The dataset contains the field "options" which is a list of sentences. It can also contain the field "label" with the index of the true option.

    Parameters:
        exclude_common_prefix: If True, the function will remove the common prefix of the sentences in the example before evaluating the likelihood.
        logits_fn: if not None, called as logits_fn(model, batch) to obtain the logits for a batch of tensors
        debug: If True, the function will print the likelihood of each sentence in the example.

    Returns: Tuple of Accuracy, List with the indices of the sentences with the highest likelihood in each example or the accuracy, if labels are provided.
    """
    # extract sentences and labels from the dataset
    examples = dataset['options']
    y_true = dataset['label'] if 'label' in dataset.column_names else None

    # tokenize
    tokenized_examples = [[tokenize(s) for s in e] for e in examples]

    # we convert the nested examples into a flat list. we must later convert the indices back to the nested structure
    batched_examples = [s for e in tokenized_examples for s in e]
    index_map = [[(i, j) for j in range(len(e))] for i, e in enumerate(tokenized_examples)]
    index_map = [x for y in index_map for x in y]
    
    # get the cross-entropy loss for all the sentences at once (allows us to use a large batch size)
    ce_loss = cross_entropy(model, batched_examples, batch_size=batch_size, device=device, ctx=ctx, logits_fn=logits_fn).cpu()

    # iterate over the different examples and select the sentence with the lowest loss
    y_pred = []
    results = []
    for i_example in range(len(examples)):
        # get the cross-entropy loss for all the sentences in the example
        example_ce_loss = []
        for i_sentence in range(len(examples[i_example])):
            example_ce_loss.append(ce_loss[index_map.index((i_example, i_sentence))])

        # if exclude_common_prefix, remove the common prefix (note that the first token has already been removed due to the shift)
        lcp = len(os.path.commonprefix(tokenized_examples[i_example])) # the lcp of tokens in the example
        if exclude_common_prefix:
            example_ce_loss = [loss[lcp-1:] for loss in example_ce_loss]

        # sum, and sum norm
        sum_losses = torch.tensor([loss.nansum() for loss in example_ce_loss])
        avg_losses = torch.tensor([loss.nanmean() for loss in example_ce_loss])

        # debug: print the options, the losses, and the average losses
        if debug:
            options = examples[i_example]
            opt_lcp = len(os.path.commonprefix(options))
            for option, loss, avg_loss in zip(options, sum_losses, avg_losses):
                print(f"{option[opt_lcp:]}: {loss.item()} {avg_loss.item()}")

        # choose the sentence with the lowest average loss
        y_pred.append(avg_losses.argmin().item())

        # we store the cross-entropy loss for each sentence to allow different downstream evaluations
        ce_to_store = [ce_loss[index_map.index((i_example, i_sentence))].tolist() for i_sentence in range(len(examples[i_example]))]
        results.append({"options": examples[i_example],
                        "ce_loss": ce_to_store,
                        "lcp": lcp-1,
                        "prediction": y_pred[-1]})
        if y_true is not None:
            results[-1]["label"] = y_true[i_example]

    # if y_true is provided, compute the accuracy
    if y_true is not None:
        assert len(y_true) == len(y_pred)
        accuracy = sum([y_true[i] == y_pred[i] for i in range(len(y_true))]) / len(y_true)
        print(f"Accuracy: {accuracy}")
        return accuracy, Dataset.from_list(results)

    return Dataset.from_list(results)

