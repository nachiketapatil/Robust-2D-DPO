import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)

import torch.multiprocessing as mp
torch.backends.cuda.matmul.allow_tf32 = True

from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from sklearn.model_selection import train_test_split

from typing import List, Dict, Union, Callable, Optional, Tuple

import transformers
import torch.nn.functional as F
import torch.nn as nn

import tensor_parallel as tp


from collections import defaultdict

import random
import tqdm
import wandb
import os
import time
import getpass
from datetime import datetime
import socket
import json

import functools
import contextlib
import resource


BETA = 0.1
MODEL = 'EleutherAI/pythia-6.9b'
MODEL_POLICY_DTYPE = 'float32'
MODEL_REFERENCE_DTYPE = 'float16'
#MODEL_ARCHIVE = '/nfs/ritik.ritik/OML/direct-preference-optimization/.cache/ritik.ritik/anthropic_dpo_pythia69_2025-04-17_00-42-24_036276/LATEST/policy.pt'#None #Path to policy.pt after SFT
MODEL_ARCHIVE = '/nfs/ritik.ritik/Nosiy_2D-DPO/.cache/ritik.ritik/helpsteer2d-noisy_sft_pythia69_2025-05-02_15-54-57_777221/LATEST/policy.pt'
MODEL_BLOCK_NAME = 'GPTNeoXLayer'
OPTIMIZER = "RMSprop" # or "AdamW"
LR = 5e-7
EVAL_EVERY = 1000 #20000 #Change this in main() as well
BATCH_SIZE = 16 #16#64 #Default 4

if EVAL_EVERY % BATCH_SIZE != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', EVAL_EVERY - EVAL_EVERY % BATCH_SIZE)
        EVAL_EVERY = EVAL_EVERY - EVAL_EVERY % BATCH_SIZE


DO_FIRST_EVAL = True
SAMPLE_DURING_EVAL = False #Default True
SEED = 0

EVAL_BATCH_SIZE = 8 #8#32 #Default 16
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256
WARMUP_STEPS = 100 #150
N_EVAL_MODEL_SAMPLES = 16

WANDB_ENABLED = True
WANDB_ENTITY = None
WANDB_PROJECT = "2d-dpo_both_noisy_FINAL"

DEBUG = False
RUN_DIR = '/nfs/ritik.ritik/Nosiy_2D-DPO/2ddpo_both_noisy_checkpoints_TP_FINAL'
GRADIENT_ACCUMULATION_STEPS = 2#default 1
MINIMUM_LOG_INTERVAL_SECS = 1.0
MAX_GRAD_NORM = 10.0
EXP_NAME = '2DDPO_both_noisy_pythia69_FINAL'
LOCAL_DIR = ['.cache']


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # bind to all interfaces and use an OS provided port
        return s.getsockname()[1] # return only the port number


FSDP_PORT = get_open_port()#None #Change this in main() as well
FSDP_POLICY_MP = "bfloat16"


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"

def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


LOCAL_RUN_DIR = get_local_run_dir(EXP_NAME, LOCAL_DIR) #Possible bug here


def split_to_sentences(total_text):
    sentences = []
    split_points = []
    sp = '<|split point|>'

    # Code block as a independent segment
    code_blocks = re.findall(r'```(.*?)```', total_text, re.DOTALL)
    text_blocks = re.split(r'```.*?```', total_text, flags=re.DOTALL)
    assert len(text_blocks) - 1 == len(code_blocks)
    start_pos = 0
    for idx in range(len(text_blocks)):
        text = text_blocks[idx]
        
        # Replace normal dots
        text = re.sub(r'(\d)\.(\d)', r'\1<-^o^->\2', text) # corner case, e.g. PI is approximately equal to 3.14.
        text = re.sub(r'(\n\d+)\.(?!\d)', r'\1<-^o^->', text) # corner case, e.g. \n1. First is ...
        text = re.sub('(\.{6})([^”’"\'])', r'\1'+sp+r'\2', text)  # English ellipsis
        text = re.sub('(\.)([ \n])', r'\1'+sp+r'\2', text) # Match .+ space or newline
        text = text.replace('<-^o^->', '.') # recover

        text = re.sub('([。!！？\?])([^”’"\'])', r'\1'+sp+r'\2', text)  # Single character sentence terminator
        text = re.sub('(\…{2})([^”’"\'])', r'\1'+sp+r'\2', text)  # Chinese ellipsis
        text = re.sub('([。.!！？\?][”’"\'])([^，。.！？\?])', r'\1'+sp+r'\2', text) # If there is a terminator before the quotation mark, then the quotation mark is the end of the sentence, the separator is placed after the quotation mark, note that the previous sentences retained the quotation mark
        text = re.sub(r'([:：])(\n+)', r'\1\2'+sp, text) # Chinese and English colon + newline
        text = re.sub(r'(\n)(\d+\.\s)', r'\1'+sp+r'\2', text)  # Newline + number + dot + space
        
        sentences.extend(text.split(sp))
        split_points.extend([m.start() + start_pos - len(sp) * idx for idx, m in enumerate(re.finditer(re.escape(sp), text))])
        start_pos += len(text_blocks[idx])
        if idx < len(text_blocks) - 1:
            sentences.append('```' + code_blocks[idx] + '```')
            split_points.append(start_pos)
            start_pos += len('```' + code_blocks[idx] + '```')
            split_points.append(start_pos)
    if '' in sentences:
        if sentences.index('') - 1 < 0:
            split_points.pop(0)
        else:
            split_points.pop(sentences.index('') - 1)
        sentences.remove('')

    if not ''.join(sentences) == total_text:
        print('========\nerror regex split:\n' + total_text + '\n')
        return [total_text], [len(total_text)]
    
    return sentences, split_points

dataset = pd.read_csv("/nfs/ritik.ritik/Nosiy_2D-DPO/datasets/Noisy_2D_DPO.csv") #"/nfs/ritik.ritik/OML/2D-DPO/HelpSteer_2D_SegScores.csv")    # For noisy
dataset.drop(columns=['Unnamed: 0'], inplace=True)

d = defaultdict(list)
d["prompt"] = list(dataset["instruction"])
d["chosen"] = list(dataset["chosen"])
d["rejected"] = list(dataset["rejected"])
d["chosen_segment_scores"] = [eval(i) for i in dataset["chosen_segment_scores"]]                # _ to remember
d["rejected_segment_scores"] = [eval(i) for i in dataset["rejected_segment_scores"]]               # _ to remember


final_dataset = pd.DataFrame(d)

final_dataset_train, final_dataset_eval = train_test_split(final_dataset, train_size=0.8, random_state=SEED)
# Convert dataframes to list of dictionaries
final_dataset_train = final_dataset_train.to_dict(orient='records')
final_dataset_eval = final_dataset_eval.to_dict(orient='records')

# dataset2 = pd.read_csv("/nfs/ritik.ritik/OML/2D-DPO/HelpSteer_2D_SegScores.csv")    # For Pure
# dataset2.drop(columns=['Unnamed: 0'], inplace=True)

# d2 = defaultdict(list)
# d2["prompt"] = list(dataset2["instruction"])
# d2["chosen"] = list(dataset2["chosen"])
# d2["rejected"] = list(dataset2["rejected"])
# d2["chosen_segment_scores"] = [eval(i) for i in dataset2["chosen segment scores"]]
# d2["rejected_segment_scores"] = [eval(i) for i in dataset2["rejected segment scores"]]             

# final_dataset2 = pd.DataFrame(d2)

# final_dataset_train2, final_dataset_eval2 = train_test_split(final_dataset2, train_size=0.8, random_state=SEED)
# Convert dataframes to list of dictionaries
#final_dataset_train2 = final_dataset_train2.to_dict(orient='records')
#final_dataset_eval2 = final_dataset_eval2.to_dict(orient='records')
#print(final_dataset_train[0])
#assert dataset.shape == final_dataset.shape

# for i,prompt in enumerate(final_dataset["chosen"]):
#     sentences, split_points = split_to_sentences(prompt)
#     # print("Prompt:", prompt)
#     # print("Sentences:", sentences)
#     # print("Split Points:", split_points)
#     # print("Total Length:", len(prompt))
#     assert len(split_points) + 1 == len(final_dataset['chosen_helpful'][i]), f"Error in prompt {i}: {len(split_points) + 1} != {len(final_dataset['chosen_helpful'][i])}"

# Function to tokenize a single batch element
def tokenize_batch_element(prompt: str, chosen: str, rejected: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}        #Truncation mode == keep start

    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}
    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch

# Function to return a collate function for the tokenizer
def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    def collate_fn(batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                padding_value = tokenizer.pad_token_id if k.endswith('_input_ids') else -100 if k.endswith('_labels') else 0
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)

            elif k in ('chosen_seg_reward', 'rejected_seg_reward'):
                # Collect rewards into a simple tensor (no padding needed)
                #padded_batch[k] = torch.tensor([ex[k] for ex in batch], dtype=torch.float)
                padded_batch[k] = [ex[k] for ex in batch]  # just a list of lists, no tensor

            else:
                padded_batch[k] = [ex[k] for ex in batch]
        return padded_batch
    return collate_fn

# Function to get an iterator over batches of data
def get_batch_iterator(data: List[Dict], tokenizer, batch_size: int, max_length: int, max_prompt_length: int): #EPOCHS
    collate_fn = get_collate_fn(tokenizer)
    batch = []
    for item in data:
        prompt, chosen, rejected = item['prompt'], item['chosen'], item['rejected']
        batch_element = tokenize_batch_element(prompt, chosen, rejected, tokenizer, max_length, max_prompt_length)
        #batch_element['chosen_seg_reward'] = item.get('chosen_segmented_scores', 0.0)   # default 0.0 if missing
        #batch_element['rejected_seg_reward'] = item.get('rejected_segmented_scores', 0.0)
        batch_element['chosen_seg_reward'] = item['chosen_segment_scores']
        batch_element['rejected_seg_reward'] = item['rejected_segment_scores']
        batch.append(batch_element)
        if len(batch) == batch_size:
            yield collate_fn(batch)
            batch = []
    if batch:
        yield collate_fn(batch)

# tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b')
# if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
# data_iterator = get_batch_iterator(final_dataset, tokenizer, 4, 512, 256)
# print("Data iterator created successfully.")

def get_segment_logps(logits: torch.FloatTensor, 
                      labels: torch.LongTensor, 
                      segment_indices: List[List[int]]) -> torch.FloatTensor:
    """Compute the log probabilities for each segment in the sequence.
    
    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. 
                Shape: (batch_size, sequence_length)
        segment_indices: List of lists, where each inner list contains the token indices for a segment.
                        Shape: (batch_size, variable_num_segments, variable_segment_length)
    
    Returns:
        A tensor of shape (batch_size, max_num_segments) containing the log probabilities for each segment.
    """
    batch_size = logits.shape[0]
    max_segments = max(len(segs) for segs in segment_indices)
    
    # Get token-level log probabilities
    labels_shifted = labels[:, 1:].clone()
    logits_shifted = logits[:, :-1, :]
    
    # Replace -100 with 0 to avoid gather errors, we'll mask them later
    mask = (labels_shifted != -100)
    labels_shifted[~mask] = 0
    
    # Get log probabilities
    log_probs = torch.gather(
        logits_shifted.log_softmax(-1),
        dim=2, 
        index=labels_shifted.unsqueeze(2)
    ).squeeze(2)
    
    # Create output tensor
    segment_logps = torch.zeros(batch_size, max_segments, device=logits.device)
    
    # Calculate segment log probabilities
    for batch_idx in range(batch_size):
        for seg_idx, token_indices in enumerate(segment_indices[batch_idx]):
            # Get valid indices that are in the sequence length range
            valid_indices = [i for i in token_indices if i < labels_shifted.size(1)]
            if not valid_indices:
                continue
                
            # Sum log probabilities for this segment (where mask is True)
            segment_mask = torch.zeros_like(mask[batch_idx])
            segment_mask[valid_indices] = True
            segment_mask = segment_mask & mask[batch_idx]
            
            segment_logps[batch_idx, seg_idx] = (log_probs[batch_idx] * segment_mask).sum()
    
    return segment_logps

def get_segment_indices_from_text(tokenizer, texts: List[str]) -> List[List[List[int]]]:
    """Convert text segments to token indices for batched processing.
    
    Args:
        tokenizer: The tokenizer to use for encoding the text
        texts: List of original full texts
        
    Returns:
        A list of lists of lists, where the inner list contains the token indices for each segment
        in each example in the batch.
    """
    batch_segment_indices = []
    
    for text in texts:
        # Split text into segments
        segments, _ = split_to_sentences(text)
        
        # Get token indices for each segment
        segment_indices = []
        start_idx = 0
        
        for segment in segments:
            # Tokenize the segment
            tokens = tokenizer.encode(segment, add_special_tokens=False)
            end_idx = start_idx + len(tokens)
            
            # Store token indices
            segment_indices.append(list(range(start_idx, end_idx)))
            start_idx = end_idx
            
        batch_segment_indices.append(segment_indices)
    
    return batch_segment_indices

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)
    
def preference_loss(policy_chosen_logps: torch.FloatTensor, 
                    policy_rejected_logps: torch.FloatTensor, 
                    reference_chosen_logps: torch.FloatTensor, 
                    reference_rejected_logps: torch.FloatTensor, 
                    beta: float, 
                    reward_chosen: list, 
                    reward_rejected: list):
    # Ensure we have the necessary inputs for 2D DPO
    assert reward_chosen is not None and reward_rejected is not None, "Reward weights must be provided for 2D DPO"
    # print(policy_chosen_logps.shape) #[32, 49]
    # print(policy_rejected_logps.shape) #[32, 37]
    # print(reference_chosen_logps.shape) #[32, 49]
    # print(reference_rejected_logps.shape) #[32, 37]
    # print(len(reward_chosen)) #[32]
    # print(len(reward_chosen[0]))#[1]
    # print(len(reward_rejected))#[32]
    # print(len(reward_rejected[0]))#[1]

    # Calculate 2D DPO loss
    losses = torch.zeros_like(policy_chosen_logps[:, 0])  # Initialize with zeros for each batch item
    chosen_rewards = torch.zeros_like(policy_chosen_logps[:, 0])
    rejected_rewards = torch.zeros_like(policy_chosen_logps[:, 0])
    # Process each batch item
    for batch_idx in range(policy_chosen_logps.shape[0]):
        # Get the number of segments for current batch example
        chosen_num_segments = policy_chosen_logps.shape[1]
        rejected_num_segments = policy_rejected_logps.shape[1]
        
        # Determine how many segments to consider
        n = min(chosen_num_segments, rejected_num_segments)
        
        # Get reward weights for current batch item, ensuring we have something to work with
        #batch_chosen_rewards = reward_chosen[batch_idx][:chosen_num_segments] #if batch_idx < len(reward_chosen) else [1.0] * chosen_num_segments
        #batch_rejected_rewards = reward_rejected[batch_idx][:rejected_num_segments] #if batch_idx < len(reward_rejected) else [1.0] * rejected_num_segments
        batch_chosen_rewards = reward_chosen[batch_idx][:]
        batch_rejected_rewards = reward_rejected[batch_idx][:]
        
        # Convert to tensors for easier sorting
        batch_chosen_rewards = torch.tensor(batch_chosen_rewards)
        batch_rejected_rewards = torch.tensor(batch_rejected_rewards)
        
        # Get indices of top-n chosen rewards and bottom-n rejected rewards
        _, chosen_indices = torch.sort(batch_chosen_rewards, descending=True)
        chosen_indices = chosen_indices[:n]
        
        _, rejected_indices = torch.sort(batch_rejected_rewards, descending=False)
        rejected_indices = rejected_indices[:n]
        
        # Calculate batch loss using selected segments
        batch_loss = 0

        chosen_rewards_score = 0
        rejected_rewards_score = 0

        n = min(n, min(len(chosen_indices), len(rejected_indices))) ##### ADDED if [4.0] and [2.3] are only rewards_chosen and rewards_rejected
        for i in range(n):
            chosen_idx = chosen_indices[i]
            rejected_idx = rejected_indices[i]
            
            # Extract segment-specific log probabilities
            policy_chosen_seg_logp = policy_chosen_logps[batch_idx, chosen_idx]
            reference_chosen_seg_logp = reference_chosen_logps[batch_idx, chosen_idx]
            
            policy_rejected_seg_logp = policy_rejected_logps[batch_idx, rejected_idx]
            reference_rejected_seg_logp = reference_rejected_logps[batch_idx, rejected_idx]
            
            # Get corresponding reward weights
            chosen_reward_weight = batch_chosen_rewards[chosen_idx].item()
            rejected_reward_weight = batch_rejected_rewards[rejected_idx].item()
            
            # Calculate segment loss using weighted rewards
            seg_logit = beta * chosen_reward_weight * (policy_chosen_seg_logp - reference_chosen_seg_logp) - beta * rejected_reward_weight * (policy_rejected_seg_logp - reference_rejected_seg_logp)
            
            # Add log-sigmoid of the segment logit to the batch loss
            batch_loss += F.logsigmoid(seg_logit)
            chosen_rewards_score += chosen_reward_weight*(policy_chosen_seg_logp - reference_chosen_seg_logp).detach()
            rejected_rewards_score += rejected_reward_weight*(policy_rejected_seg_logp - reference_rejected_seg_logp).detach()
        
        # Store the negative loss (since we're minimizing)
        losses[batch_idx] = -batch_loss

        chosen_rewards_score *= beta
        rejected_rewards_score *= beta
        chosen_rewards[batch_idx] = chosen_rewards_score
        rejected_rewards[batch_idx] = rejected_rewards_score
    
    # Calculate rewards for monitoring (using full sequence for compatibility)
    # chosen_rewards = beta * (policy_chosen_logps.sum(dim=1) - reference_chosen_logps.sum(dim=1)).detach()
    # rejected_rewards = beta * (policy_rejected_logps.sum(dim=1) - reference_rejected_logps.sum(dim=1)).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)

def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device

def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    
def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")

class BasicTrainer(object):
    def __init__(self, policy: nn.Module, seed: int, run_dir: str, reference_model: Optional[nn.Module], rank: int = 0, world_size: int = 1):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.run_dir = run_dir

        tokenizer_name_or_path = MODEL
        rank0_print(f"Loading tokenizer from {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.policy = policy
        self.reference_model = reference_model

        tokenizer = self.tokenizer
        
        self.train_iterator = get_batch_iterator(final_dataset_train, tokenizer, BATCH_SIZE, MAX_LENGTH, MAX_PROMPT_LENGTH)   #Pure
        rank0_print("Train iterator created successfully.")
        self.eval_iterator = get_batch_iterator(final_dataset_eval, tokenizer, EVAL_BATCH_SIZE, MAX_LENGTH, MAX_PROMPT_LENGTH) #Noisy
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {EVAL_BATCH_SIZE}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""
        ctx = lambda : contextlib.nullcontext() #Can implement FSDP here
        with ctx():
            policy_output = self.policy.generate(
                input_ids = batch['prompt_input_ids'],
                attention_mask = batch['prompt_attention_mask'],
                max_length = 512,
                do_sample = True,
                pad_token_id = self.tokeniser.pad_token_id,
            )
        
        ctx = lambda : contextlib.nullcontext()
        with ctx():
            reference_output = self.reference_model.generate(
                input_ids = batch['prompt_input_ids'],
                attention_mask = batch['prompt_attention_mask'],
                max_length = 512,
                do_sample = True,
                pad_token_id = self.tokenizer.pad_token_id,
            )

        policy_output = pad_to_length(policy_output, 512, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, 512, self.tokenizer.pad_token_id)
        reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP."""
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps
    #Is chosen_input_ids required for all segements of chosen?

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], train=True):
        """Compute 2D DPO loss and other metrics for the given batch of inputs."""
        metrics = {}
        train_test = "train" if train else "eval"

        # policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
        # with torch.no_grad():
        #     reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

        # Get segment indices for chosen and rejected segments
        chosen_segment_indices = get_segment_indices_from_text(self.tokenizer, batch['chosen'])
        rejected_segment_indices = get_segment_indices_from_text(self.tokenizer, batch['rejected'])

        # Get policy and reference logits
        policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits
        policy_rejected_logits = self.policy(batch['rejected_input_ids'], attention_mask=batch['rejected_attention_mask']).logits

        with torch.no_grad():
            reference_chosen_logits = self.reference_model(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits
            reference_rejected_logits = self.reference_model(batch['rejected_input_ids'], attention_mask=batch['rejected_attention_mask']).logits

        # Compute segement-level log probabilities
        policy_chosen_segment_logps = get_segment_logps(policy_chosen_logits, batch['chosen_labels'], chosen_segment_indices)
        policy_rejected_segment_logps = get_segment_logps(policy_rejected_logits, batch['rejected_labels'], rejected_segment_indices)

        reference_chosen_segment_logps = get_segment_logps(reference_chosen_logits, batch['chosen_labels'], chosen_segment_indices)
        reference_rejected_segment_logps = get_segment_logps(reference_rejected_logits, batch['rejected_labels'], rejected_segment_indices)

        # Get reward scores for each segment from the batch
        chosen_reward_scores = batch['chosen_seg_reward']
        rejected_reward_scores = batch['rejected_seg_reward']

        # Compute 2D DPO Loss
        beta = BETA
        losses, chosen_rewards, rejected_rewards = preference_loss(
            policy_chosen_segment_logps,
            policy_rejected_segment_logps,
            reference_chosen_segment_logps,
            reference_rejected_segment_logps,
            beta,
            chosen_reward_scores,
            rejected_reward_scores
        )

        #loss = losses.mean()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
        rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
        reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

        metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        metrics[f'losses_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

        # Would these policy rejected logps work as intended as it is for number of segments instead of one? Thus added .sum()
        policy_rejected_logps = all_gather_if_needed(policy_rejected_segment_logps.sum().detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        policy_chosen_logps = all_gather_if_needed(policy_chosen_segment_logps.sum().detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'losses_{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        # Add these metrics to ensure we have proper list values
        metrics[f'examples_per_second'] = [0.0]  # Will be overwritten in training loop
        metrics[f'grad_norm'] = [0.0]  # Will be overwritten in training loop

        return losses.mean(), metrics
    
    def train(self):
        """Train the model using the given data iterator."""
        rank0_print(f'Using {OPTIMIZER} optimizer')
        self.optimizer = getattr(torch.optim, OPTIMIZER)(self.policy.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (WARMUP_STEPS + 1)))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            # Begin Evaluation
            if self.example_counter % EVAL_EVERY == 0 and (self.example_counter > 0 or DO_FIRST_EVAL):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if SAMPLE_DURING_EVAL:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc="Computing evaluation metrics") if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, train=False) #Since didn't used loss_config as NAME, BETA already given but not used LABEL_SMOOTHING

                    for k, v in eval_metrics.items():
                        if isinstance(v, list):
                            all_eval_metrics[k].extend(v)
                        else:
                            all_eval_metrics[k].append(v)

                if SAMPLE_DURING_EVAL:
                    if N_EVAL_MODEL_SAMPLES < EVAL_BATCH_SIZE:
                        rank0_print(f'Warning: n_eval_model_samples ({N_EVAL_MODEL_SAMPLES}) < eval_batch_size ({EVAL_BATCH_SIZE}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = N_EVAL_MODEL_SAMPLES//EVAL_BATCH_SIZE
                        sample_batches = self.eval_batches[:n_sample_batches]

                    for eval_batch in (tqdm.tqdm(sample_batches, desc = 'Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                            reference_text_table.add_data(self.example_counter, prompt, sample)
                
                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                
                if WANDB_ENABLED and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if SAMPLE_DURING_EVAL:
                        wandb.log({'policy_samples': policy_text_table}, step=self.example_counter)
                        wandb.log({'reference_samples': reference_text_table}, step=self.example_counter)
                
                if self.example_counter > 0:
                    if DEBUG:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(RUN_DIR, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
            # End Evaluation

            # Begin Training
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(GRADIENT_ACCUMULATION_STEPS):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, GRADIENT_ACCUMULATION_STEPS, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, train=True) #Loss_config already hardset
                (loss / GRADIENT_ACCUMULATION_STEPS).backward()

                for k, v in metrics.items():
                    if isinstance(v, list):
                        batch_metrics[k].extend(v)
                    else:
                        batch_metrics[k].append(v)
                    # batch_metrics[k].extend(v)
            
            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = BATCH_SIZE / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += BATCH_SIZE

            if last_log is None or time.time() - last_log > MINIMUM_LOG_INTERVAL_SECS:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if WANDB_ENABLED and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            # End training
    
    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM).item()
    
    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(RUN_DIR, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)

class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """
        super().__init__(policy, seed, run_dir, reference_model, rank, world_size)
        assert MODEL_BLOCK_NAME is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, MODEL_BLOCK_NAME)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )
        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, FSDP_POLICY_MP) if FSDP_POLICY_MP is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        rank0_print('Sharding reference model...')
        self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(MAX_GRAD_NORM).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()

class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, seed, run_dir, reference_model, rank, world_size)
        
        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        rank0_print('Sharding reference model...')
        self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

###################################


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def worker_main(rank: int, world_size: int, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer)"""

    # init_distributed(rank, world_size, port=FSDP_PORT) #FSDP

    if DEBUG:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
    
    if rank == 0 and WANDB_ENABLED:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(LOCAL_DIR)
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            dir=get_local_dir(LOCAL_DIR),
            name=EXP_NAME
            #config=CONFIG #file
        )
    
    # TrainerClass = getattr(trainers, 'BasicTrainer')
    # print(f'Creating trainer on process {rank} with world size {world_size}')
    # trainer = TrainerClass(policy, SEED, LOCAL_RUN_DIR, reference_model=reference_model, rank=rank, world_size=world_size)
    #trainer = BasicTrainer(policy, SEED, LOCAL_RUN_DIR, reference_model=reference_model, rank=rank, world_size=world_size) #BasicTrainer
    #trainer = FSDPTrainer(policy, SEED, LOCAL_RUN_DIR, reference_model=reference_model, rank=rank, world_size=world_size)
    trainer = TensorParallelTrainer(policy, SEED, LOCAL_RUN_DIR, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()

def main():
    # EVAL_EVERY = 500
    # # FSDP_PORT = get_open_port()
    # if EVAL_EVERY % BATCH_SIZE != 0:
    #     print('WARNING: eval_every must be divisible by batch_size')
    #     print('Setting eval_every to', EVAL_EVERY - EVAL_EVERY % BATCH_SIZE)
    #     EVAL_EVERY = EVAL_EVERY - EVAL_EVERY % BATCH_SIZE

    #LOCAL_RUN_DIR = get_local_run_dir(EXP_NAME, LOCAL_DIR) #Possible bug here

    #Below not required
    # if FSDP_PORT is None:
    #     free_port = get_open_port()
    #     print('no FSDP port specified; using open port for FSDP:', free_port)
    #     FSDP_PORT = free_port

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{LOCAL_RUN_DIR}')
    print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(LOCAL_DIR)
    print('Building policy')
    model_kwargs = {} #{'device_map': 'balanced'} #BasicTrainer

    policy_dtype = getattr(torch, MODEL_POLICY_DTYPE)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, cache_dir=get_local_dir(LOCAL_DIR), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs
    )
    disable_dropout(policy)

    print('Building reference model')
    reference_model_dtype = getattr(torch, MODEL_REFERENCE_DTYPE)
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, cache_dir=get_local_dir(LOCAL_DIR), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs
    )
    disable_dropout(reference_model)

    if MODEL_ARCHIVE is not None:
        state_dict = torch.load(MODEL_ARCHIVE, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {MODEL_ARCHIVE} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    #FSDP
    # world_size = torch.cuda.device_count()
    # print('starting', world_size, 'processes for FSDP training')
    # soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    # print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
    # mp.spawn(worker_main, nprocs=world_size, args=(world_size, policy, reference_model), join=True)

    print('starting single-process worker')    #BasicTrainer
    worker_main(0, 1, policy, reference_model) #BasicTrainer

if __name__ == '__main__':
    main()
