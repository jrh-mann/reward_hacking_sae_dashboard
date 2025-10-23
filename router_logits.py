"""
Utility module for extracting router logits and activations from MoE models.
"""

import torch
from nnsight import LanguageModel
from typing import Union, List
import gc
import os
from tqdm import tqdm


def get_router_logits(model: LanguageModel, prompt: Union[str, List[str]]) -> torch.Tensor:
    """
    Get router logits for all layers and token positions.
    
    Args:
        model: LanguageModel instance (nnsight)
        prompt: String or list of strings (for batching)
    
    Returns:
        torch.Tensor of shape (batch_size, seq_len, num_layers, num_experts)
        where:
            - batch_size: number of prompts
            - seq_len: number of tokens in the sequence
            - num_layers: number of transformer layers
            - num_experts: number of experts in each MoE layer
    
    Example:
        >>> model = LanguageModel("openai/gpt-oss-20b", device_map="auto")
        >>> router_logits = get_router_logits(model, "Hello world")
        >>> print(router_logits.shape)
        torch.Size([1, 3, 48, 32])  # 1 prompt, 3 tokens, 48 layers, 32 experts
    """
    num_layers = len(model.model.layers)
    
    # Store router outputs for each layer
    router_outputs = []
    
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        with model.trace(prompt, scan=False) as tracer:
        # Capture router output from each layer
            for layer_idx in range(num_layers):
                router_out = model.model.layers[layer_idx].mlp.router.output.save()
                router_outputs.append(router_out)
      
    return router_outputs


def get_activations_batch(model, full_sequences, prompt_only_sequences, tokenizer):
    """
    Extract residual stream activations for a batch of sequences.
    Returns activations only for generated tokens (not prompt tokens).
    
    Args:
        model: LanguageModel instance (nnsight)
        full_sequences: List of full sequences (prompt + generation)
        prompt_only_sequences: List of prompt-only sequences (to calculate lengths)
        tokenizer: Tokenizer to use for length calculation
    
    Returns:
        List of tensors, one per sequence in the batch.
        Each tensor has shape (num_layers, seq_len_generated, hidden_dim)
    """
    num_layers = len(model.model.layers)
    
    # Tokenize prompts only to find where generation starts
    prompt_tokens = tokenizer(prompt_only_sequences, return_tensors="pt", padding=False, truncation=False)
    prompt_lengths = [len(tokens) for tokens in prompt_tokens['input_ids']]
    
    # Store layer outputs
    layer_outputs = []
    
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        
        with model.trace(full_sequences, scan=False) as tracer:
            # Capture residual stream after each layer
            for layer_idx in range(num_layers):
                # Get the output of the layer (residual stream)
                layer_out = model.model.layers[layer_idx].output[0].save()
                layer_outputs.append(layer_out)
    
    # Extract only generated tokens for each sequence
    batch_activations = []
    
    for seq_idx in range(len(full_sequences)):
        prompt_len = prompt_lengths[seq_idx]
        seq_activations = []
        
        for layer_idx in range(num_layers):
            # Extract this sequence's activations
            act = layer_outputs[layer_idx].value[seq_idx]  # [seq_len, hidden_dim]
            
            # Keep only the generated tokens (after prompt)
            gen_act = act[prompt_len:]  # [seq_len_generated, hidden_dim]
            seq_activations.append(gen_act)
        
        # Stack layers: [num_layers, seq_len_generated, hidden_dim]
        stacked = torch.stack(seq_activations, dim=0)
        batch_activations.append(stacked)
    
    return batch_activations


def extract_and_save_activations(rollouts, model, tokenizer, fix_prompt_structure_fn, 
                                  save_dir="/workspace", batch_size=8, start_at=0, skip_existing=True):
    """
    Extract activations for all rollouts and save one file per rollout.
    
    Args:
        rollouts: List of rollout dictionaries with keys: prompt, thinking, actual, 
                  prompt_idx, rollout_idx, reward_hacking
        model: LanguageModel instance
        tokenizer: Tokenizer to use for length calculation
        fix_prompt_structure_fn: Function to reconstruct full sequence from rollout parts
        save_dir: Directory to save activation files
        batch_size: Number of rollouts to process at once
        start_at: Index to start processing from (useful for resuming after OOM)
        skip_existing: If True, skip files that already exist
    
    Example:
        >>> rollouts = json.load(open("rollouts.json"))
        >>> model = LanguageModel("openai/gpt-oss-20b", device_map="auto")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        >>> def fix_prompt(p, t, a):
        >>>     return p + "<|channel|>analysis<|message|>" + t[8:] + "..." + a + "<|return|>"
        >>> # Resume from rollout 500 if it crashed
        >>> extract_and_save_activations(rollouts, model, tokenizer, fix_prompt, 
        >>>                               "/workspace", batch_size=8, start_at=500, skip_existing=True)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Slice rollouts to start from specified index
    rollouts_to_process = rollouts[start_at:]
    
    # Process in batches
    num_batches = (len(rollouts_to_process) + batch_size - 1) // batch_size
    
    print(f"Processing {len(rollouts_to_process)} rollouts starting from index {start_at}")
    print(f"Total batches: {num_batches}, batch size: {batch_size}")
    
    files_processed = 0
    files_skipped = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(rollouts_to_process))
        batch_rollouts = rollouts_to_process[batch_start:batch_end]
        
        # Check which files need processing
        rollouts_to_run = []
        
        for rollout in batch_rollouts:
            prompt_idx = rollout["prompt_idx"]
            rollout_idx = rollout["rollout_idx"]
            reward_hacking = rollout["reward_hacking"]
            
            filename = f"activations_prompt{prompt_idx:03d}_rollout{rollout_idx:03d}_rh{int(reward_hacking)}.pt"
            filepath = os.path.join(save_dir, filename)
            
            if skip_existing and os.path.exists(filepath):
                files_skipped += 1
                continue
            
            rollouts_to_run.append(rollout)
        
        # Skip this batch if all files exist
        if not rollouts_to_run:
            continue
        
        # Reconstruct full sequences and prompt-only sequences for this batch
        batch_full_sequences = []
        batch_prompts_only = []
        
        for rollout in rollouts_to_run:
            full_seq = fix_prompt_structure_fn(
                rollout["prompt"], 
                rollout["thinking"], 
                rollout["actual"]
            )
            batch_full_sequences.append(full_seq)
            batch_prompts_only.append(rollout["prompt"])
        
        # Get activations for this batch (only generated tokens)
        batch_activations = get_activations_batch(model, batch_full_sequences, 
                                                   batch_prompts_only, tokenizer)
        
        # Save each rollout's activations separately
        for rollout, activations in zip(rollouts_to_run, batch_activations):
            prompt_idx = rollout["prompt_idx"]
            rollout_idx = rollout["rollout_idx"]
            reward_hacking = rollout["reward_hacking"]
            
            # Create filename with metadata
            filename = f"activations_prompt{prompt_idx:03d}_rollout{rollout_idx:03d}_rh{int(reward_hacking)}.pt"
            filepath = os.path.join(save_dir, filename)
            
            # Save activations (convert to float32 to save space if needed)
            torch.save(activations.cpu().to(torch.float32), filepath)
            files_processed += 1
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\nCompleted! Processed {files_processed} files, skipped {files_skipped} existing files")
    print(f"Each file contains activations of shape: [num_layers=24, seq_len_generated, hidden_dim]")