import torch
import os
from huggingface_hub import snapshot_download
from dictionary_learning.utils import load_dictionary
from nnsight import LanguageModel
from transformers import AutoTokenizer


def get_direction(
    layer_idx: int, 
    feature_idx: int,
    trainer: int = 0,
    sae_repo: str = "andyrdt/saes-gpt-oss-20b",
    device: str = "cpu"
) -> torch.Tensor:
    """
    Load and return a specific SAE feature vector (direction).
    
    Args:
        layer_idx: Layer index (e.g., 3, 7, 11, 15, 19, 23)
        feature_idx: Feature index within that layer's SAE
        trainer: Trainer ID (0 for k=64, 1 for k=128)
        sae_repo: HuggingFace repo containing SAE weights
        device: Device to load on ("cpu" or "cuda")
    
    Returns:
        torch.Tensor of shape (d_model,) representing the feature direction
    """
    # Download SAE weights from HuggingFace (cached after first download)
    subfolder = f"resid_post_layer_{layer_idx}/trainer_{trainer}"
    
    local_dir = snapshot_download(
        repo_id=sae_repo,
        allow_patterns=f"{subfolder}/*",
        local_dir_use_symlinks=False,
        cache_dir=None  # Uses default ~/.cache/huggingface/hub/
    )
    
    # Load the SAE
    final_path = os.path.join(local_dir, subfolder)
    sae, config = load_dictionary(final_path, device=device)
    
    # Extract decoder weights
    # decoder.weight has shape (d_model, n_features)
    # Each column is a feature vector
    if hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight'):
        W_dec = sae.decoder.weight.detach().cpu()
    elif hasattr(sae, 'W_dec'):
        W_dec = sae.W_dec.detach().cpu()
    else:
        raise AttributeError(f"Could not find decoder weights in SAE")
    
    # Extract the specific feature (column)
    if feature_idx >= W_dec.shape[1]:
        raise IndexError(
            f"feature_idx {feature_idx} out of range. "
            f"This SAE has {W_dec.shape[1]} features."
        )
    
    feature_vector = W_dec[:, feature_idx]
    
    return feature_vector

def monitor_on_sequence(model: LanguageModel, sequence: str, direction: torch.Tensor, layer) -> torch.Tensor:
    """
    Monitor the activations of a specific direction on a sequence.
    """
    with torch.no_grad():
        with model.trace(sequence) as tracer:
            activations = model.model.layers[layer].output.save()
    return (activations @ direction).detach().squeeze()

def convert_prompt_to_format(tokenizer: AutoTokenizer, sample: dict) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": sample['prompt_text']}],
        tokenize=False,
        add_generation_prompt=True
    ) + "<|channel|>analysis<|message|>" + sample['reasoning'] + "<|end|><|start|>assistant<|channel|>final<|message|>" + sample['response'] + "<|return|>"

def parse_cot_entry(cot_entry: dict, prompts: list) -> dict:
    """
    Parse a CoT entry and match it with the corresponding prompt.
    
    Args:
        cot_entry: A single entry from all_complete_cots.json
        prompts: List of prompt dictionaries
    
    Returns:
        dict with keys: experiment, behavior_label, prompt, reasoning, response
    """
    # Extract task name from experiment field
    # Experiment format: "{task_name}_{behavior}_{index}"
    experiment = cot_entry['experiment']
    
    # Find matching prompt by checking which prompt name appears in the experiment
    matching_prompt = None
    for prompt in prompts:
        if experiment.startswith(prompt['name'] + '_'):
            matching_prompt = prompt
            break
    
    # Build the full prompt text
    if matching_prompt:
        prompt_text = f"{matching_prompt['task']}\n{matching_prompt['rules']}\n\n{matching_prompt['grader']}"
    else:
        prompt_text = None
    
    return {
        'experiment': cot_entry['experiment'],
        'behavior_label': cot_entry['behavior_label'],
        'prompt': matching_prompt,  # Full prompt dict
        'prompt_text': prompt_text,  # Combined text
        'reasoning': cot_entry['reasoning'],
        'response': cot_entry['response']
    }
