"""
Utility module for extracting router logits from MoE models.
"""

import torch
from nnsight import LanguageModel
from typing import Union, List


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
    
    with model.trace(prompt, scan=False) as tracer:
        # Capture router output from each layer
        for layer_idx in range(num_layers):
            router_out = model.model.layers[layer_idx].mlp.router.output.save()
            router_outputs.append(router_out)
    
    # Stack all router outputs
    # Each router_out should be (batch_size, seq_len, num_experts)
    # Stack along a new dimension to get (batch_size, seq_len, num_layers, num_experts)    
    return router_outputs