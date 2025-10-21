"""
Generation with SAE feature tracking.

This module handles text generation while tracking SAE feature activations
at each generated token.
"""
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from dictionary_learning.utils import load_dictionary
from huggingface_hub import snapshot_download
import os


class GenerationAnalyzer:
    """Analyzes SAE features during text generation."""
    
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        sae_repo: str = "andyrdt/saes-gpt-oss-20b",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.sae_repo = sae_repo
        self.device = device
        
        self.tokenizer = None
        self.model = None
        self.saes = {}  # Cache SAEs by (layer, trainer)
        
    def load_model(self):
        """Load the base language model."""
        if self.model is not None:
            return
            
        print(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",  # Let the model use its native dtype
        )
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
        
    def load_sae(self, layer: int, trainer: int = 0) -> torch.nn.Module:
        """Load SAE for a specific layer and trainer."""
        key = (layer, trainer)
        if key in self.saes:
            return self.saes[key]
            
        print(f"Loading SAE for layer {layer}, trainer {trainer}...")
        subfolder = f"resid_post_layer_{layer}/trainer_{trainer}"
        
        local_dir = snapshot_download(
            repo_id=self.sae_repo,
            allow_patterns=f"{subfolder}/*",
            local_dir_use_symlinks=False,
            cache_dir=None
        )
        
        final_path = os.path.join(local_dir, subfolder)
        sae, config = load_dictionary(final_path, device=self.device)
        sae.eval()
        
        self.saes[key] = sae
        print(f"SAE loaded for layer {layer}, trainer {trainer}")
        return sae
        
    def generate_with_features(
        self,
        prompt: str,
        layer: int,
        trainer: int = 0,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k_features: int = 20,
        do_sample: bool = True,
        reasoning_effort: str = "medium",
        assistant_prefill: str = "",
    ) -> Dict:
        """
        Generate text and track SAE features at each token.
        
        Args:
            prompt: Input text prompt (will be formatted with Harmony template)
            layer: Which layer to extract features from (0-indexed for hidden states)
            trainer: Which SAE trainer to use
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k_features: Number of top features to return per token
            do_sample: Whether to use sampling (vs greedy)
            reasoning_effort: Reasoning effort level for Harmony format ("low", "medium", "high")
            assistant_prefill: Optional text to prefill the assistant's response with
            
        Returns:
            Dictionary with:
                - prompt_text: Original prompt
                - formatted_prompt: Harmony-formatted prompt (including prefill)
                - assistant_prefill: The prefill text used
                - generated_text: Full generated text (prefill + new generation)
                - generated_tokens: List of generated token IDs
                - generated_token_strs: List of decoded token strings
                - features_per_token: List of dicts with feature info per token
        """
        self.load_model()
        sae = self.load_sae(layer, trainer)
        
        # Format prompt using OpenAI Harmony Response Format
        # Reference: demo.py lines 82-86
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            reasoning_effort=reasoning_effort
        )
        
        # Add assistant prefill if provided
        if assistant_prefill:
            formatted_prompt = formatted_prompt + assistant_prefill
        
        # Tokenize the formatted prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Storage for results
        all_tokens = []
        all_token_strs = []
        all_features = []
        
        # Generate token by token to track features
        current_input_ids = inputs.input_ids
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass with hidden states
                outputs = self.model(
                    current_input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                
                # Get logits for next token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Sample or greedy
                if do_sample and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Get hidden state at the last position (the token we just processed)
                # layer+1 because hidden_states[0] is embeddings
                layer_activation = outputs.hidden_states[layer + 1][:, -1, :]
                
                # Run through SAE
                feature_activations = sae.encode(layer_activation)  # Shape: [1, num_features]
                feature_activations = feature_activations.squeeze(0)  # Shape: [num_features]
                
                # Get top-k features
                top_values, top_indices = torch.topk(feature_activations, k=top_k_features)
                
                # Store token and features
                token_id = next_token.item()
                # Decode without special tokens for display
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                token_str_display = self.tokenizer.decode([token_id], skip_special_tokens=True)
                
                all_tokens.append(token_id)
                all_token_strs.append(token_str)
                all_features.append({
                    "token_id": token_id,
                    "token_text": token_str_display if token_str_display else token_str,  # Use display version
                    "token_text_raw": token_str,  # Keep raw for debugging
                    "position": step,
                    "top_features": [
                        {
                            "feature_id": int(top_indices[i].item()),
                            "activation": float(top_values[i].item()),
                        }
                        for i in range(len(top_indices))
                    ]
                })
                
                # Check for EOS
                if token_id == self.tokenizer.eos_token_id:
                    break
                
                # Prepare next input
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
        
        # Decode full generation (skip special tokens for display)
        full_output = self.tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
        generated_only = self.tokenizer.decode(current_input_ids[0][prompt_length:], skip_special_tokens=True)
        
        return {
            "prompt_text": prompt,
            "formatted_prompt": formatted_prompt,
            "assistant_prefill": assistant_prefill,
            "full_text": full_output,
            "generated_text": generated_only,
            "generated_tokens": all_tokens,
            "generated_token_strs": all_token_strs,
            "features_per_token": all_features,
            "layer": layer,
            "trainer": trainer,
            "reasoning_effort": reasoning_effort,
        }


# Global instance (will be initialized by server)
_analyzer = None


def get_analyzer(model_name: str = "openai/gpt-oss-20b") -> GenerationAnalyzer:
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = GenerationAnalyzer(model_name=model_name)
    return _analyzer

