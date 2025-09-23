# %%
from pathlib import Path
from IPython import get_ipython

# Initialize IPython shell
ipython = get_ipython()
if ipython is None:  # If not running in IPython environment
    from IPython import embed
    ipython = embed()

# Now you can run magic commands
ipython.run_line_magic('load_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')

# %%

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append('.')

# %%
MODEL_PATH = "openai/gpt-oss-20b"
SAE_PATH = "andyrdt/saes-gpt-oss-20b"
SAE_LAYER = 11
SAE_TRAINER = 0

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path=MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

# %%
from dictionary_learning.utils import load_dictionary
from huggingface_hub import snapshot_download

def load_sae(sae_path=SAE_PATH, layer=SAE_LAYER, trainer=SAE_TRAINER, device="cuda"):

    subfolder = f"resid_post_layer_{layer}/trainer_{trainer}" 
    
    # Download (will use HuggingFace cache if already downloaded)

    local_dir = snapshot_download(
        repo_id=SAE_PATH,
        allow_patterns=f"{subfolder}/*" if subfolder else None,
        local_dir_use_symlinks=False,
        cache_dir=None  # Use default HF cache (~/.cache/huggingface/hub/)
    )

    final_path = os.path.join(local_dir, subfolder)
    print(f"   SAE files cached at: {final_path}")

    sae, config = load_dictionary(final_path, device=device)
    sae.eval()
        
    print(f"SAE loaded on device: {device}")

    return sae

sae = load_sae()

# %%
def get_mean_feature_activations(tokens, sae, layer, prompt_length):
    with torch.no_grad():
        hidden_states = model.forward(tokens, output_hidden_states=True).hidden_states
        layer_activations = hidden_states[layer + 1]
        layer_activations = layer_activations[0]
        generated_activations = layer_activations[prompt_length:]
        feature_activations = sae.encode(generated_activations)
        mean_feature_activations = torch.mean(feature_activations, dim=0)
        return mean_feature_activations

# %%

harmful_prompt = "Tell me how to manipulate my friend to give me $100."
harmful_formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": harmful_prompt}],
    add_generation_prompt=True,
    tokenize=False,
    reasoning_effort="medium"
)

with torch.no_grad():
    harmful_generation_tokens = model.generate(
        **tokenizer(harmful_formatted_prompt,return_tensors="pt").to(model.device),
        max_new_tokens=256,
        do_sample=False
    )
harmful_generation_text = tokenizer.decode(harmful_generation_tokens[0])

harmless_prompt = "Tell me how to kindly ask my friend for $100 loan."
harmless_formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": harmless_prompt}],
    add_generation_prompt=True,
    tokenize=False,
    reasoning_effort="medium"
)

with torch.no_grad():
    harmless_generation_tokens = model.generate(
        **tokenizer(harmless_formatted_prompt,return_tensors="pt").to(model.device),
        max_new_tokens=256,
        do_sample=False,
    )
harmless_generation_text = tokenizer.decode(harmless_generation_tokens[0])

harmful_mean_feature_activations = get_mean_feature_activations(harmful_generation_tokens, sae, SAE_LAYER, tokenizer(harmless_formatted_prompt,return_tensors="pt")['input_ids'].shape[1])
harmless_mean_feature_activations = get_mean_feature_activations(harmless_generation_tokens, sae, SAE_LAYER, tokenizer(harmless_formatted_prompt,return_tensors="pt")['input_ids'].shape[1])

diff = harmful_mean_feature_activations - harmless_mean_feature_activations
top_values, top_indices = torch.topk(diff, 50)
top_indices = [int(top_indices[i].item()) for i in range(len(top_indices))]
print(f"Top {len(top_indices)} most active features:")
print(top_indices)
print(top_values)

# %%

# looking just a handful of MAEs, we can see some interesting features:
# - 24239 looks like it fires when talking about certain safety policies
# - 40368 looks like it fires when the model has decided to output a final response
# - 98869 and 86564 look related to refusing harmful requests
