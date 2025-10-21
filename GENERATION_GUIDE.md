# SAE Feature Generation Guide

This guide explains how to use the interactive generation feature to visualize SAE features at each generated token.

## Overview

The generation endpoint allows you to:
1. Input a text prompt
2. Generate text using the GPT model
3. Track SAE feature activations at each generated token
4. Visualize which features are most active
5. Click on features to see their max-activating examples

## Starting the Server

```bash
uv run -m app.run_mae_app --port 7863
```

Or with custom settings:

```bash
uv run -m app.run_mae_app \
  --max_activating_example_cache_dir .mae_cache \
  --port 7863 \
  --default_layer 11 \
  --default_trainer 0
```

## Using the Web Interface

1. Navigate to `http://localhost:7863/generate` in your browser
2. Enter your prompt in the text area
3. (Optional) Enter an assistant prefill to start the response with specific text
4. Configure settings:
   - **Layer**: Which transformer layer to extract features from (3, 7, 11, 15, 19, or 23)
   - **Trainer**: Which SAE trainer to use (0 or 1)
   - **Max Tokens**: How many tokens to generate (1-1000)
   - **Temperature**: Sampling temperature (0 = greedy, higher = more random)
   - **Top-K Features**: Number of top features to display per token (1-50)
   - **Reasoning Effort**: Harmony format reasoning level (low, medium, high)

5. Click "Generate" and wait for results (first generation loads the model, so it may take a minute)

## Understanding the Results

### Output Sections

1. **Prompt**: Your input text
2. **Generated Text**: The complete generated text
3. **Token-by-Token Features**: A table showing:
   - Position: Token index in the generation
   - Token: The actual generated token
   - Top Features: Clickable feature badges showing feature ID and activation strength

### Interacting with Features

- **Hover** over a feature badge to see the exact activation value
- **Click** on a feature to open its max-activating examples in a new tab
- Features are color-coded and sized by activation strength

## Example Use Cases

### 1. Understanding Specific Concepts

```
Prompt: "The concept of democracy is"
```
Track which features activate for political/governance concepts.

### 2. Analyzing Style Changes

```
Prompt: "Once upon a time, in a land far away"
```
See which features activate for narrative/storytelling style.

### 2b. Using Assistant Prefill for Steering

```
Prompt: "Tell me about cats"
Assistant Prefill: "Cats are fascinating creatures. Let me explain in detail:"
```
Force the assistant to start with a specific tone or approach.

### 3. Safety/Refusal Analysis

```
Prompt: "How to manipulate someone"
```
Observe which features activate for potentially harmful content.

### 4. Factual vs Creative

Compare features between:
- "The capital of France is" (factual)
- "In my imagination, the capital of France is" (creative)

## API Usage

You can also use the endpoint programmatically:

```python
import requests
import json

response = requests.post(
    "http://localhost:7863/api/generate",
    json={
        "prompt": "The cat sat on the",
        "assistant_prefill": "",  # Optional: prefill assistant response
        "layer": 11,
        "trainer": 0,
        "max_new_tokens": 30,
        "temperature": 1.0,
        "top_k_features": 20,
        "reasoning_effort": "medium",
    }
)

result = response.json()
print(f"Generated: {result['generated_text']}")

for token_data in result['features_per_token']:
    print(f"\nToken: {token_data['token_text']}")
    for feat in token_data['top_features'][:5]:
        print(f"  Feature {feat['feature_id']}: {feat['activation']:.3f}")
```

## Performance Notes

- **First request**: Loads model (~40GB) and SAE, may take 1-2 minutes
- **Subsequent requests**: Much faster, only generation time
- **Memory**: Requires GPU with sufficient VRAM (recommended: 40GB+)
- **Generation speed**: Slower than normal inference due to feature extraction

## Tips

1. Start with shorter generations (10-30 tokens) to iterate faster
2. Use layer 11 as a good default middle layer
3. Temperature 0.0 for deterministic generation
4. Temperature 1.0+ for more diverse outputs
5. Compare multiple generations with the same prompt to see variation
6. Use assistant prefill to steer the response in a specific direction
7. Special tokens are hidden in the display but still used during generation
8. Higher reasoning effort may produce more thorough analysis channels

## Troubleshooting

### "CUDA out of memory"
- Model requires significant GPU memory
- Try reducing batch operations or using a smaller model

### "Model loading takes forever"
- First load downloads from HuggingFace (~40GB)
- Subsequent loads use cache

### "Features not showing up"
- Check that layer and trainer combination exists
- Ensure SAE was downloaded correctly

## Advanced: Multi-Layer Analysis

To analyze features across multiple layers, you can make multiple requests with different layer settings and compare the results manually.

