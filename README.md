General setup:

```bash
uv venv
uv pip install -r requirements.txt
```

## Features

### 1. Max-Activating-Example (MAE) Explorer

Run the server to browse pre-computed max-activating examples:

```bash
uv run -m app.run_mae_app \
  --max_activating_example_cache_dir .mae_cache \
  --port 7863
```

Navigate to `http://localhost:7863` to explore features.

**Warning** - this will download large files to your local `.mae_cache` directory (~120GB total). Please make sure you have sufficient disk space available!

### 2. Interactive Generation with Feature Tracking

Generate text and see SAE features at each token:

```bash
uv run -m app.run_mae_app --port 7863
```

Navigate to `http://localhost:7863/generate` to:
- Input prompts and generate text
- Track SAE feature activations at each generated token
- Click on features to see their max-activating examples
- Analyze features across different layers and trainers

See [GENERATION_GUIDE.md](GENERATION_GUIDE.md) for detailed usage instructions.

### 3. Find Similar Features to Steering Vectors

Analyze steering vectors and find the most similar SAE features at each layer:

```bash
uv run python find_similar_features.py \
  --steering_vectors_path steering_vectors.pt \
  --trainer 0 \
  --top_k 10 \
  --layers 3,7,11,15,19,23
```

This script:
- Loads a `steering_vectors.pt` file (expected shape: `[24, 2880]`)
- Downloads SAE weights from HuggingFace (`andyrdt/saes-gpt-oss-20b`)
- For each layer, computes cosine similarity with all SAE features
- Reports the top-K most similar features
- Provides dashboard URLs to view the features in the MAE explorer

**Note:** The SAE weights (~500MB per layer/trainer combo) are separate from the MAE cache and will be downloaded to `~/.cache/huggingface/hub/` automatically.

### 4. Decompose Steering Vector into Feature Combinations

Find which **combination of SAE features** best reconstructs your steering vector:

```bash
uv run python decompose_steering_vector.py \
  --steering_vectors_path steering_vectors.pt \
  --layer 19 \
  --trainer 0 \
  --top_n 20 \
  --method all
```

This script provides **three different methods**:

1. **SAE Encoder** (most principled): Uses the SAE's encoder to decompose your vector into sparse features (how the SAE was trained to decompose activations)
2. **Greedy Search** (most interpretable): Iteratively selects features that maximize cosine similarity with optimal weights
3. **Top-K Baseline**: Simply sums the most similar individual features

**Output:** For each method, you get:
- List of feature IDs and their weights/activations
- Cosine similarity between reconstruction and original vector
- L2 reconstruction error
- Dashboard URLs to view the features

**Use cases:**
- Understand which features combine to create a steering direction
- Compare different decomposition strategies
- Find interpretable feature combinations for your steering vectors

### 5. Demo Script

I've also included `demo.py` which shows an example of loading the model, an SAE, and doing some very basic feature surfacing programmatically.

## Testing

Test the generation API:

```bash
uv run python test_generation.py
```
