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

### 3. Demo Script

I've also included `demo.py` which shows an example of loading the model, an SAE, and doing some very basic feature surfacing programmatically.

## Testing

Test the generation API:

```bash
uv run python test_generation.py
```
