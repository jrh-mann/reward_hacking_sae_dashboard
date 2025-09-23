General setup:

```bash
uv venv
uv pip install -r requirements.txt
```

Run the max-activating-example server:
```bash
uv run -m open_source_em_features.app.run_mae_app \
  --max_activating_example_cache_dir .mae_cache \
  --port 7863
```

Warning - this will download large files to your local .mae_cache directory (~120GB total). Please make sure have a lot of disk space available!

I've also included `demo.py` which shows an example of loading the model, an SAE, and doing some very basic feature surfacing.
