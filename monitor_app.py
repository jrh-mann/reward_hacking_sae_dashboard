from flask import Flask, render_template, jsonify, request
from probe_attribution import get_direction, parse_cot_entry, convert_prompt_to_format, monitor_on_sequence
import torch
from nnsight import LanguageModel
import json
import numpy as np

app = Flask(__name__)

# Global variables for loaded models and data
model = None
tokenizer = None
direction = None
samples = []
formatted_sequences = []

def initialize():
    """Initialize model and data on startup"""
    global model, tokenizer, direction, samples, formatted_sequences
    
    print("Loading model...")
    model = LanguageModel(
        "openai/gpt-oss-20b",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = model.tokenizer
    
    print("Loading direction...")
    # Try to load from steering_vectors.pt first, fallback to get_direction
    '''try:
        1 / 0
        direction = torch.load("steering_vectors.pt")[15].to("cuda:0").to(torch.bfloat16)
    except:
        
        direction = get_direction(layer_idx=15, feature_idx=71190)
        direction = direction.to("cuda:0").to(torch.bfloat16)'''

    direction = get_direction(layer_idx=15, feature_idx=71190)
    direction = direction.to("cuda:0").to(torch.bfloat16)
    
    print("Loading data...")
    all_cots = json.load(open("all_complete_cots.json"))
    prompts = json.load(open("reward_hacking_prompts.json"))
    
    print("Processing samples...")
    samples = [parse_cot_entry(cot, prompts) for cot in all_cots[:100]]
    formatted_sequences = [convert_prompt_to_format(tokenizer, sample) for sample in samples]
    
    print("Initialization complete!")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('monitor.html')

@app.route('/api/samples')
def get_samples():
    """Get list of all samples with metadata"""
    sample_list = []
    for i, sample in enumerate(samples):
        sample_list.append({
            'index': i,
            'experiment': sample['experiment'],
            'behavior_label': sample['behavior_label'],
            'task_name': sample['prompt']['name'] if sample['prompt'] else 'Unknown'
        })
    return jsonify(sample_list)

@app.route('/api/sequence/<int:index>')
def get_sequence(index):
    """Get tokens, firings, and metadata for a specific sequence"""
    if index < 0 or index >= len(samples):
        return jsonify({'error': 'Index out of range'}), 400
    
    sample = samples[index]
    sequence = formatted_sequences[index]
    
    # Get firings
    firings = monitor_on_sequence(model, sequence, direction, 15)
    
    # Tokenize to get individual tokens
    tokens = tokenizer.encode(sequence)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    
    # Find where reasoning and response start
    # The sequence format is: <prompt><|channel|>analysis<|message|><reasoning><|end|><|start|>assistant<|channel|>final<|message|><response><|return|>
    reasoning_marker = "<|channel|>analysis<|message|>"
    response_marker = "<|channel|>final<|message|>"
    
    reasoning_start_idx = sequence.find(reasoning_marker)
    response_start_idx = sequence.find(response_marker)
    
    if reasoning_start_idx != -1:
        reasoning_start_idx += len(reasoning_marker)
    if response_start_idx != -1:
        response_start_idx += len(response_marker)
    
    # Map character positions to token positions
    reasoning_token_idx = 0
    response_token_idx = 0
    
    char_count = 0
    for i, token_str in enumerate(token_strs):
        if reasoning_start_idx != -1 and char_count >= reasoning_start_idx and reasoning_token_idx == 0:
            reasoning_token_idx = i
        if response_start_idx != -1 and char_count >= response_start_idx and response_token_idx == 0:
            response_token_idx = i
        char_count += len(token_str)
    
    # Filter to only reasoning and response tokens
    if reasoning_token_idx > 0:
        token_strs = token_strs[reasoning_token_idx:]
        firings = firings[reasoning_token_idx:]
    
    # Convert firings to list (convert to float32 first since numpy doesn't support bfloat16)
    firings_list = firings.cpu().to(torch.float32).numpy().tolist()
    
    return jsonify({
        'experiment': sample['experiment'],
        'behavior_label': sample['behavior_label'],
        'task_name': sample['prompt']['name'] if sample['prompt'] else 'Unknown',
        'tokens': token_strs,
        'firings': firings_list,
        'reasoning_start': 0,
        'response_start': response_token_idx - reasoning_token_idx if response_token_idx > reasoning_token_idx else len(token_strs)
    })

@app.route('/api/max_activating')
def get_max_activating():
    """Get max activating dataset"""
    try:
        with open('max_activating_dataset.json', 'r') as f:
            dataset = json.load(f)
        
        # Get query parameters for filtering/pagination
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Return paginated results
        return jsonify({
            'total': len(dataset),
            'offset': offset,
            'limit': limit,
            'data': dataset[offset:offset+limit]
        })
    except FileNotFoundError:
        return jsonify({'error': 'Max activating dataset not found. Please generate it first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000, debug=False)

