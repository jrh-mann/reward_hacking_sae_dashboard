#!/usr/bin/env python
import pathlib, h5py, html, re, contextlib, json
from flask import Flask, Response, request, jsonify
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# Suppress backend presence warnings from transformers
hf_logging.set_verbosity_error()

# -------------------- Global state (initialized by init_app) -----------------
MODEL_SPECS = {}
DEFAULT_LAYER, DEFAULT_TRAINER = 11, 0

current_model_key = None
MODEL_NAME = None
BASE_DIR = None

app = Flask(__name__, static_folder=None)

# -------------------- tokenizer ---------------------------------------------
tokenizer = None
PAD_ID = None


def init_app(model_specs: dict, default_layer: int, default_trainer: int):
    global MODEL_SPECS, DEFAULT_LAYER, DEFAULT_TRAINER
    global current_model_key, MODEL_NAME, BASE_DIR
    global tokenizer, PAD_ID

    MODEL_SPECS = model_specs
    DEFAULT_LAYER, DEFAULT_TRAINER = default_layer, default_trainer

    if not MODEL_SPECS:
        raise ValueError("MODEL_SPECS is empty")

    # Use first key as default
    current_model_key = next(iter(MODEL_SPECS.keys()))
    MODEL_NAME = MODEL_SPECS[current_model_key]["name"]
    BASE_DIR = pathlib.Path(MODEL_SPECS[current_model_key]["base_dir"]).expanduser().resolve()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer.pad_token_id

    # Open default variant
    set_variant(current_model_key, DEFAULT_LAYER, DEFAULT_TRAINER)


# -------------------- variant handling --------------------------------------
current_layer, current_trainer = None, None
h5_chat = h5_pt = h5_sim = None


def variant_path(layer:int, trainer:int):
    return BASE_DIR / f"resid_post_layer_{layer}" / f"trainer_{trainer}"


def open_h5_group(layer:int, trainer:int):
    root = variant_path(layer, trainer)
    return (
        h5py.File(root / "chat_topk.h5", "r"),
        h5py.File(root / "pt_topk.h5", "r"),
        h5py.File(root / "embed_unembed_similarity.h5", "r"),
    )


def set_variant(model_key: str, layer: int, trainer: int):
    global current_model_key, MODEL_NAME, BASE_DIR, tokenizer, PAD_ID
    global h5_chat, h5_pt, h5_sim, current_layer, current_trainer

    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key '{model_key}'. Allowed: {list(MODEL_SPECS)}")

    prev_model, prev_layer, prev_trainer = current_model_key, current_layer, current_trainer

    if model_key != current_model_key:
        current_model_key = model_key
        MODEL_NAME = MODEL_SPECS[model_key]["name"]
        BASE_DIR = pathlib.Path(MODEL_SPECS[model_key]["base_dir"]).expanduser().resolve()

        # reload tokenizer for the new model
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tokenizer = tok
        PAD_ID = tokenizer.pad_token_id

    if (model_key != prev_model) or (layer != prev_layer) or (trainer != prev_trainer):
        new_chat, new_pt, new_sim = open_h5_group(layer, trainer)
        with contextlib.suppress(Exception):
            h5_chat.close(); h5_pt.close(); h5_sim.close()
        h5_chat, h5_pt, h5_sim = new_chat, new_pt, new_sim
        current_layer, current_trainer = layer, trainer


# -------------------- helpers ----------------------------------------------
def rgba(a:float)->str: return f"rgba(216,68,0,{max(a,0.05):.3f})"
SPACE_SYM,NL_SYM,TAB_SYM="␣","↵","⇥"
def show(t:int)->str:
    raw=tokenizer.decode([int(t)],skip_special_tokens=False)
    if raw.startswith("<|") and raw.endswith("|>"): return raw
    rep=repr(raw)[1:-1].replace("\\n",NL_SYM).replace("\\t",TAB_SYM)
    return SPACE_SYM if rep==" " else html.escape(rep,False)
_sci_re=re.compile(r"([0-9.]+)e([+-]?)\d+")
def sci(v:float,sig:int=2)->str:
    if v==0: return "0"
    s=f"{v:.{sig}e}";return s


def example_html(tok_ids,acts,score):
    while len(tok_ids)>1 and tok_ids[-2]==PAD_ID: tok_ids,acts=tok_ids[:-1],acts[:-1]
    pieces,starts,pos=[],[],0
    for tid in tok_ids: txt=show(tid);pieces.append(txt);starts.append(pos);pos+=len(txt)
    ends,total=[s+len(p) for s,p in zip(starts,pieces)],pos
    peak=acts.argmax();ps=starts[peak];pl=len(pieces[peak]);pe=ps+pl
    CHARS_LEFT, PREVIEW_LEN = 60, 120
    left=min(ps,CHARS_LEFT);ws, we = ps-left, min(total, pe+(PREVIEW_LEN-CHARS_LEFT-pl))
    raw="".join(pieces)[ws:we]; padl=" "*(CHARS_LEFT-left); padr=" "*(PREVIEW_LEN-len(raw)-len(padl))
    win_tokens,win_acts=[],[]
    for p,a,s,e in zip(pieces,acts,starts,ends):
        if e<=ws or s>=we: continue
        win_tokens.append(p[max(ws-s,0):len(p)-max(e-we,0)]);win_acts.append(a)
    wmax = max(win_acts) if win_acts else 1.0
    safe_wmax = wmax if wmax > 0 else 1.0
    cursor = len(padl)
    spans  = []
    for tp, a in zip(win_tokens, win_acts):
        cls = "peak" if cursor == CHARS_LEFT else "preview-tok"
        spans.append(
            f'<span class=\"{cls}\" title=\"{a:.4f}\" '
            f'style=\"background:{rgba(max(a/safe_wmax,0.05))};\">{html.escape(tp,False)}</span>')
        cursor += len(tp)
    preview=f'<span class=\"mono-window\">{padl}{"".join(spans)}{padr}</span><span class=\"score\">(score {score:.3f})</span>'
    fmax=acts.max() or 1.0
    full="".join(f'<span class=\"tok\" title=\"{a:.4f}\" style=\"background:{rgba(max(a/fmax,0.05))};\">{html.escape(p)}</span>' for p,a in zip(pieces,acts))
    return ("<div class=\"example collapsed\" onclick=\"this.classList.toggle('collapsed')\">"
            f"<div class=\"snippet\">{preview}</div><div class=\"full\">{full}</div></div>")


def sim_panel(fid:int)->str:
    F=h5_sim["in_top_tokens"].shape[0]
    if fid<0 or fid>=F: return ""
    def strip(tok,css): return "<div class='token-strip "+css+"'>"+"".join(f"<span class='pill'>{show(t)}</span>" for t in tok)+"</div>"
    enc_t,enc_b=h5_sim["in_top_tokens"][fid],h5_sim["in_bottom_tokens"][fid]
    dec_t,dec_b=h5_sim["out_top_tokens"][fid],h5_sim["out_bottom_tokens"][fid]
    return ("<div class='sim-panel'><table class='sim'>"
            "<tr><th>Embedding → Encoder</th><th>Decoder → Unembedding</th></tr>"
            f"<tr class='top-row'><td>{strip(enc_t,'top')}</td><td>{strip(dec_t,'top')}</td></tr>"
            f"<tr class='bottom-row'><td>{strip(enc_b,'bottom')}</td><td>{strip(dec_b,'bottom')}</td></tr>"
            "</table></div>")


# -------------------- routes -------------------------------------------------
@app.route("/")
def index():
    model_options = "\n".join(
        f"<option value=\"{html.escape(k)}\" {'selected' if k==current_model_key else ''}>{html.escape(MODEL_SPECS[k]['name'])}</option>"
        for k in MODEL_SPECS
    )
    page = """
<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>
<title>EM Feature Explorer</title>
<style>
 @import url(\"https://fonts.googleapis.com/css2?family=Noto+Sans+Mono:wght@400&display=swap\");
 body{font-family:'Noto Sans Mono',monospace;margin:20px;}
 #ctrl{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;margin-bottom:16px;gap:12px;}
 #ctrl .left, #ctrl .right{display:flex;align-items:flex-end;gap:8px;flex-wrap:wrap;}
 input[type=number]{width:120px;font-size:1em;}
 .feature{margin-bottom:32px;}
 .feat-header{display:flex;justify-content:space-between;align-items:flex-end;border-left:4px solid transparent;padding-left:4px;}
 .sim{table-layout:fixed;width:100%;border-collapse:collapse;font-size:0.9em;}
 .sim th,.sim td{width:50%;border:1px solid #ccc;padding:2px 4px;vertical-align:top;}
 .token-strip{display:block;white-space:nowrap;overflow-x:auto;padding:2px;border-radius:4px;}
 .token-strip.top{background:#e8f9e8;} .token-strip.bottom{background:#fbeaea;}
 .pill{display:inline-block;padding:1px 4px;margin:1px;border:1px solid #bbb;border-radius:6px;}
 .sec-header{cursor:pointer;background:#eee;padding:6px 8px;border:1px solid #ccc;border-radius:4px;margin-top:12px;}
 .sec-header:hover{background:#ddd;}
 .sec-body{border:1px solid #ccc;border-top:none;padding:8px;}
 .sec-body.collapsed{display:none;}
 .example{border:1px solid #c3c3c3;margin:14px 0;padding:10px;border-radius:6px;background:#fafafa;}
 .snippet{display:flex;white-space:pre;overflow-x:auto;}
 .mono-window{white-space:pre;}
 .peak{background:#d84400;color:#fff;border-radius:8px;padding:0;}
 .preview-tok{padding:0;} .score{margin-left:auto;color:#555;font-size:0.9em;}
 .tok{display:inline-block;padding:0 1px;border-right:1px solid rgba(0,0,0,0.08);}
 .tok:last-child{border-right:none;} .tok:hover{border-right-color:#888;}
 .full{display:none;white-space:pre-wrap;margin-top:12px;padding-top:6px;border-top:1px dashed #ccc;}
 .example.collapsed .full{display:none;} .example:not(.collapsed) .full{display:block;}
 /* Panels */
 .panel{background:#f7f8fa;border:1px solid #dcdcdc;border-radius:8px;padding:10px 12px;}
 .panel h3{margin:0 0 6px 0;font-size:1.0em;font-weight:700;}
 .help{color:#666;font-size:0.85em;margin-bottom:8px;}
 .freq{color:#555;font-size:0.85em;margin-left:6px;}
</style>
</head><body>
<h1>EM Feature Explorer</h1>
<div style=\"margin-bottom:16px;\">
  <a href=\"/generate\" style=\"color:#0066cc;text-decoration:none;margin-right:12px;\">🔬 Interactive Generation →</a>
  <span style=\"color:#999;\">|</span>
  <span style=\"margin-left:12px;color:#666;\">Explore max-activating examples below</span>
</div>

<div id=\"ctrl\">
  <div class=\"left\">
    <div class=\"panel\">
      <h3>Feature selector</h3>
      <div class=\"help\">Specify a list of feature indices to load.</div>
      <label>Features:&nbsp;<input id=\"fids\" type=\"text\" style=\"width:260px\"
               placeholder=\"e.g. 3, 17  42\" onkeyup=\"if(event.key==='Enter') load();\"></label>
      <button onclick=\"load()\">Load</button>
    </div>
  </div>
  <div class=\"right\">
    <div class=\"panel\">
      <h3>SAE selector</h3>
      <div class=\"help\">Select a model, layer, and trainer to load.</div>
      <label>Model:
        <select id=\"model\">
          __MODEL_OPTIONS__
        </select>
      </label>
      <label>Layer:
        <select id=\"layer\">
          <option value=\"3\">3</option>
          <option value=\"7\">7</option>
          <option value=\"11\" selected>11</option>
          <option value=\"15\">15</option>
          <option value=\"19\">19</option>
          <option value=\"23\">23</option>
        </select>
      </label>
      <label>Trainer:
        <select id=\"trainer\"><option value=\"0\" selected>0 (k=64)</option><option value=\"1\">1 (k=128)</option></select>
      </label>
      <button onclick=\"setVariant()\">Apply</button>
    </div>
  </div>
</div>
<div id=\"out\"></div>

<script>
async function load(){
  const raw=document.getElementById('fids').value.trim();
  if(!raw) return;
  const ids=raw.split(/[,\\s]+/).filter(Boolean);
  const out=document.getElementById('out'); 
  out.innerHTML='<p>Loading ' + ids.length + ' features...</p>';

  const placeholders = ids.map(id => {
    const div = document.createElement('div');
    div.id = `placeholder-${id}`;
    div.innerHTML = `<p>Loading feature ${id}...</p>`;
    return div;
  });

  out.innerHTML = '';
  placeholders.forEach(placeholder => out.appendChild(placeholder));

  updateURL();

  const BATCH_SIZE = 5;
  async function loadFeature(id) {
    try {
      const response = await fetch(`/feature/${id}`);
      const html = await response.text();
      const placeholder = document.getElementById(`placeholder-${id}`);
      placeholder.innerHTML = html;
    } catch (error) {
      const placeholder = document.getElementById(`placeholder-${id}`);
      placeholder.innerHTML = `<p style=\"color:red\">Error loading feature ${id}: ${error.message}</p>`;
    }
  }

  for (let i = 0; i < ids.length; i += BATCH_SIZE) {
    const batch = ids.slice(i, i + BATCH_SIZE);
    await Promise.all(batch.map(id => loadFeature(id)));
  }
}

async function setVariant(){
  const model = document.getElementById('model').value;
  const layer = document.getElementById('layer').value;
  const trainer = document.getElementById('trainer').value;
  if(model===''||layer===''||trainer==='') return;
  const out=document.getElementById('out'); out.innerHTML='<p>Loading…</p>';
  const msg = await (await fetch(`/variant/${model}/${layer}/${trainer}`)).text();
  out.innerHTML=msg;
  updateURL();
}

function updateURL(){
  const params = new URLSearchParams();
  params.set('model', document.getElementById('model').value);
  params.set('layer', document.getElementById('layer').value);
  params.set('trainer', document.getElementById('trainer').value);
  const fids = document.getElementById('fids').value.trim();
  if(fids) params.set('fids', fids);
  history.replaceState(null, '', location.pathname + '?' + params.toString());
}

(function initFromURL(){
  const params = new URLSearchParams(window.location.search);
  if(params.has('model'))   document.getElementById('model').value = params.get('model');
  if(params.has('layer'))   document.getElementById('layer').value = params.get('layer');
  if(params.has('trainer')) document.getElementById('trainer').value = params.get('trainer');
  if(params.has('fids'))    document.getElementById('fids').value   = params.get('fids');

  setVariant().then(()=>{ if(params.has('fids')) load(); });
})();
</script>
</body></html>
"""
    page = page.replace("__MODEL_OPTIONS__", model_options)
    return Response(page, mimetype="text/html")


@app.route("/feature/<int:fid>")
def feature(fid:int):
    return Response(feature_block(fid), mimetype="text/html")


@app.route("/variant/<model_key>/<int:layer>/<int:trainer>")
def variant(model_key:str, layer:int, trainer:int):
    try:
        set_variant(model_key, int(layer), int(trainer))
        human_name = MODEL_SPECS[model_key]["name"]
        msg = (f"<p style='color:green'>Switched to model '{html.escape(human_name)}', "
               f"layer {layer}, trainer {trainer}</p>")
    except FileNotFoundError:
        msg=(f"<p style='color:red'>Files not found for model {model_key}, "
             f"layer {layer}, trainer {trainer}</p>")
    except Exception as e:
        msg=f"<p style='color:red'>Error: {html.escape(str(e))}</p>"
    return Response(msg, mimetype="text/html")


def feature_block(fid:int)->str:
    F=h5_chat["scores"].shape[0]
    if fid<0 or fid>=F: return f"<p style='color:red'>Feature {fid} out of range.</p>"

    head=(f"<div class='feat-header' id='feat-{fid}'>"
          f"<div><h2>Feature {fid}</h2><span class='meta'>(layer {current_layer}, trainer {current_trainer})</span></div>"
          f"</div>")
    def section(name,h5file,css_id,TOP_N=20):
        tok,acts,score=h5file["tokens"][fid][:TOP_N],h5file["sae_acts"][fid][:TOP_N],h5file["scores"][fid][:TOP_N]
        freq,total=h5file["frequency"][fid],h5file.attrs.get("tokens_seen", 0)
        freq_txt = sci(freq/total if total else 0.0)
        hdr=f"<div class='sec-header' onclick=\"document.getElementById('{css_id}').classList.toggle('collapsed')\"><b>{name}</b><span class='freq'>({freq_txt})</span></div>"
        body=f"<div id='{css_id}' class='sec-body collapsed'>"+"".join(example_html(tok[k],acts[k],score[k]) for k in range(len(tok)))+"</div>"
        return hdr+body
    html_rows=[head,sim_panel(fid),"<hr/>",section("CHAT",h5_chat,f"chat-sec-{fid}"),section("PRETRAIN",h5_pt,f"pt-sec-{fid}")]
    return "<div class='feature'>"+"\n".join(html_rows)+"</div>"


# -------------------- Generation routes --------------------------------------
@app.route("/generate")
def generate_page():
    """Interactive generation page with feature tracking."""
    page = """
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>SAE Feature Generation</title>
<style>
 @import url("https://fonts.googleapis.com/css2?family=Noto+Sans+Mono:wght@400&display=swap");
 body{font-family:'Noto Sans Mono',monospace;margin:20px;background:#fafafa;}
 .container{max-width:1200px;margin:0 auto;}
 h1{margin-bottom:8px;}
 .subtitle{color:#666;margin-bottom:20px;}
 .nav{margin-bottom:20px;}
 .nav a{margin-right:16px;color:#0066cc;text-decoration:none;}
 .nav a:hover{text-decoration:underline;}
 .panel{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:20px;}
 .panel h3{margin:0 0 12px 0;font-size:1.1em;}
 .form-row{margin-bottom:12px;}
 .form-row label{display:block;margin-bottom:4px;font-weight:600;}
 .form-row input, .form-row select, .form-row textarea{width:100%;padding:8px;font-family:inherit;font-size:0.95em;border:1px solid #ccc;border-radius:4px;}
 .form-row textarea{resize:vertical;min-height:80px;}
 .form-row-inline{display:flex;gap:12px;}
 .form-row-inline > div{flex:1;}
 button{background:#0066cc;color:#fff;border:none;padding:10px 20px;border-radius:4px;font-size:1em;cursor:pointer;font-family:inherit;}
 button:hover{background:#0052a3;}
 button:disabled{background:#ccc;cursor:not-allowed;}
 #status{margin-top:12px;padding:10px;border-radius:4px;display:none;}
 #status.loading{display:block;background:#fff3cd;border:1px solid #ffc107;}
 #status.error{display:block;background:#f8d7da;border:1px solid #dc3545;}
 #status.success{display:block;background:#d4edda;border:1px solid:#28a745;}
 #results{margin-top:20px;}
 .result-header{background:#fff;border:1px solid #ddd;padding:16px;border-radius:8px;margin-bottom:12px;}
 .result-header h3{margin:0 0 8px 0;}
 .meta{color:#666;font-size:0.9em;}
 .generated-text{background:#fff;border:1px solid #ddd;padding:16px;border-radius:8px;margin-bottom:20px;white-space:pre-wrap;font-family:'Noto Sans Mono',monospace;}
 .token-table{width:100%;border-collapse:collapse;background:#fff;}
 .token-table th,.token-table td{padding:10px;text-align:left;border:1px solid #ddd;}
 .token-table th{background:#f0f0f0;font-weight:600;position:sticky;top:0;}
 .token-cell{font-family:'Noto Sans Mono',monospace;font-weight:600;}
 .feature-list{font-size:0.85em;}
 .feature-item{display:inline-block;margin:2px 4px 2px 0;padding:2px 6px;background:#e3f2fd;border:1px solid #90caf9;border-radius:4px;white-space:nowrap;}
 .feature-item:hover{background:#bbdefb;cursor:pointer;}
 .feature-id{color:#1976d2;font-weight:600;}
 .feature-act{color:#666;margin-left:4px;}
 .prompt-display{background:#f5f5f5;padding:12px;border-left:4px solid #0066cc;margin-bottom:12px;border-radius:4px;white-space:pre-wrap;}
</style>
</head><body>
<div class="container">
<h1>🔬 SAE Feature Generation</h1>
<div class="subtitle">Generate text and visualize SAE feature activations at each token</div>
<div class="nav">
  <a href="/">← Back to MAE Explorer</a>
</div>

<div class="panel">
  <h3>Generation Settings</h3>
  <form id="genForm" onsubmit="generate(event)">
    <div class="form-row">
      <label for="prompt">User Prompt:</label>
      <textarea id="prompt" required placeholder="Enter your prompt here...">The cat sat on the</textarea>
    </div>
    <div class="form-row">
      <label for="assistant_prefill">Assistant Prefill (optional):</label>
      <textarea id="assistant_prefill" placeholder="Start the assistant's response with specific text..."></textarea>
    </div>
    <div class="form-row-inline">
      <div>
        <label for="layer">Layer:</label>
        <select id="layer">
          <option value="3">3</option>
          <option value="7">7</option>
          <option value="11" selected>11</option>
          <option value="15">15</option>
          <option value="19">19</option>
          <option value="23">23</option>
        </select>
      </div>
      <div>
        <label for="trainer">Trainer:</label>
        <select id="trainer">
          <option value="0" selected>0 (k=64)</option>
          <option value="1">1 (k=128)</option>
        </select>
      </div>
      <div>
        <label for="max_tokens">Max Tokens:</label>
        <input type="number" id="max_tokens" value="30" min="1" max="1000">
      </div>
      <div>
        <label for="temperature">Temperature:</label>
        <input type="number" id="temperature" value="1.0" min="0" max="2" step="0.1">
      </div>
      <div>
        <label for="top_k_features">Top-K Features:</label>
        <input type="number" id="top_k_features" value="15" min="1" max="50">
      </div>
      <div>
        <label for="reasoning_effort">Reasoning Effort:</label>
        <select id="reasoning_effort">
          <option value="low">Low</option>
          <option value="medium" selected>Medium</option>
          <option value="high">High</option>
        </select>
      </div>
    </div>
    <div class="form-row">
      <button type="submit" id="genButton">Generate</button>
    </div>
  </form>
  <div id="status"></div>
</div>

<div id="results"></div>
</div>

<script>
async function generate(event) {
  event.preventDefault();
  
  const button = document.getElementById('genButton');
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  
  button.disabled = true;
  status.className = 'loading';
  status.textContent = 'Generating... this may take a minute...';
  results.innerHTML = '';
  
  const payload = {
    prompt: document.getElementById('prompt').value,
    assistant_prefill: document.getElementById('assistant_prefill').value,
    layer: parseInt(document.getElementById('layer').value),
    trainer: parseInt(document.getElementById('trainer').value),
    max_new_tokens: parseInt(document.getElementById('max_tokens').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    top_k_features: parseInt(document.getElementById('top_k_features').value),
    reasoning_effort: document.getElementById('reasoning_effort').value,
  };
  
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Generation failed');
    }
    
    const data = await response.json();
    displayResults(data);
    
    status.className = 'success';
    status.textContent = 'Generation complete!';
  } catch (error) {
    status.className = 'error';
    status.textContent = 'Error: ' + error.message;
  } finally {
    button.disabled = false;
  }
}

function displayResults(data) {
  const results = document.getElementById('results');
  
  let html = `
    <div class="result-header">
      <h3>Generation Results</h3>
      <div class="meta">Layer: ${data.layer} | Trainer: ${data.trainer} | Tokens: ${data.generated_tokens.length} | Reasoning: ${data.reasoning_effort}</div>
    </div>
    
    <div class="panel">
      <h3>User Prompt</h3>
      <div class="prompt-display">${escapeHtml(data.prompt_text)}</div>
    </div>
    
    <div class="panel">
      <h3>Formatted Prompt (Harmony Format)</h3>
      <div class="prompt-display" style="border-left-color:#28a745;">${escapeHtml(data.formatted_prompt)}</div>
    </div>
    
    ${data.assistant_prefill ? `
    <div class="panel">
      <h3>Assistant Prefill</h3>
      <div class="prompt-display" style="border-left-color:#9c27b0;">${escapeHtml(data.assistant_prefill)}</div>
    </div>
    ` : ''}
    
    <div class="panel">
      <h3>Generated Text</h3>
      <div class="generated-text">${escapeHtml(data.generated_text)}</div>
    </div>
    
    <div class="panel">
      <h3>Token-by-Token Features</h3>
      <table class="token-table">
        <thead>
          <tr>
            <th>Position</th>
            <th>Token</th>
            <th>Top Features (ID: activation)</th>
          </tr>
        </thead>
        <tbody>
  `;
  
  data.features_per_token.forEach((tokenData, idx) => {
    const featuresHtml = tokenData.top_features.map(f => 
      `<span class="feature-item" onclick="viewFeature(${f.feature_id})" title="Feature ${f.feature_id}: ${f.activation.toFixed(3)}">` +
      `<span class="feature-id">${f.feature_id}</span>` +
      `<span class="feature-act">${f.activation.toFixed(2)}</span>` +
      `</span>`
    ).join('');
    
    html += `
      <tr>
        <td>${idx}</td>
        <td class="token-cell">${escapeHtml(tokenData.token_text)}</td>
        <td class="feature-list">${featuresHtml}</td>
      </tr>
    `;
  });
  
  html += `
        </tbody>
      </table>
    </div>
  `;
  
  results.innerHTML = html;
}

function viewFeature(featureId) {
  const layer = document.getElementById('layer').value;
  const trainer = document.getElementById('trainer').value;
  const url = `/?model=gpt&layer=${layer}&trainer=${trainer}&fids=${featureId}`;
  window.open(url, '_blank');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
</script>
</body></html>
"""
    return Response(page, mimetype="text/html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """API endpoint for generation with feature tracking."""
    try:
        from .generation_analyzer import get_analyzer
        
        data = request.json
        prompt = data.get("prompt", "")
        assistant_prefill = data.get("assistant_prefill", "")
        layer = int(data.get("layer", 11))
        trainer = int(data.get("trainer", 0))
        max_new_tokens = int(data.get("max_new_tokens", 50))
        temperature = float(data.get("temperature", 1.0))
        top_k_features = int(data.get("top_k_features", 20))
        reasoning_effort = data.get("reasoning_effort", "medium")
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Get analyzer instance
        analyzer = get_analyzer(model_name=MODEL_NAME)
        
        # Generate with features using OpenAI Harmony Response Format
        result = analyzer.generate_with_features(
            prompt=prompt,
            assistant_prefill=assistant_prefill,
            layer=layer,
            trainer=trainer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k_features=top_k_features,
            reasoning_effort=reasoning_effort,
        )
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def run_server(model_specs: dict, default_layer: int = 11, default_trainer: int = 0, port: int = 7863):
    init_app(model_specs=model_specs, default_layer=default_layer, default_trainer=default_trainer)
    print(f" * running on http://0.0.0.0:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


