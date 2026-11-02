#!/usr/bin/env python3
"""
Multi-Model Lab v2 — ask Grok, Gemini, Claude and ChatGPT at once.

Endpoints:
  GET  /        -> web UI (ask once, see all four answers side by side)
  GET  /health  -> {"ok": true}
  POST /ask     -> JSON {"question": "...", "mode": "compare"|"debate",
                         "models": ["grok","gemini","claude","chatgpt"]}
                   Header: X-API-Key: <BRIDGE_API_KEY>
                   Returns {"answers": {model: text}, "errors": {model: reason}}

Env vars (never hardcode keys):
  BRIDGE_API_KEY    - password protecting this app itself
  XAI_API_KEY       - xAI console (console.x.ai) — Grok, paid per token
  GEMINI_API_KEY    - Google AI Studio (aistudio.google.com) — free tier
  ANTHROPIC_API_KEY - Anthropic console (console.anthropic.com) — Claude, paid
  OPENAI_API_KEY    - OpenAI platform (platform.openai.com) — ChatGPT, paid
  GROK_MODEL        - default: grok-4.7
  GEMINI_MODEL      - default: gemini-2.0-flash
  CLAUDE_MODEL      - default: claude-sonnet-5-5
  OPENAI_MODEL      - default: gpt-6-luna
  PORT              - default: 8000

Stdlib only. Run:  python app.py
"""

import json
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BRIDGE_API_KEY = os.environ.get("BRIDGE_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4.7")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
PORT = int(os.environ.get("PORT", "8000"))
TIMEOUT = 120


def _post(url, payload, headers):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat_openai_compatible(url, api_key, model, question, extra_headers=None):
    data = _post(
        url,
        {"model": model, "messages": [{"role": "user", "content": question}]},
        {"Authorization": f"Bearer {api_key}", **(extra_headers or {})},
    )
    return data["choices"][0]["message"]["content"].strip()


def ask_grok(question):
    # xAI — OpenAI-compatible endpoint
    return _chat_openai_compatible(
        "https://api.x.ai/v1/chat/completions", XAI_API_KEY, GROK_MODEL, question
    )


def ask_gemini(question):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    data = _post(url, {"contents": [{"parts": [{"text": question}]}]}, {})
    return "".join(
        part.get("text", "")
        for c in data.get("candidates", [])
        for part in c.get("content", {}).get("parts", [])
    ).strip()


def ask_claude(question):
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {"model": CLAUDE_MODEL, "max_tokens": 2048,
         "messages": [{"role": "user", "content": question}]},
        {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
    )
    return "".join(
        b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
    ).strip()


def ask_chatgpt(question):
    return _chat_openai_compatible(
        "https://api.openai.com/v1/chat/completions",
        OPENAI_API_KEY, OPENAI_MODEL, question,
    )


MODEL_FNS = {
    "grok": (ask_grok, "XAI_API_KEY"),
    "gemini": (ask_gemini, "GEMINI_API_KEY"),
    "claude": (ask_claude, "ANTHROPIC_API_KEY"),
    "chatgpt": (ask_chatgpt, "OPENAI_API_KEY"),
}
ALL_MODELS = list(MODEL_FNS)


def run_compare(question, models):
    answers, errors = {}, {}
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futs = {pool.submit(MODEL_FNS[m][0], question): m for m in models if m in MODEL_FNS}
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                answers[m] = fut.result()
            except Exception as e:  # noqa: BLE001 - report per-model failure
                errors[m] = f"{type(e).__name__}: {e}"
    for m in models:
        if m not in MODEL_FNS:
            errors[m] = "unknown model"
        elif m not in answers and m not in errors:
            errors[m] = f"missing {MODEL_FNS[m][1]} on server"
    return answers, errors


def run_debate(question, models):
    """Round 1: each model answers. Round 2: each model critiques the others."""
    answers, errors = run_compare(question, models)
    if len(answers) < 2:
        return answers, errors
    others = "\n\n".join(f"[{m.upper()}]:\n{a}" for m, a in answers.items())
    verdicts, verrs = {}, {}
    with ThreadPoolExecutor(max_workers=len(answers)) as pool:
        futs = {}
        for m in answers:
            prompt = (
                f"Question: {question}\n\nOther AI models answered like this:\n\n{others}\n\n"
                "Briefly critique the other answers (point out any errors), then give "
                "your own final answer to the question."
            )
            futs[pool.submit(MODEL_FNS[m][0], prompt)] = m
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                verdicts[m] = fut.result()
            except Exception as e:  # noqa: BLE001
                verrs[m] = f"{type(e).__name__}: {e}"
    errors.update({f"{m}_debate": e for m, e in verrs.items()})
    return {"round1": answers, "round2_critiques": verdicts}, errors


UI_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Model Lab</title>
<style>body{font-family:system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:90px}button{padding:.6rem 1.2rem;margin:.5rem .25rem .5rem 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
.card{border:1px solid #ccc;border-radius:8px;padding:1rem;white-space:pre-wrap}
h3{margin:.2rem 0}.err{color:#a00}.models{margin:.5rem 0}
.models label{margin-right:1rem}</style></head><body>
<h1>Multi-Model Lab</h1>
<p>Ask once — Grok, Gemini, Claude and ChatGPT answer together.</p>
<textarea id="q" placeholder="Type your question..."></textarea><br>
<div class="models" id="models"></div>
<label><input type="checkbox" id="debate"> Debate mode (models critique each other)</label><br>
<button onclick="ask()">Ask all models</button>
<div id="out"></div>
<script>
const MODELS = ['grok','gemini','claude','chatgpt'];
document.getElementById('models').innerHTML = MODELS.map(m =>
  `<label><input type="checkbox" class="m" value="${m}" checked> ${m}</label>`).join('');
let key = sessionStorage.getItem('mmk') || '';
async function ask(){
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  const models = [...document.querySelectorAll('.m:checked')].map(c=>c.value);
  if(!models.length){ alert('Pick at least one model'); return; }
  if(!key){ key = prompt('Bridge API key (the BRIDGE_API_KEY you set on the server):') || ''; sessionStorage.setItem('mmk', key); }
  const out = document.getElementById('out'); out.innerHTML = '<p>Thinking...</p>';
  try{
    const r = await fetch('/ask',{method:'POST',
      headers:{'Content-Type':'application/json','X-API-Key':key},
      body:JSON.stringify({question:q, models:models,
        mode:document.getElementById('debate').checked?'debate':'compare'})});
    const d = await r.json();
    if(!r.ok){ out.innerHTML = '<p class=err>'+escapeHtml(d.error||r.status)+'</p>';
      if(r.status===401){sessionStorage.removeItem('mmk');key='';} return; }
    let html = '';
    const show = (obj)=>{ let h='<div class=grid>'; for(const m in obj){
      h += '<div class=card><h3>'+m+'</h3>'+escapeHtml(obj[m])+'</div>'; } return h+'</div>'; };
    if(d.mode==='debate'){ html+='<h2>Round 1 — answers</h2>'+show(d.answers.round1||{});
      html+='<h2>Round 2 — critiques</h2>'+show(d.answers.round2_critiques||{}); }
    else html = show(d.answers||{});
    for(const m in (d.errors||{})){ html += '<div class="card err"><h3>'+m+'</h3>'+escapeHtml(d.errors[m])+'</div>'; }
    out.innerHTML = html || '<p class=err>No answers returned.</p>';
  }catch(e){ out.innerHTML = '<p class=err>'+escapeHtml(e)+'</p>'; }
}
function escapeHtml(s){ return String(s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "models": ALL_MODELS})
        elif self.path in ("/", "/index.html"):
            self._send(200, UI_HTML.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, {"error": "not found"})
            return
        if not BRIDGE_API_KEY or self.headers.get("X-API-Key") != BRIDGE_API_KEY:
            self._send(401, {"error": "bad or missing X-API-Key"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send(400, {"error": "invalid JSON"})
            return
        question = (payload.get("question") or "").strip()
        if not question:
            self._send(400, {"error": "question is required"})
            return
        mode = payload.get("mode", "compare")
        models = payload.get("models") or ALL_MODELS
        try:
            if mode == "debate":
                answers, errors = run_debate(question, models)
            else:
                answers, errors = run_compare(question, models)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return
        self._send(200, {"answers": answers, "errors": errors, "mode": mode})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Multi-Model Lab v2 on port {PORT} | models: {', '.join(ALL_MODELS)}")
    for name, (_, env) in MODEL_FNS.items():
        if not os.environ.get(env):
            print(f"  [warn] {env} not set - {name} will report 'missing key'")
    if not BRIDGE_API_KEY:
        print("  [warn] BRIDGE_API_KEY not set - /ask will reject everything")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
