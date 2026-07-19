

import json
import os
import re

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expansion_cache.json")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_VARIANTS = 3

_model = None
_tokenizer = None
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _save_cache():
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f, indent=2, ensure_ascii=False)


def _load_model():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading expansion model {MODEL_NAME} (first cache miss)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _model, _tokenizer


PROMPT = (
    "You rewrite web search queries to improve retrieval over research papers and technical docs.\n"
    "Output {n} alternative queries, one per line:\n"
    "- expand acronyms/abbreviations to full form\n"
    "- fix spelling mistakes\n"
    "- if the query describes a concept indirectly, NAME the concept or the seminal paper\n"
    "Output ONLY the {n} queries, no numbering, no explanations.\n\n"
    "Query: rmse metric\n"
    "root mean squared error\n"
    "RMSE regression evaluation metric\n"
    "root mean squared error formula\n\n"
    "Query: why computers bad at chess but good at math\n"
    "Moravec's paradox\n"
    "computational complexity of chess versus arithmetic\n"
    "why hard problems easy for computers and easy problems hard\n\n"
    "Query: {q}"
)



def expand_query(q: str, n: int = N_VARIANTS) -> list[str]:
    """Return up to `n` alternative queries (never includes the original)."""
    cache = _load_cache()
    key = q.strip().lower()
    if key in cache:
        return cache[key]

    model, tokenizer = _load_model()
    messages = [{"role": "user", "content": PROMPT.format(n=n, q=q)}]
    text_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    variants = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*\d.)\s]+", "", line).strip()  # strip bullets/numbering
        if line and line.lower() != key and line.lower() not in [v.lower() for v in variants]:
            variants.append(line)
    variants = variants[:n]

    cache[key] = variants
    _save_cache()
    return variants
