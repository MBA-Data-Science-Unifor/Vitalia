import json
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "chat_history.json"
PARAMS_FILE  = Path(__file__).parent / "main_params.json"


DEFAULT_PARAMS = {
    "model_select":         "Bom",
    "max_new_tokens":       70,
    "do_sample":            False,
    "num_beams":            4,
    "repetition_penalty":   2.2,
    "no_repeat_ngram_size": 4,
    "temperature":          0.7,
    "top_p":                0.9,
    "top_k":                50,
    "length_penalty":       1.0,
    "early_stopping":       False,
    "seed":                 42,
    "cut_first_sentence":   True,
}


# ============= Mensagens =============

def load_messages() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_messages(messages):
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    tmp.replace(HISTORY_FILE)


def clear_messages():
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


# ============= Parâmetros =============

def load_params() -> dict:
    if not PARAMS_FILE.exists():
        return DEFAULT_PARAMS.copy()
    try:
        with PARAMS_FILE.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        return {**DEFAULT_PARAMS, **saved}
    except (json.JSONDecodeError, OSError):
        return DEFAULT_PARAMS.copy()


def save_params(params):
    tmp = PARAMS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    tmp.replace(PARAMS_FILE)