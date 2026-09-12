import json
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "chat_history.json"
PARAMS_FILE  = Path(__file__).parent / "main_params.json"


# Defaults espelham o bloco generation: do notebook model_2.ipynb
DEFAULT_PARAMS = {
    "model_select":         "Bom",
    # --- tamanho da resposta ---
    "min_new_tokens":       1,
    "max_new_tokens":       70,
    "length_penalty":       1.0,
    # --- decodificação ---
    "do_sample":            False,
    "num_beams":            4,
    "early_stopping":       False,
    "num_return_sequences": 1,
    # --- amostragem (só efeito se do_sample=True) ---
    "temperature":          0.7,
    "top_p":                0.9,
    "top_k":                0,
    # --- anti-repetição ---
    "repetition_penalty":   2.2,
    "no_repeat_ngram_size": 4,
}


# ============= Mensagens =============

def load_messages() -> list:
    """Carrega as mensagens do disco."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_messages(messages):
    """Salva as mensagens no histórico (escrita atômica)."""
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    tmp.replace(HISTORY_FILE)


def clear_messages():
    """Apaga todo o histórico de mensagens."""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


# ============= Parâmetros =============

def load_params() -> dict:
    """Carrega parâmetros do disco, mesclando com os defaults."""
    if not PARAMS_FILE.exists():
        return DEFAULT_PARAMS.copy()
    try:
        with PARAMS_FILE.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        return {**DEFAULT_PARAMS, **saved}
    except (json.JSONDecodeError, OSError):
        return DEFAULT_PARAMS.copy()


def save_params(params):
    """Sobrescreve os parâmetros salvos (escrita atômica)."""
    tmp = PARAMS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    tmp.replace(PARAMS_FILE)