import json
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "chat_history.json"
PARAMS_FILE = Path(__file__).parent / "main_params.json"


DEFAULT_PARAMS = {
    "model_select": "Bom",
    "max_new_tokens": 60.0,
    "do_sample": False,
    "num_beams": 4.0,
    "repetition_penalty": 2.0,
    "no_repeat_ngram_size": 4.0,
    "temperature": 0.7,
    "top_p": 0.9
}

# ============= Mensagens =============

def load_messages() -> list:
    """ Carrega as mensagens do disco """

    if not HISTORY_FILE.exists():
        return []
    try:
        with  HISTORY_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # se o arquivo estiver corrompido retorna []
        return [] 


def save_messages(messages):
    """Salva as mensagens para o historico de mensagens"""
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    tmp.replace(HISTORY_FILE) 


def clear_messages():
    """Apaga todo o historico de mensagens"""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


# ============= Parâmetros =============

def load_params() -> list:
    """ Carrega dos parâmetros do disco """

    if not PARAMS_FILE.exists():
        return DEFAULT_PARAMS.copy() 
    try:
        with  PARAMS_FILE.open("r", encoding="utf-8") as f:
            saved = json.load(f)

        # Mesclagem com os Defaults (para que as novas chaves apareçam)
        return {**DEFAULT_PARAMS, **saved}
        
    except (json.JSONDecodeError, OSError):
        # caso algum dos parâmetros esteja corrompido irá retornar os parâmetros padrão
        return DEFAULT_PARAMS.copy() 


def save_params(params):
    """sobrescrita dos parâmetros default com novos parâmetros"""
    tmp = PARAMS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    tmp.replace(PARAMS_FILE) 


