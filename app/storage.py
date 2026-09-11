import json
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "chat_history.json"


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