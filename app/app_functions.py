"""
Funções auxiliares do app Vitalia.

Espelha a lógica do notebook model_2.ipynb e adiciona parâmetros
extras para controle fino da geração.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import logging
import random
import re

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================
# Logging
# ============================================================
logger = logging.getLogger("vitalia")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ============================================================
# Caminhos
# ============================================================
CONFIG_PATH  = Path(__file__).resolve().parents[1] / "config.yaml"
PROJECT_ROOT = CONFIG_PATH.parent


# ============================================================
# Config
# ============================================================
def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel_path: str) -> str:
    p = Path(rel_path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


# ============================================================
# Device / dtype
# ============================================================
def _resolve_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


# ============================================================
# Fingerprint do adaptador
# ============================================================
def adapter_fingerprint(model) -> str:
    h = hashlib.md5()
    n_tensors = 0
    for name, param in sorted(model.named_parameters()):
        if "lora_" in name:
            flat = param.detach().cpu().to(torch.float32).flatten()
            h.update(name.encode())
            h.update(flat[:64].numpy().tobytes())
            n_tensors += 1
    return f"{h.hexdigest()[:8]} ({n_tensors} tensores)"


# ============================================================
# Carregamento de modelo
# ============================================================
@lru_cache(maxsize=3)
def load_model(model_key: str, adapter_path: str, base_name: str):
    adapter = Path(adapter_path)

    logger.info("=" * 64)
    logger.info("load_model() — %s", model_key)
    logger.info("  adapter_path : %s", adapter)
    logger.info("  base_name    : %s", base_name)

    if not adapter.exists():
        raise FileNotFoundError(
            f"❌ Pasta do adaptador não existe:\n   {adapter}\n\n"
            f"💡 Verifique o caminho em config.yaml (chave `models`)."
        )

    if not (adapter / "adapter_config.json").exists():
        encontrados = list(adapter.rglob("adapter_config.json"))
        if encontrados:
            raise FileNotFoundError(
                f"❌ adapter_config.json NÃO está em:\n   {adapter}\n\n"
                f"   mas foi encontrado em:\n   {encontrados[0].parent}\n\n"
                f"💡 Ajuste o caminho em config.yaml para esse valor."
            )
        raise FileNotFoundError(
            f"❌ adapter_config.json não encontrado em:\n   {adapter}\n"
            f"   nem em nenhuma subpasta."
        )

    try:
        with (adapter / "adapter_config.json").open("r", encoding="utf-8") as f:
            acfg = json.load(f)
        logger.info(
            "  adapter_config: r=%s  alpha=%s  target=%s  dropout=%s",
            acfg.get("r"), acfg.get("lora_alpha"),
            acfg.get("target_modules"), acfg.get("lora_dropout"),
        )
        bm = acfg.get("base_model_name_or_path")
        if bm and bm != base_name:
            logger.warning(
                "  ⚠️  base_model no adapter_config (%s) difere do config.yaml (%s)",
                bm, base_name,
            )
    except Exception as e:
        logger.warning("  não foi possível ler adapter_config.json: %s", e)

    device, dtype = _resolve_device()
    logger.info("  device=%s  dtype=%s", device, dtype)

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.to(device)
    model.eval()

    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    if n_lora == 0:
        logger.error("  ⚠️  NENHUM parâmetro lora_* encontrado!")
    else:
        logger.info("  ✅ adaptador aplicado — %d tensores lora_*", n_lora)

    logger.info("  active_adapters: %s", model.active_adapters)
    logger.info("  fingerprint: %s", adapter_fingerprint(model))
    logger.info("=" * 64)

    return model, tok, device


def reload_models():
    logger.info("♻️  reload_models(): cache limpo")
    load_model.cache_clear()


def cached_models_info() -> list[dict]:
    info = []
    cache = getattr(load_model, "cache", None)
    if cache is None:
        return info
    try:
        for (mk, ap, bn), (model, _tok, dev) in cache.items():
            info.append({
                "key":          mk,
                "adapter_path": ap,
                "base_name":    bn,
                "device":       str(dev),
                "fingerprint":  adapter_fingerprint(model),
            })
    except Exception:
        pass
    return info


# ============================================================
# Inferência
# ============================================================
def respond(
    model,
    tokenizer,
    device,
    instruction: str,
    *,
    context: str = "",
    prompt_template: str = "### Pergunta:\n{instruction}\n\n### Resposta:\n",
    # --- parâmetros principais ---
    max_new_tokens: int = 70,
    do_sample: bool = False,
    num_beams: int = 4,
    repetition_penalty: float = 2.2,
    no_repeat_ngram_size: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
    # --- parâmetros extras ---
    top_k: int = 50,
    length_penalty: float = 1.0,
    early_stopping: bool = False,
    seed: int | None = None,
    cut_first_sentence: bool = True,
) -> str:
    """Gera uma resposta com os parâmetros configurados."""

    # ---------- reprodutibilidade ----------
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed))

    # ---------- prompt ----------
    prefixo = f"### Contexto:\n{context}\n\n" if context.strip() else ""
    prompt  = prefixo + prompt_template.format(instruction=instruction)

    logger.info("respond(): prompt >>>\n%s\n<<< (fim)", prompt)
    logger.info(
        "respond(): max_new_tokens=%s do_sample=%s num_beams=%s "
        "rep_pen=%s no_repeat=%s temp=%s top_p=%s top_k=%s "
        "length_penalty=%s early_stopping=%s seed=%s cut=%s",
        max_new_tokens, do_sample, num_beams,
        repetition_penalty, no_repeat_ngram_size,
        temperature, top_p, top_k,
        length_penalty, early_stopping, seed, cut_first_sentence,
    )

    ids = tokenizer(prompt, return_tensors="pt").to(device)

    # ---------- kwargs de geração ----------
    gen_kwargs = {
        "max_new_tokens":       int(max_new_tokens),
        "do_sample":            bool(do_sample),
        "num_beams":            int(num_beams),
        "repetition_penalty":   float(repetition_penalty),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "pad_token_id":         tokenizer.eos_token_id,
        "eos_token_id":         tokenizer.eos_token_id,
    }

    # length_penalty / early_stopping só fazem sentido com beam search
    if num_beams > 1:
        gen_kwargs["length_penalty"] = float(length_penalty)
        gen_kwargs["early_stopping"] = bool(early_stopping)

    # temperature / top_p / top_k só fazem sentido com sampling
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"]       = float(top_p)
        if top_k > 0:
            gen_kwargs["top_k"] = int(top_k)

    with torch.no_grad():
        outputs = model.generate(**ids, **gen_kwargs)

    # ---------- decodificação ----------
    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Resposta:" in texto:
        texto = texto.split("### Resposta:")[-1].strip()
    else:
        texto = texto.strip()

    logger.info("respond(): texto bruto = %r", texto)

    # ---------- corte opcional ----------
    if cut_first_sentence:
        m = re.search(r"^([^.!?]*[.!?])", texto)
        resposta = m.group(1).strip() if m else texto
    else:
        resposta = texto

    logger.info("respond(): resposta final = %r", resposta)
    return resposta


# ============================================================
# Utilitário legado
# ============================================================
def load_dataset(st):
    return st.file_uploader("Escolha sua base de dados", type="jsonl")