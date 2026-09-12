"""
Funções auxiliares do app Vitalia.

Espelha a lógica do notebook model_2.ipynb, com parâmetros de geração estendidos:
- carregar_modelo()  →  load_model()
- carregar_epoca()   →  load_model() com adapter_path de época
- responder()        →  respond()
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import logging
import re

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
    """Carrega o config.yaml da raiz do projeto."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel_path: str) -> str:
    """Converte caminho relativo (do config) em absoluto."""
    p = Path(rel_path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


# ============================================================
# Device / dtype
# ============================================================
def _resolve_device():
    """Retorna (device, dtype) — cuda+fp16 se disponível, senão cpu+fp32."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


# ============================================================
# Fingerprint do adaptador
# ============================================================
def adapter_fingerprint(model) -> str:
    """
    Hash curto dos pesos LoRA. Épocas diferentes do mesmo treino
    geram fingerprints DIFERENTES — prova de que o modelo foi trocado.
    """
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
# Carregamento do modelo
# ============================================================
@lru_cache(maxsize=3)
def load_model(model_key: str, adapter_path: str, base_name: str):
    """
    Carrega e cacheia base + adaptador LoRA.

    Parameters
    ----------
    model_key     : rótulo lógico ("Ruim", "Bom", "Ótimo") — só chave de cache.
    adapter_path  : caminho da pasta do adaptador (com adapter_config.json).
    base_name     : nome do modelo base no HF Hub.

    Returns
    -------
    (model, tokenizer, device)
    """
    adapter = Path(adapter_path)

    logger.info("=" * 64)
    logger.info("load_model() — %s", model_key)
    logger.info("  adapter_path : %s", adapter)
    logger.info("  base_name    : %s", base_name)

    # ---- validações amigáveis ----
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

    # ---- lê o adapter_config.json para logar ----
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

    # ---- carrega base + adaptador ----
    device, dtype = _resolve_device()
    logger.info("  device=%s  dtype=%s", device, dtype)

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.to(device)
    model.eval()

    # ---- validação: o LoRA foi aplicado? ----
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
    """Limpa o cache de modelos. Útil após editar config ou adaptadores."""
    logger.info("♻️  reload_models(): cache limpo")
    load_model.cache_clear()


def cached_models_info() -> list[dict]:
    """Lista o que está em memória (para o painel de debug)."""
    info = []
    cache = getattr(load_model, "cache", None)
    if cache is None:
        return info
    try:
        for (mk, ap, bn), (model, _tok, dev) in cache.items():
            info.append({
                "key":         mk,
                "adapter_path": ap,
                "base_name":   bn,
                "device":      str(dev),
                "fingerprint": adapter_fingerprint(model),
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
    # --- tamanho ---
    min_new_tokens: int = 1,
    max_new_tokens: int = 70,
    length_penalty: float = 1.0,
    # --- decodificação ---
    do_sample: bool = False,
    num_beams: int = 4,
    early_stopping: bool = False,
    num_return_sequences: int = 1,
    # --- amostragem ---
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    # --- anti-repetição ---
    repetition_penalty: float = 2.2,
    no_repeat_ngram_size: int = 4,
) -> str:
    """
    Gera uma resposta.

    Retorna somente a primeira sequência gerada, cortada na
    primeira frase completa (terminada em '.', '!' ou '?').
    """

    # ---- validações defensivas ----
    if min_new_tokens > max_new_tokens:
        logger.warning(
            "min_new_tokens (%d) > max_new_tokens (%d) — usando min=1",
            min_new_tokens, max_new_tokens,
        )
        min_new_tokens = 1

    if num_return_sequences > 1 and not do_sample and num_beams == 1:
        logger.warning(
            "num_return_sequences>1 exige do_sample=True ou num_beams>1. "
            "Reduzindo para 1."
        )
        num_return_sequences = 1

    # ---- prompt ----
    prefixo = f"### Contexto:\n{context}\n\n" if context.strip() else ""
    prompt  = prefixo + prompt_template.format(instruction=instruction)

    logger.info("respond(): prompt >>>\n%s\n<<< (fim do prompt)", prompt)
    logger.info(
        "respond(): min=%s max=%s len_pen=%.2f | sample=%s beams=%s "
        "early_stop=%s n_ret=%s | temp=%.2f top_p=%.2f top_k=%s | "
        "rep=%.2f no_rep=%s",
        min_new_tokens, max_new_tokens, length_penalty,
        do_sample, num_beams, early_stopping, num_return_sequences,
        temperature, top_p, top_k,
        repetition_penalty, no_repeat_ngram_size,
    )

    # ---- tokeniza ----
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ---- kwargs de geração ----
    gen_kwargs = dict(
        min_new_tokens=int(min_new_tokens),
        max_new_tokens=int(max_new_tokens),
        length_penalty=float(length_penalty),
        do_sample=bool(do_sample),
        num_beams=int(num_beams),
        early_stopping=bool(early_stopping),
        num_return_sequences=int(num_return_sequences),
        repetition_penalty=float(repetition_penalty),
        no_repeat_ngram_size=int(no_repeat_ngram_size),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"]       = float(top_p)
        if top_k and int(top_k) > 0:
            gen_kwargs["top_k"] = int(top_k)

    # ---- geração ----
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # ---- decodifica a primeira sequência ----
    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Resposta:" in texto:
        texto = texto.split("### Resposta:")[-1].strip()
    else:
        texto = texto.strip()

    logger.info("respond(): texto bruto = %r", texto)

    # ---- corta na primeira frase completa ----
    m = re.search(r"^([^.!?]*[.!?])", texto)
    resposta = m.group(1).strip() if m else texto

    logger.info("respond(): resposta final = %r", resposta)
    return resposta


# ============================================================
# Utilitário legado
# ============================================================
def load_dataset(st):
    return st.file_uploader("Escolha sua base de dados", type="jsonl")