# ============ Importação de Bibliotecas ============
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import streamlit as st
import time

from storage import (
    load_messages, save_messages, clear_messages,
    load_params, save_params, DEFAULT_PARAMS,
)
from app_functions import (
    load_config, resolve_path, load_model, respond,
    reload_models, adapter_fingerprint, cached_models_info,
)


# ============ Configuração Inicial ============
st.set_page_config(
    page_title="Comparativo de Assistentes de IA (Saúde)",
    layout="wide",
)

st.title("Vital AI (Vitalia)")
st.write("Avalie os Modelos — Ruim · Bom · Ótimo")


# ============ Config global ============
CFG             = load_config()
MODEL_OPTIONS   = list(CFG["models"].keys())
MODEL_EPOCHS    = CFG.get("model_epochs", {})
PROMPT_TEMPLATE = CFG["prompt_template"]


def _fmt_model(m: str) -> str:
    ep = MODEL_EPOCHS.get(m)
    return f"{m} (época {ep})" if ep else m


# =============== Inicialização dos Estados ===============
if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

if "saved_params" not in st.session_state:
    st.session_state.saved_params = load_params()

if "last_loaded" not in st.session_state:
    st.session_state.last_loaded = None


def _return_param(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


# =============== Lista de Parâmetros ===============
param_list = st.session_state.saved_params
column_list = [
    "model_select", "max_new_tokens", "do_sample", "num_beams",
    "repetition_penalty", "no_repeat_ngram_size", "temperature",
    "top_p", "top_k", "length_penalty", "early_stopping", "seed",
    "cut_first_sentence",
]


# =============== SideBar ===============
with st.sidebar:
    st.title("Menu Principal")
    st.header("Parâmetros")

    for c in column_list:
        _return_param(c, param_list[c])

    # ---------- Modelo ----------
    st.selectbox(
        "Escolha seu Modelo",
        MODEL_OPTIONS,
        format_func=_fmt_model,
        key="model_select",
    )

    # ---------- Comprimento ----------
    st.subheader("📏 Comprimento")
    st.number_input(
        "Máx. de tokens novos",
        min_value=10, max_value=500,
        step=10, key="max_new_tokens",
        help="Notebook usava 70. Valores altos permitem respostas longas.",
    )
    st.checkbox(
        "Cortar na 1ª frase",
        key="cut_first_sentence",
        help="Fiel ao notebook. Desmarque para ver o texto completo "
             "(recomendado quando max_new_tokens > 120).",
    )

    # ---------- Estratégia de busca ----------
    st.subheader("🔎 Estratégia de busca")
    st.checkbox(
        "Usar amostragem (do_sample)",
        key="do_sample",
        help="False = determinístico (beam/greedy). True = estocástico.",
    )
    st.number_input(
        "Número de Hipóteses (Beams)",
        min_value=1, max_value=15,
        step=1, key="num_beams",
        help="Só faz sentido com do_sample=False. Notebook usava 4.",
    )
    st.number_input(
        "Length penalty",
        min_value=0.1, max_value=3.0,
        step=0.05, format="%.2f", key="length_penalty",
        help="Só com beam search. >1.0 favorece respostas longas.",
    )
    st.checkbox(
        "Early stopping no beam search",
        key="early_stopping",
        help="Só com beam search. Para quando encontra o EOS.",
    )

    # ---------- Repetição ----------
    st.subheader("🔁 Repetição")
    st.number_input(
        "Penalidade de Repetição",
        min_value=1.0, max_value=3.0,
        step=0.05, format="%.2f", key="repetition_penalty",
        help="Notebook usava 2.2. Valores muito altos degradam a saída.",
    )
    st.number_input(
        "n-gramas sem repetição",
        min_value=0, max_value=10,
        step=1, key="no_repeat_ngram_size",
        help="Notebook usava 4. Bloqueia repetição de n-gramas.",
    )

    # ---------- Sampling ----------
    st.subheader("🎲 Sampling")
    st.number_input(
        "Temperatura",
        min_value=0.1, max_value=2.0,
        step=0.05, format="%.2f", key="temperature",
        help="Só com do_sample=True.",
    )
    st.slider(
        "Top-P (nucleus)",
        min_value=0.1, max_value=1.0,
        step=0.05, key="top_p",
        help="Só com do_sample=True.",
    )
    st.number_input(
        "Top-K",
        min_value=0, max_value=200,
        step=5, key="top_k",
        help="Só com do_sample=True. 0 = desativado.",
    )

    # ---------- Reprodutibilidade ----------
    st.subheader("🌱 Reprodutibilidade")
    st.number_input(
        "Seed",
        min_value=0, max_value=2**31 - 1,
        step=1, key="seed",
        help="Mesmo valor = mesma saída (com do_sample=True).",
    )

    # ---------- Avisos ----------
    if st.session_state["do_sample"] and st.session_state["num_beams"] > 1:
        st.warning(
            "⚠️ `do_sample=True` **com** `num_beams>1` ativa *sampling de beam "
            "search*. Considere `num_beams=1` se quiser só amostragem."
        )
    if (not st.session_state["do_sample"]) and (
        st.session_state["temperature"] != 0.7
        or st.session_state["top_p"] != 0.9
        or st.session_state["top_k"] != 50
    ):
        st.info("ℹ️ `temperature`, `top_p` e `top_k` só têm efeito com `do_sample=True`.")

    st.divider()

    # ---------- Botões ----------
    col1, col2 = st.columns([2, 1])
    with col1:
        params_button = st.button("💾 Salvar Parâmetros", use_container_width=True)
    with col2:
        params_reset_button = st.button("↺ Resetar", use_container_width=True)

    if st.button("🔄 Recarregar modelos do disco", use_container_width=True):
        reload_models()
        st.session_state.last_loaded = None
        st.success("Cache limpo. A próxima pergunta recarrega do disco.")

    # ---------- Exportar parâmetros ----------
    with st.expander("📋 Exportar parâmetros atuais (JSON)"):
        st.code(
            json.dumps(
                {k: st.session_state[k] for k in column_list},
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )

    # ---------- Debug ----------
    st.divider()
    with st.expander("🔍 Debug — modelos em cache", expanded=False):
        st.caption("Modelo selecionado:")
        st.code(_fmt_model(st.session_state["model_select"]), language=None)

        st.caption("Caminho do adaptador:")
        _rel = CFG["models"][st.session_state["model_select"]]
        st.code(resolve_path(_rel), language=None)

        st.caption("Última resposta gerada por:")
        st.code(st.session_state.last_loaded or "(nenhuma ainda)", language=None)

        st.caption("Modelos em memória:")
        infos = cached_models_info()
        if not infos:
            st.code("(cache vazio)", language=None)
        else:
            for i in infos:
                st.markdown(
                    f"**{i['key']}**  \n"
                    f"`path:` {i['adapter_path']}  \n"
                    f"`device:` {i['device']}  \n"
                    f"`fingerprint:` `{i['fingerprint']}`"
                )


# =============== Salvar / resetar parâmetros ===============
if params_button:
    new_params = {
        "model_select":         st.session_state["model_select"],
        "max_new_tokens":       int(st.session_state["max_new_tokens"]),
        "do_sample":            bool(st.session_state["do_sample"]),
        "num_beams":            int(st.session_state["num_beams"]),
        "repetition_penalty":   float(st.session_state["repetition_penalty"]),
        "no_repeat_ngram_size": int(st.session_state["no_repeat_ngram_size"]),
        "temperature":          float(st.session_state["temperature"]),
        "top_p":                float(st.session_state["top_p"]),
        "top_k":                int(st.session_state["top_k"]),
        "length_penalty":       float(st.session_state["length_penalty"]),
        "early_stopping":       bool(st.session_state["early_stopping"]),
        "seed":                 int(st.session_state["seed"]),
        "cut_first_sentence":   bool(st.session_state["cut_first_sentence"]),
    }
    save_params(new_params)
    st.session_state.saved_params = new_params

    alerta = st.success("Parâmetros salvos!")
    time.sleep(2)
    alerta.empty()

elif params_reset_button:
    save_params(DEFAULT_PARAMS.copy())
    st.session_state.saved_params = DEFAULT_PARAMS.copy()
    for c in column_list:
        st.session_state.pop(c, None)
    st.rerun()


# =============== Chat ===============
st.header("💬 Discussão com o Agente")

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Limpar Histórico", use_container_width=True):
        st.session_state.messages = []
        clear_messages()
        st.rerun()

chat_container = st.container(height=500, autoscroll=True)
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                st.caption(msg["meta"])

if prompt := st.chat_input("Escreva Sua Pergunta"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    adapter_rel  = CFG["models"][st.session_state["model_select"]]
    adapter_path = resolve_path(adapter_rel)

    try:
        with st.spinner(
            f"Carregando '{_fmt_model(st.session_state['model_select'])}'..."
        ):
            model, tokenizer, device = load_model(
                st.session_state["model_select"],
                adapter_path,
                CFG["base_model"],
            )
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    fp = adapter_fingerprint(model)
    label = (
        f"🤖 **{_fmt_model(st.session_state['model_select'])}** · "
        f"device=`{device}` · fp=`{fp}` · "
        f"`tokens={st.session_state['max_new_tokens']}` · "
        f"`beams={st.session_state['num_beams']}` · "
        f"`sample={st.session_state['do_sample']}` · "
        f"`cut={st.session_state['cut_first_sentence']}`"
    )
    st.session_state.last_loaded = label

    with st.chat_message("assistant"):
        with st.spinner("Gerando resposta..."):
            resposta = respond(
                model, tokenizer, device,
                instruction=prompt,
                prompt_template=PROMPT_TEMPLATE,
                max_new_tokens=int(st.session_state["max_new_tokens"]),
                do_sample=bool(st.session_state["do_sample"]),
                num_beams=int(st.session_state["num_beams"]),
                repetition_penalty=float(st.session_state["repetition_penalty"]),
                no_repeat_ngram_size=int(st.session_state["no_repeat_ngram_size"]),
                temperature=float(st.session_state["temperature"]),
                top_p=float(st.session_state["top_p"]),
                top_k=int(st.session_state["top_k"]),
                length_penalty=float(st.session_state["length_penalty"]),
                early_stopping=bool(st.session_state["early_stopping"]),
                seed=int(st.session_state["seed"]),
                cut_first_sentence=bool(st.session_state["cut_first_sentence"]),
            )
        st.markdown(resposta)
        st.caption(label)

    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta,
        "meta": label,
    })
    save_messages(st.session_state.messages)
    st.rerun()