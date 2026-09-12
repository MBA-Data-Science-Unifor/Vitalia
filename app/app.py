# ============ Importação de Bibliotecas ============
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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

st.title("🩺 Vital AI (Vitalia)")
st.caption(
    "Compare as respostas dos **3 checkpoints** do mesmo treino LoRA "
    "(**Ruim** = época 1 · **Bom** = época 5 · **Ótimo** = época 10)."
)


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
    "model_select",
    "min_new_tokens", "max_new_tokens", "length_penalty",
    "do_sample", "num_beams", "early_stopping", "num_return_sequences",
    "temperature", "top_p", "top_k",
    "repetition_penalty", "no_repeat_ngram_size",
]


# =============== SideBar ===============
with st.sidebar:
    st.title("⚙️ Painel de Controle")
    st.caption("Ajuste os parâmetros e compare os modelos em tempo real.")

    for c in column_list:
        _return_param(c, param_list[c])

    # ============================================================
    # 🧠 MODELO
    # ============================================================
    st.subheader("🧠 Modelo")
    st.selectbox(
        "Checkpoint LoRA",
        MODEL_OPTIONS,
        format_func=_fmt_model,
        key="model_select",
        help=(
            "Cada opção é uma época diferente do mesmo treino LoRA.\n\n"
            "• Ruim (época 1) → underfitting, respostas desconexas\n"
            "• Bom (época 5)  → ponto de equilíbrio (~94% do ganho)\n"
            "• Ótimo (época 10) → menor val_loss, sem overfitting"
        ),
    )

    st.divider()

    # ============================================================
    # 📏 TAMANHO
    # ============================================================
    st.subheader("📏 Tamanho da resposta")

    st.number_input(
        "Mín. de tokens novos",
        min_value=1, max_value=100, step=1,
        key="min_new_tokens",
        help=(
            "Força um tamanho MÍNIMO de tokens antes de permitir <eos>.\n\n"
            "Use para evitar respostas de 1 palavra (ex.: 'Sim.').\n"
            "Se ficar maior que o máximo, é corrigido automaticamente para 1."
        ),
    )

    st.number_input(
        "Máx. de tokens novos",
        min_value=1, max_value=300, step=5,
        key="max_new_tokens",
        help=(
            "Tamanho MÁXIMO da resposta em tokens.\n\n"
            "• 20–40 → respostas diretas\n"
            "• 60–80 → baseline do notebook\n"
            "• 120+  → parágrafos (pode divagar)\n\n"
            "O respond() ainda corta na primeira frase completa."
        ),
    )

    st.number_input(
        "Length penalty",
        min_value=0.0, max_value=3.0, step=0.1, format="%.1f",
        key="length_penalty",
        help=(
            "Ajusta o score das hipóteses no beam search.\n\n"
            "• <1  → favorece respostas curtas\n"
            "• 1.0 → neutro (padrão)\n"
            "• >1  → favorece respostas longas\n\n"
            "⚠️ Só tem efeito quando 'Número de Hipóteses' > 1."
        ),
    )

    st.divider()

    # ============================================================
    # 🎲 DECODIFICAÇÃO
    # ============================================================
    st.subheader("🎲 Decodificação")

    st.checkbox(
        "Usar amostragem (do_sample)",
        key="do_sample",
        help=(
            "Chave mestra entre modo determinístico e estocástico.\n\n"
            "• OFF → mesma pergunta sempre dá a MESMA resposta (reproduz o notebook)\n"
            "• ON  → respostas variam a cada execução\n\n"
            "⚠️ Sem isso, Temperatura / top_p / top_k não têm efeito."
        ),
    )

    st.number_input(
        "Número de Hipóteses (Beams)",
        min_value=1, max_value=10, step=1,
        key="num_beams",
        help=(
            "Beam search: quantos caminhos paralelos explorar.\n\n"
            "• 1 → greedy (rápido, míope)\n"
            "• 4 → equilíbrio (recomendado, default do notebook)\n"
            "• 8+ → marginalmente melhor, mas ~N× mais lento"
        ),
    )

    st.checkbox(
        "Early stopping (beam search)",
        key="early_stopping",
        help=(
            "Para o beam search assim que TODAS as hipóteses emitem <eos>.\n\n"
            "Economiza tempo. Só faz sentido com num_beams > 1.\n"
            "Se as respostas ficarem curtas demais, desligue."
        ),
    )

    st.number_input(
        "Nº de respostas alternativas",
        min_value=1, max_value=5, step=1,
        key="num_return_sequences",
        help=(
            "Gera N respostas diferentes por pergunta.\n\n"
            "⚠️ Precisa de do_sample=True OU num_beams > 1.\n"
            "⚠️ O app atual exibe apenas a 1ª sequência."
        ),
    )

    st.divider()

    # ============================================================
    # 🎨 AMOSTRAGEM
    # ============================================================
    st.subheader("🎨 Amostragem")
    st.caption("🛈 Só têm efeito se **do_sample** estiver marcado.")

    st.number_input(
        "Temperatura",
        min_value=0.1, max_value=2.0, step=0.05, format="%.2f",
        key="temperature",
        help=(
            "Achata ou afina a distribuição de probabilidade.\n\n"
            "• 0.1–0.5 → muito previsível\n"
            "• 0.7     → equilibrado (notebook)\n"
            "• 1.0+    → criativo\n"
            "• 1.5+    → caótico / incoerente"
        ),
    )

    st.slider(
        "Amostragem de Núcleo (top_p)",
        min_value=0.1, max_value=1.0, step=0.05,
        key="top_p",
        help=(
            "Nucleus sampling: mantém o MENOR conjunto de tokens "
            "cuja probabilidade acumulada ≥ p.\n\n"
            "• 0.5 → muito restritivo\n"
            "• 0.9 → padrão (notebook)\n"
            "• 1.0 → sem filtro"
        ),
    )

    st.number_input(
        "Top-K (0 = desligado)",
        min_value=0, max_value=100, step=1,
        key="top_k",
        help=(
            "Limite RÍGIDO: mantém apenas os K tokens mais prováveis.\n\n"
            "Difere do top_p (que escolhe pela soma de probabilidade).\n"
            "• 0  → desligado\n"
            "• 40 → recomendado\n"
            "• 1  → greedy (só o mais provável)"
        ),
    )

    st.divider()

    # ============================================================
    # 🔁 ANTI-REPETIÇÃO
    # ============================================================
    st.subheader("🔁 Anti-repetição")

    st.number_input(
        "Penalidade de Repetição",
        min_value=1.0, max_value=10.0, step=0.1, format="%.1f",
        key="repetition_penalty",
        help=(
            "Penaliza tokens que já apareceram na resposta.\n\n"
            "• 1.0 → sem penalidade (modelo pode entrar em loop)\n"
            "• 2.2 → forte (default do notebook)\n"
            "• 3.0+ → exagerado, palavras-chave somem"
        ),
    )

    st.number_input(
        "n-gramas sem repetição",
        min_value=0, max_value=6, step=1,
        key="no_repeat_ngram_size",
        help=(
            "Bloqueia a repetição de qualquer n-grama de tamanho N.\n\n"
            "• 0 → desligado\n"
            "• 2 → muito restritivo (bloqueia bigramas)\n"
            "• 4 → equilibrado (notebook)\n"
            "• 6 → permite repetições curtas"
        ),
    )

    st.divider()

    # ============================================================
    # ⚙️ AÇÕES
    # ============================================================
    col1, col2 = st.columns([2, 1])
    with col1:
        params_button = st.button(
            "💾 Salvar Parâmetros",
            use_container_width=True,
            help="Grava os valores atuais em main_params.json.",
        )
    with col2:
        params_reset_button = st.button(
            "↺ Resetar",
            use_container_width=True,
            help="Restaura todos os parâmetros para os defaults do notebook.",
        )

    if st.button(
        "🔄 Recarregar modelos do disco",
        use_container_width=True,
        help="Limpa o cache. A próxima pergunta recarrega do zero.",
    ):
        reload_models()
        st.session_state.last_loaded = None
        st.success("Cache limpo. A próxima pergunta recarrega do disco.")

    # ==========================================================
    # 🔍 Painel de Debug
    # ==========================================================
    st.divider()
    with st.expander("🔍 Debug — modelos em cache", expanded=False):
        st.caption("Modelo selecionado:")
        st.code(_fmt_model(st.session_state["model_select"]), language=None)

        st.caption("Caminho do adaptador (resolvido):")
        _rel = CFG["models"][st.session_state["model_select"]]
        st.code(resolve_path(_rel), language=None)

        st.caption("Última resposta gerada por:")
        st.code(st.session_state.last_loaded or "(nenhuma ainda)", language=None)

        st.caption("Modelos atualmente em memória:")
        infos = cached_models_info()
        if not infos:
            st.code("(cache vazio)", language=None)
        else:
            for i in infos:
                st.markdown(
                    f"**{i['key']}** · device=`{i['device']}`  \n"
                    f"`path:` {i['adapter_path']}  \n"
                    f"`fingerprint:` `{i['fingerprint']}`"
                )


# =============== Salvar / resetar parâmetros ===============
if params_button:
    new_params = {
        "model_select":         st.session_state["model_select"],
        "min_new_tokens":       int(st.session_state["min_new_tokens"]),
        "max_new_tokens":       int(st.session_state["max_new_tokens"]),
        "length_penalty":       float(st.session_state["length_penalty"]),
        "do_sample":            bool(st.session_state["do_sample"]),
        "num_beams":            int(st.session_state["num_beams"]),
        "early_stopping":       bool(st.session_state["early_stopping"]),
        "num_return_sequences": int(st.session_state["num_return_sequences"]),
        "temperature":          float(st.session_state["temperature"]),
        "top_p":                float(st.session_state["top_p"]),
        "top_k":                int(st.session_state["top_k"]),
        "repetition_penalty":   float(st.session_state["repetition_penalty"]),
        "no_repeat_ngram_size": int(st.session_state["no_repeat_ngram_size"]),
    }
    save_params(new_params)
    st.session_state.saved_params = new_params

    alerta = st.success("Parâmetros salvos em main_params.json!")
    time.sleep(2)
    alerta.empty()

elif params_reset_button:
    save_params(DEFAULT_PARAMS.copy())
    st.session_state.saved_params = DEFAULT_PARAMS.copy()
    for c in column_list:
        st.session_state.pop(c, None)
    st.rerun()


# =============== Abas ===============
aba_chat, aba_analise = st.tabs(["💬 Chat", "📈 Análise"])


# =============== Chat ===============
with aba_chat:
    st.header("💬 Discussão com o Agente")

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button(
            "🗑️ Limpar Histórico",
            use_container_width=True,
            help="Apaga todas as mensagens da sessão e do disco.",
        ):
            st.session_state.messages = []
            clear_messages()
            st.rerun()

    chat_container = st.container(height=400, autoscroll=True)
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

        # ---- resolve caminho e carrega modelo ----
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

        # ---- fingerprint para provar qual modelo está ativo ----
        fp = adapter_fingerprint(model)
        label = (
            f"🤖 **{_fmt_model(st.session_state['model_select'])}** · "
            f"device=`{device}` · fingerprint=`{fp}`"
        )
        st.session_state.last_loaded = label

        # ---- gera resposta ----
        with st.chat_message("assistant"):
            with st.spinner("Gerando resposta..."):
                resposta = respond(
                    model, tokenizer, device,
                    instruction=prompt,
                    prompt_template=PROMPT_TEMPLATE,
                    # tamanho
                    min_new_tokens=int(st.session_state["min_new_tokens"]),
                    max_new_tokens=int(st.session_state["max_new_tokens"]),
                    length_penalty=float(st.session_state["length_penalty"]),
                    # decodificação
                    do_sample=bool(st.session_state["do_sample"]),
                    num_beams=int(st.session_state["num_beams"]),
                    early_stopping=bool(st.session_state["early_stopping"]),
                    num_return_sequences=int(st.session_state["num_return_sequences"]),
                    # amostragem
                    temperature=float(st.session_state["temperature"]),
                    top_p=float(st.session_state["top_p"]),
                    top_k=int(st.session_state["top_k"]),
                    # anti-repetição
                    repetition_penalty=float(st.session_state["repetition_penalty"]),
                    no_repeat_ngram_size=int(st.session_state["no_repeat_ngram_size"]),
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


# =============== Análise ===============
with aba_analise:
    st.header("📈 Análise do Histórico de Perda")
    st.caption(
        "Valores reais do notebook (10 épocas, LoRA r=4). "
        "**Ruim = época 1**, **Bom = época 5**, **Ótimo = época 10**."
    )

    historico = {
        "Época":      [1,     2,     3,     4,     5,     6,     7,     8,     9,     10],
        "Train Loss": [3.245, 2.679, 2.575, 2.580, 2.508, 2.562, 2.498, 2.514, 2.522, 2.480],
        "Val Loss":   [2.807, 2.523, 2.469, 2.435, 2.414, 2.398, 2.389, 2.386, 2.384, 2.384],
    }

    st.subheader("Curvas de perda por época")
    st.line_chart(
        data={k: v for k, v in historico.items() if k != "Época"},
        x_label="Época",
        y_label="Loss",
    )

    st.subheader("Resumo comparativo")
    st.dataframe(
        {
            "Perfil":       ["Ruim", "Bom", "Ótimo"],
            "Época":        [1, 5, 10],
            "Val Loss":     [2.807, 2.414, 2.384],
            "Perplexidade": [16.55, 11.18, 10.85],
            "Ganho (%)":    [0.0, 94.0, 100.0],
        },
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "💡 **Dica de comparação justa:** faça a **mesma pergunta** em cada "
        "modelo mantendo os demais parâmetros iguais. Confira a etiqueta com "
        "o **fingerprint** embaixo de cada resposta — se for o mesmo entre "
        "Ruim/Bom/Ótimo, o adaptador não está sendo trocado."
    )