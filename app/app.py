# ============ Importação de Bibliotecas ============
import streamlit as st
import time
from storage import load_messages, save_messages, clear_messages, load_params, save_params, DEFAULT_PARAMS


# ============ Configuração Inicial ============
st.set_page_config(
    page_title="Comparativo de Assistentes de IA (Saude)",
    layout="wide"
)

st.title("Vital AI (Vital)")
st.write("Avalie os Modelos")


# =============== Inicialização dos Estados ===============

# Criação do Historico de Mensagens
if 'messages' not in st.session_state:
    st.session_state.messages = load_messages()

if 'saved_params' not in st.session_state:
    st.session_state.saved_params = load_params()

# Preparação do Modelo
#if 'model_ready' not in st.session_state:
    #st.session_state.model_ready = False

# Criação do Historico de Perda
#if 'loss_history' not in st.session_state:
    #st.session_state.loss_history = []



def _return_param(key, value):
    """Define o valor inicial para os parâmetros"""

    if key not in st.session_state:
        st.session_state[key] = value


# =============== Lista de Parâmetros ===============
param_list = st.session_state.saved_params

# =============== SideBar ===============
with st.sidebar:
    st.title('Menu Principal')
    st.header("Parâmetros")

    # Definição das variaveis dos parâmetros
    column_list = ['model_select', 'max_new_tokens', 'do_sample', 'num_beams', 
                   'repetition_penalty', 'no_repeat_ngram_size','temperature', 'top_p']

    for c in column_list:
        _return_param(c, param_list[c])


    # Escolha o Modelo que deseja 
    model_select = st.selectbox("Escolha seu Modelo", 
                                ['Bom', 'Mediano', 'Ruim'], key='model_select')

    # max_tokens
    max_new_tokens = st.number_input(
        "Máx. de tokens novos",
        min_value=1.0, max_value=60.0, 
        step=1.0, key='max_new_tokens',
        help='Numero máximo de tokens novos que o modelo vai gerar'
    ) #value=60.0, 

    # Amostragem
    do_sample = st.checkbox("Usar amostragem (do_sample)", key="do_sample",  
                            help='chave mestra entre modelo determinístico e estocástico')  #value=False, 

    # Número de Beams
    num_beams = st.number_input("Número de Hipóteses (Beams)", 
                    min_value=1.0, max_value=10.0, key='num_beams', step=1.0, 
                    help='Quantas hipóteses paralelas o modelo mantém ao mesmo tempo durante a decodificação') #value=4.0, 

    # Penalidade
    repetition_penalty = st.number_input(
        "Penalidade de Repetição",
        min_value=1.0, max_value=10.0,
        step=1.0, key='repetition_penalty' 
    ) #value=1.0,

    # Temperatura
    temperature_input = st.number_input(
        label='Selecione a Temperatura do Modelo',
        min_value=0.1, max_value=1.0, 
        step=0.1, format="%.2f", key='temperature',
        help="a temperatura 'achata' ou 'afina' a distribuição de probabilidade antes de amostrar"
    ) #value=0.7, 

    # NGRAM SIZE
    no_repeat_ngram_size = st.number_input(
        label='Tamanho de n-gramas sem repetição',
        min_value=1.0, max_value=4.0, 
        step=1.0, key='no_repeat_ngram_size',
        help='bloqueia a repetição de n-gramas já vistos'
    ) #value=4.0, 

    # Top P
    top_p = st.slider(
        "Amostrágem de Núcleo (top p)", 
        min_value=0.1, max_value=1.0, step=0.05, key='top_p', 
        help="considera apenas o menor conjunto de tokens cuja probabilidade acumulada >= p"
    ) #value=0.9, 

    col1, col2 = st.columns([2,1])

    with col1:
        params_button = st.button("💾 Salvar Parâmetros", use_container_width=True)

    with col2:
        params_reset_button = st.button("Re-Setar Parâmetros", use_container_width=True)


# =============== Logica de Salvar Parâmetros ===============
if params_button:
    new_params = {
        "model_select": st.session_state['model_select'],
        "max_new_tokens": int(st.session_state['max_new_tokens']),
        "do_sample": bool(st.session_state['do_sample']),
        "num_beams": int(st.session_state['num_beams']),
        "repetition_penalty": float(st.session_state['repetition_penalty']),
        "no_repeat_ngram_size": int(st.session_state['no_repeat_ngram_size']),
        "temperature": float(st.session_state['temperature']),
        "top_p": float(st.session_state['top_p']),
    }

    save_params(new_params)
    st.session_state.saved_params = new_params

    alerta = st.empty()
    alerta = st.success("Parâmetros salvos!")
    time.sleep(5)
    alerta.empty()


elif params_reset_button:
    
    save_params(DEFAULT_PARAMS.copy())
    st.session_state.saved_params = DEFAULT_PARAMS.copy()

    for c in column_list:
        st.session_state.pop(c, param_list[c])

    st.rerun()





# =============== Abas ===============
aba_chat, aba_analise = st.tabs(['💬 Chat', '📈 Análise'])

# =============== Chat ===============
with aba_chat:
    st.header("💬 Discussão com o Agente")

    # Definição do Layout Padrão do ChatBot
    col1, col2 = st.columns([6,1])

    with col2:
        if st.button("Limpar Histórico", use_container_width=True):
            st.session_state.messages = []           
            clear_messages()
            st.rerun()

    # Criação de um Container de mensagens
    chat_container = st.container(height=200, autoscroll=True)
        
    # Exibição do Histórico de Mensagens
    with chat_container:
        for msg in (st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


    if prompt := st.chat_input("Escreva Sua Pergunta"):
        # Salva a pergunta do usuário
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Salva a resposta do assistente, porém ainda está no modo Dummy
        resposta = f"Resposta do Assistente ({param_list}):"

        st.session_state.messages.append({'role': 'assistant', 'content': resposta})

        # 💾 Persiste em disco
        save_messages(st.session_state.messages) 
        st.rerun()


# =============== Análise do Histórico de Perda ===============
with aba_analise:
    st.header("📈 Análise do Histórico de Perda")

    # Exibição de um gráfico de histórico de perda
    # Isso é apenas uma simulação, poderá ser alterado através da integração
    loss_history = [0.5, 3, 9.5, 2, 1.5] # (épocas por historico de perda)
    st.line_chart(loss_history)
