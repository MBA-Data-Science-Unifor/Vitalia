import streamlit as st
import time

st.set_page_config(
    page_title="Comparativo de Assistentes de IA (Saude)",
    layout="wide"
)

# Definição do Título e do Subtítulo
st.title("Vital AI (Vital)")
st.write("Avalie os Modelos")


# Criação de Variáveis de Sessão do StreamLit

# Preparação do Modelo
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

# Criação do Historico de Mensagens
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Criação do Historico de Perda
if 'loss_story' not in st.session_state:
    st.session_state.loss_story = []

# Validação de qual botão executar
if 'acao_ativa' not in st.session_state:
    st.session_state.acao_ativa = ''    



# Menu de Opções 
barra_opcoes = st.sidebar

with barra_opcoes:

    barra_opcoes.title('Menu Principal')
    barra_opcoes.header("Parâmetros")

    # Recuperação do arquivo principal
    arquivo_principal = barra_opcoes.file_uploader("Escolha sua base de dados", type='jsonl')

    # Definição da Quantidade de Épocas
    epochs = barra_opcoes.number_input(
        label="Defina a Quantidade de Épocas", 
        min_value=1.0,
        max_value=100.0,
        value=1.0,
        step=1.0
    )

    learning_rate = barra_opcoes.number_input(
        "Defina a Taxa de Aprendizado",
        min_value=0.1,
        max_value=100.0,
        step=0.1,
        value=0.1,
        format="%.2f"
    )

    batch_size = barra_opcoes.selectbox(
        "Defina a Quantidade de Lotes",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    temperature_slider = barra_opcoes.slider(
        label='Defina a Temperatura do Modelo',
        min_value=0.1,
        max_value=100.0,
        step=0.1,
        format="%.2f"
    )

    penality_slider = barra_opcoes.slider(
        label='Defina a Penalidade do Modelo',
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        format="%.2f"
    )

    botao_treinar = barra_opcoes.button("Treinar")
    botao_retreinar = barra_opcoes.button("Re-Treinar")


# Validação do Treinamento
if botao_treinar:
    st.session_state.acao_ativa = 'botao_treinar' # persistência na escolha do botão treinar
elif botao_retreinar:
    st.session_state.acao_ativa = 'botao_retreinar' # persistência na escolha do botão re-treinar



# Logica de Treinar
if st.session_state.acao_ativa == 'botao_treinar':

    try:
        if arquivo_principal is None:
            # Criação de uma mensagem de alerta
            placeholder = barra_opcoes.empty()
            placeholder.info("Precisa de um arquivo para realizar o treinamento")

            # Intervalo de tempo para retirar o aviso prévio
            time.sleep(5)

            # Remoção da mensagem de alerta
            placeholder.empty()
        else:
            # train_model() # deve capturar o loss story
            # load_or_init_model() 
            st.session_state.model_ready=True

    except Exception as e:
        barra_opcoes.error(f'Erro durante o treinamento: {str(e)}')
        st.session_state.model_ready=False

    st.session_state.acao_ativa = ''

# Logica de Re-Treinar
elif st.session_state.acao_ativa == 'botao_retreinar':
    st.session_state.model_ready=False

    # Avaliar se o arquivo existe
    if arquivo_principal is None:
        # Criação de uma mensagem de alerta
        placeholder = barra_opcoes.empty()
        placeholder.info("Precisa de um arquivo para realizar o re-treinamento")

        # Intervalo de tempo para retirar o aviso prévio
        time.sleep(5)

        # Remoção da mensagem de alerta
        placeholder.empty()
    else:
        st.session_state.model_ready=True

    st.session_state.acao_ativa = ''




# Abas
aba_chat, aba_treinamento, aba_analise = st.tabs(['💬 Chat', '📊 Treinamento', '📈 Análise'])

with aba_chat:
    st.header("💬 Discussão com o Agente")

    # Definição do Layout Padrão do ChatBot
    col1, col2 = st.columns([6,1])

    # Logica do Botão de Remoção do Histórico de Mensagens
    with col2:
        if st.button("Limpar Histórico", use_container_width=True):
            st.session_state.messages = [] # Limpeza do Histórico
            st.rerun() # Recarregamento da página após a limpeza

    # Avalia se o modelo está pronto para responder
    if st.session_state.model_ready == False:
        st.warning("Treine Primeiro")
    else:

        # Criação de um Container de mensagens
        message_container = st.container(height=200, autoscroll=True)
        
        # Exibição do Histórico de Mensagens
        with message_container:
            for msg in (st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Execução do Prompt de Comando
        if prompt := st.chat_input("Escreva Sua Pergunta"):

            # Salva a pergunta do usuário
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            # Salva a resposta do usuário
            resposta = "A resposta é: "
            print(f"Temperatua Atual: {temperature_slider}")
            print(f'Penalidade: {penality_slider}')

            # resposta = generate_response(temperatua)
            st.session_state.messages.append({'role': 'assistant', 'content': resposta})

            # Atualiza a lista
            st.rerun()



with aba_analise:
    st.header("📈 Análise do Histórico de Perda")

    if st.session_state.loss_history == []:
        st.warning("Ausência de Histórico de Perda")