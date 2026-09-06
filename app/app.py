# ============ Importação de Bibliotecas ============
import streamlit as st
import time

# ============ Configuração Inicial ============
st.set_page_config(
    page_title="Comparativo de Assistentes de IA (Saude)",
    layout="wide"
)

st.title("Vital AI (Vital)")
st.write("Avalie os Modelos")


# ============ Inicialização dos Estados ============

# Preparação do Modelo
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

# Criação do Historico de Mensagens
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Criação do Historico de Perda
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []

# Validação de qual botão executar
if 'acao_ativa' not in st.session_state:
    st.session_state.acao_ativa = ''    



# ============ SideBar ============
with st.sidebar:
    st.title('Menu Principal')
    st.header("Parâmetros")

    # Recuperação do arquivo principal
    arquivo_principal = st.file_uploader("Escolha sua base de dados", type='jsonl')

    epochs = st.number_input(
        label="Defina a Quantidade de Épocas", 
        min_value=1.0, max_value=100.0, value=1.0, step=1.0
    )

    learning_rate = st.number_input(
        "Defina a Taxa de Aprendizado",
        min_value=0.1, max_value=100.0, value=0.1, step=0.1, format="%.2f"
    )

    batch_size = st.selectbox(
        "Defina a Quantidade de Lotes",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    temperature_slider = st.slider(
        label='Defina a Temperatura do Modelo',
        min_value=0.1, max_value=100.0, value=0.5, step=0.1, format="%.2f"
    )

    penality_slider = st.slider(
        label='Defina a Penalidade do Modelo',
        min_value=0.1, max_value=100.0, value=0.1, step=0.1, format="%.2f"
    )

    col1, col2 = st.columns(2)
    with col1:
        botao_treinar = st.button("Treinar", use_container_width=True)
    with col2:
        botao_retreinar = st.button("Re-Treinar", use_container_width=True)


# ============ Validação do Treinamento ============
if botao_treinar:
    st.session_state.acao_ativa = 'botao_treinar' # persistência na escolha do botão treinar
elif botao_retreinar:
    st.session_state.acao_ativa = 'botao_retreinar' # persistência na escolha do botão re-treinar


# ============ Logica do Botão de Treinamento ============
if st.session_state.acao_ativa == 'botao_treinar':
    try:
        if arquivo_principal is None:
            # Criação de uma mensagem de alerta
            placeholder = st.sidebar.empty()
            placeholder.info("Precisa de um arquivo para realizar o treinamento")

            # Intervalo de tempo para retirar o aviso prévio
            time.sleep(3)

            # Remoção da mensagem de alerta
            placeholder.empty()
        else:
            # train_model() # deve capturar o loss story
            # load_or_init_model() 

            # Teste Não Oficial
            with st.spinner("Treinando..."):
                simulacao_loss = [0.33, 0.55, 0.81, 0.2]
                st.session_state.loss_history  = simulacao_loss
                st.session_state.model_ready=True
                time.sleep(5) 

            st.sidebar.success("Treinamento realizado com sucesso!")

    except Exception as e:
        st.sidebar.error(f'Erro durante o treinamento: {str(e)}')
        st.session_state.model_ready=False

    st.session_state.acao_ativa = ''
    print(f'Ação Ativa: {st.session_state.acao_ativa}')


# ============ Logica do botão de Re-Treinamento ============
elif st.session_state.acao_ativa == 'botao_retreinar':

    if not st.session_state.get('model_ready', False):
        st.sidebar.warning("Nenhum modelo treinado para re-treinar. Use 'Treinar' primeiro.")

    elif arquivo_principal is None:
        placeholder = st.sidebar.empty()
        placeholder.info("Precisa de um arquivo para realizar o treinamento")
        
        # Intervalo de tempo para retirar o aviso prévio
        time.sleep(3)
        
        # Remoção da mensagem de alerta
        placeholder.empty()
    else:
        with st.spinner("Re-Treinando..."):
            simulacao_loss = [0.9, 0.6, 0.33, 0.21]
            st.session_state.loss_history.extend(simulacao_loss)
            st.session_state.model_ready=True
            time.sleep(5) 

        st.sidebar.success("Re-Treinamento realizado com sucesso!")
    st.session_state.acao_ativa = ''


# ============ Abas ============
aba_chat, aba_treinamento, aba_analise = st.tabs(['💬 Chat', '📊 Treinamento', '📈 Análise'])

# ============ Chat ============
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
        chat_container = st.container(height=200, autoscroll=True)
        
        # Exibição do Histórico de Mensagens
        with chat_container:
            for msg in (st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Execução do Prompt de Comando
        if prompt := st.chat_input("Escreva Sua Pergunta"):

            # Salva a pergunta do usuário
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            # Salva a resposta do usuário
            resposta = f"Reposta do Assistente ({temperature_slider} {penality_slider}): "

            # resposta = generate_response(temperatua)
            st.session_state.messages.append({'role': 'assistant', 'content': resposta})

            # Atualiza a lista
            st.rerun()


# ============ Análise ============
with aba_analise:
    st.header("📈 Análise do Histórico de Perda")

    if st.session_state.loss_history:
        # Exebição de um gráfico de histórico de perda
        st.line_chart(st.session_state.loss_history)
    else:
        st.warning("Nenhum Histórico de Perda disponível. Treine novamente o modelo")



# ============ Treinamento ============
with aba_treinamento:
    st.header("📊 Treinamento")

    # Avalia se o modelo está pronto para responder
    if st.session_state.model_ready == False:
        st.info("Treine Primeiro")
    else:
        st.info("TREINO")
