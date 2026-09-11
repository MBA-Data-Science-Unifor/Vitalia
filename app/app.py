# ============ Importação de Bibliotecas ============
import streamlit as st
import time
from app_functions import train_new_model, load_dataset, continue_training_model
from storage import load_messages, save_messages, clear_messages


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
    st.session_state.messages = load_messages()

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
    #arquivo_principal = load_dataset(st)

    # Escolha o Modelo que deseja 
    model_select = st.selectbox("Escolha seu Modelo", 
                                ['Bom', 'Mediano', 'Ruim'])

    # max_tokens
    max_new_tokens = st.number_input(
        "Máx. de tokens novos",
        min_value=1.0, max_value=60.0, value=60.0, step=1.0,
        help='Numero máximo de tokens novos que o modelo vai gerar'
    )

    # Amostragem
    do_sample = st.checkbox("Usar amostragem (do_sample)", value=False, 
                            help='chave mestra entre modelo determinístico e estocástico')

    # Número de Beams
    num_beams = st.number_input("Número de Hipóteses (Beams)", 
                        min_value=1.0, max_value=10.0, value=4.0, step=1.0,
                        help='Quantas hipóteses paralelas o modelo mantém ao mesmo tempo durante a decodificação')

    # Penalidade
    repetition_penalty = st.number_input(
        "Penalidade de Repetição",
         min_value=1.0, max_value=10.0, value=1.0, step=1.0
    )

    # Temperatura
    temperature_input = st.number_input(
        label='Selecione a Temperatura do Modelo',
        min_value=0.1, max_value=1.0, value=0.7, step=0.1, format="%.2f",
        help="a temperatura 'achata' ou 'afina' a distribuição de probabilidade antes de amostrar"
    )

    # NGRAM SIZE
    no_repeat_ngram_size = st.number_input(
        label='Tamanho de n-gramas sem repetição',
        min_value=1.0, max_value=4.0, value=4.0, step=1.0,
        help='bloqueia a repetição de n-gramas já vistos'
    )

    # Top P
    top_p = st.slider(
        "Amostrágem de Núcleo (top p)", 0.1, 1.0, 0.9, 0.05,
        help="considera apenas o menor conjunto de tokens cuja probabilidade acumulada >= p"
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
        #if arquivo_principal is None:
            # Criação de uma mensagem de alerta
           # placeholder = st.sidebar.empty()
           # placeholder.info("Precisa de um arquivo para realizar o treinamento")

            # Intervalo de tempo para retirar o aviso prévio
           # time.sleep(3)

            # Remoção da mensagem de alerta
            #placeholder.empty()
            # train_model() # deve capturar o loss story
            # load_or_init_model() 

            # Teste Não Oficial
            with st.spinner("Treinando..."):
                model, loss_story = train_new_model(
                    temperature_input=temperature_input,
                    penalty_input=repetition_penalty
                )

                # Atualiza o Modelo e a adição de suas perdas
                st.session_state.model = model
                st.session_state.loss_history = loss_story
                st.session_state.model_ready=True

            st.sidebar.success("Treinamento realizado com sucesso!")
            st.session_state.acao_ativa = ''


# ============ Logica do botão de Re-Treinamento ============
elif st.session_state.acao_ativa == 'botao_retreinar':
    #try:
        if st.session_state.model_ready == False:
            st.sidebar.warning("Nenhum modelo treinado para re-treinar. Use 'Treinar' primeiro.")

        #elif arquivo_principal is None:
           # placeholder = st.sidebar.empty()
            #placeholder.info("Precisa de um arquivo para realizar o treinamento")
            
            # Intervalo de tempo para retirar o aviso prévio
            #time.sleep(3)
            
            # Remoção da mensagem de alerta
            #placeholder.empty()
        else:

            # Efetuar o Re-Treinamento
            with st.spinner("Re-Treinando..."):

                model, loss_story = continue_training_model(
                    model=st.session_state.model,
                    temperature_input=temperature_input,
                    penalty_input=repetition_penalty
                )

                st.session_state.model = model
                st.session_state.loss_history.extend(loss_story)
                st.session_state.model_ready=True

            st.sidebar.success("Re-Treinamento realizado com sucesso!")
   # except Exception as e:
        #st.sidebar.error(f'Erro durante o re-treinamento: {str(e)}')
        #st.session_state.model_ready=False

        st.session_state.acao_ativa = ''



aba_chat, aba_analise = st.tabs(['💬 Chat', '📈 Análise'])


# ============ Chat ============
with aba_chat:
    st.header("💬 Discussão com o Agente")

    # Definição do Layout Padrão do ChatBot
    col1, col2 = st.columns([6,1])

    # Logica do Botão de Remoção do Histórico de Mensagens
    with col2:
        if st.button("Limpar Histórico", use_container_width=True):
            st.session_state.messages = [] # Limpeza do Histórico
            clear_messages()
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
            resposta = f"Reposta do Assistente ({temperature_input} {repetition_penalty}): "

            # resposta = generate_response(temperatua)
            st.session_state.messages.append({'role': 'assistant', 'content': resposta})

            # persiste no salvamento de mensagens
            save_messages(st.session_state.messages)
        
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
