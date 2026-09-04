import streamlit as st

st.set_page_config(layout="wide")

st.title("Vital AI (Vital)")
st.write("Avalie os Modelos")


if 'acao_ativa' not in st.session_state:
    st.session_state.acao_ativa = ''    


# Criação do Menu de Opções
with st.sidebar:
    st.title('Menu Principal')

    st.header("Parâmetros")

    epochs = st.number_input(
        label="Defina a Quantidade de Épocas", 
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=1.0
    )

    learning_rate = st.number_input(
        "Defina a Taxa de Aprendizado",
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        value=0.1,
        format="%.2f"
    )

    temperature_slider = st.slider(
        label='Defina a Temperatura do Modelo',
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        format="%.2f"
    )

    penality_slider = st.slider(
        label='Defina a Penalidade do Modelo',
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        format="%.2f"
    )

    botao_treinar = st.button("Treinar")
    botao_retreinar = st.button("Re-Treinar")


if botao_treinar:
    st.session_state.acao_ativa = 'botao_treinar' # persistência na escolha do botão treinar
elif botao_retreinar:
    st.session_state.acao_ativa = 'botao_retreinar' # persistência na escolha do botão re-treinar


if st.session_state.acao_ativa == 'botao_treinar':
    st.subheader("Treinando")
    st.session_state.acao_ativa = ''


elif st.session_state.acao_ativa == 'botao_retreinar':
    st.subheader("Re-Treinando")
    st.session_state.acao_ativa = ''
