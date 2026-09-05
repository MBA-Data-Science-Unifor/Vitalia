# Definição da versão do Python
FROM python:3.12

# Definição do diretório de trabalho
WORKDIR /app

# Instalação de Dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalação direta do torch
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

# Copia do restante do código fonte
COPY . . 

# Definição da portaa padrão do streamlit
EXPOSE 7860

# Execução do comando de execução
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
