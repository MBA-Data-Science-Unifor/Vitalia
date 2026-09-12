# syntax=docker/dockerfile:1

# ---------- Estágio 1: Builder ----------
FROM python:3.12-slim AS builder

# Instala o uv oficial (gerenciador de pacotes rápido)
COPY --from=ghcr.io/astral-sh/uv:0.6.11 /uv /uvx /bin/

WORKDIR /app

# Dependências de sistema necessárias para compilar alguns pacotes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia APENAS os arquivos de definição de projeto (cache de camadas)
COPY pyproject.toml uv.lock ./

# Cria venv e instala dependências usando o lock (reprodutível)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --frozen --no-install-project --no-dev

# Copia o código da aplicação
COPY . .

# Instala o projeto em si (sem reinstalar dependências)
RUN . /opt/venv/bin/activate && \
    uv pip install --no-deps -e .

# ---------- Estágio 2: Runtime ----------
FROM python:3.12-slim

# curl é usado pelo healthcheck do docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Usuário não-root
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Copia o venv do estágio builder
COPY --from=builder /opt/venv /opt/venv

# Copia o código
COPY --from=builder /app/src ./src

# Diretório de cache do HuggingFace (montado como volume no compose)
RUN mkdir -p /home/appuser/app/.cache/huggingface
ENV HF_HOME=/home/appuser/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]