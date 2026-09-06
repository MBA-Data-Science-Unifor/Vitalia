# Vitalia — Assistente Inteligente para Saúde e Bem-Estar

**Grupo:** [definir nome do grupo]**Integrantes:**

- Nome1, Matrícula1
- Nome2, Matrícula2
- Nome3, Matrícula3

**Repositório:** [https://github.com/MBA-Data-Science-Unifor/Vitalia](https://github.com/MBA-Data-Science-Unifor/Vitalia)

---

## Descrição

**Vitalia** é uma aplicação que permite carregar a base de dados de **Saúde** (1200 pares instrução‑resposta em português), ajustar hiperparâmetros, treinar um modelo de linguagem com **LoRA** e, ao final, conversar com o modelo treinado via interface de chat.

A base utilizada é um recorte temático do `dominguesm/alpaca-data-pt-br`, com foco em exercício, alimentação, sono e bem-estar.

**Modelo base obrigatório:**
Usamos o [`pierreguillou/gpt2-small-portuguese`](https://huggingface.co/pierreguillou/gpt2-small-portuguese), um GPT-2 com ~124M parâmetros, já pré-treinado em português. Ele é leve, roda em CPU e no Colab gratuito, e é o modelo exigido pelo trabalho.

---

## Tecnologias

- Python 3.12+
- [Streamlit](https://streamlit.io/) – interface web interativa
- [Plotly](https://plotly.com/) – visualização de métricas e análises (opcional)
- [Transformers](https://huggingface.co/docs/transformers/index) – modelos e tokenizadores
- [PEFT](https://huggingface.co/docs/peft/index) – LoRA para fine-tuning eficiente
- [Datasets](https://huggingface.co/docs/datasets/index) – manipulação da base JSONL
- [PyTorch](https://pytorch.org/) – backend de treino
- Docker / docker-compose (recomendado para reprodutibilidade)

---

## Estrutura do projeto

```
.
├── app/                        # Código da aplicação Streamlit
│   └── app.py                  # Interface principal (chat, controles, treino)
├── src/                        # Lógica de backend e treinamento
│   ├── __init__.py             # Torna src um pacote Python
│   ├── train.py                # Funções de treino com LoRA (loop manual)
│   └── model_utils.py          # Carregamento do modelo, tokenizador e inferência
├── analysis/                   # Análises exploratórias e comparativas
│   └── evaluation_metrics.py   # Scripts para avaliar o modelo treinado
├── presentation/               # Materiais da apresentação
│   ├── slides.pdf              # Slides da apresentação final
│   └── video_link.txt          # Link para o vídeo de demonstração (2-3 min)
├── trained_model/              # Modelo ajustado (adaptadores LoRA) – gerado após treino
│   ├── adapter_model.bin
│   └── adapter_config.json
├── pyproject.toml              # Configuração do projeto e dependências
├── Dockerfile                  # Imagem Docker
├── docker-compose.yml          # Orquestração para subir o container
├── .gitignore                  # Arquivos ignorados (inclui a base de dados)
├── .dockerignore               # Arquivos ignorados na construção da imagem
├── LICENSE                     # Licença MIT
├── README.md                   # Este arquivo
└── saude.jsonl                 # NÃO INCLUIR – a base é fornecida à parte
```

> **Nota:** A pasta `trained_model/` contém apenas os adaptadores LoRA (alguns MB) e será incluída no ZIP de entrega, conforme solicitado. A base `saude.jsonl` **nunca** deve ser versionada ou enviada.

---

## Instalação

### Sem Docker (ambiente local)

1. **Clone o repositório**:

```bash
   git clone https://github.com/MBA-Data-Science-Unifor/Vitalia.git
   cd Vitalia
```

2. **Crie e ative um ambiente virtual** (recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. **Instale o projeto e suas dependências**:

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```
4. **Coloque a base** `saude.jsonl` na raiz do projeto (ao lado da pasta `app/`).

### Com Docker (recomendado para reprodutibilidade)

1. **Certifique-se de ter o Docker e o docker-compose instalados**.
2. **Coloque o arquivo** `saude.jsonl` na raiz do projeto (ao lado do `Dockerfile`).
3. **Construa a imagem**:
   ```bash
   docker-compose build
   ```
4. **Execute o container**:
   ```bash
   docker-compose up
   ```

A aplicação estará disponível em `http://localhost:7860`.

> O Dockerfile copia a base para dentro do container. Se preferir usar um volume, remova o `COPY` do `Dockerfile` e adicione volumes no `docker-compose.yml`.

---

## Como usar

### Formatação dos dados de treino

O modelo é treinado com o seguinte formato fixo, que deve ser usado **exatamente igual** durante o treino e na inferência:

```
### Pergunta:
{instruction}

### Resposta:
{output}{eos_token}
```

- `instruction` é o campo da base.
- `output` é a resposta esperada.
- `eos_token` é adicionado ao final para ensinar o modelo a parar de gerar texto.

Esse formato é aplicado automaticamente pelo código de treino. **Não altere** esse padrão, senão o modelo não reconhecerá as perguntas na hora da conversa.

### Parâmetros padrão de treinamento

A interface já vem com valores padrão sensíveis:

| Parâmetro                | Valor padrão        | Onde entra                                       |
| ------------------------- | -------------------- | ------------------------------------------------ |
| Épocas                   | AINDA A SER DEFINIDO | Treino – quantas vezes o modelo vê a base      |
| Learning Rate             | AINDA A SER DEFINIDO | Treino – tamanho do passo do ajuste             |
| Batch Size                | AINDA A SER DEFINIDO | Treino – exemplos por lote                      |
| Temperatura               | AINDA A SER DEFINIDO | Inferência – criatividade da resposta          |
| Penalidade de repetição | AINDA A SER DEFINIDO | Inferência – desestimula repetição de frases |

Todos podem ser ajustados na barra lateral antes de treinar ou conversar.

### Primeira execução (sem modelo salvo)

- Ao abrir a aplicação, a barra lateral exibe os controles e um aviso de que nenhum modelo foi encontrado.
- O chat fica bloqueado até que um modelo seja treinado.

### Treinando um modelo

1. **Carregue a base** (opcional): use o botão "Upload" para enviar um arquivo `.jsonl`. Se não fizer upload, a aplicação usará `saude.jsonl` na raiz.
2. **Ajuste os hiperparâmetros** conforme desejado.
3. **Clique em "Treinar / Retreinar"** – o progresso é exibido e a perda (loss) é registrada a cada passo.
4. Após o treino, o modelo é salvo em `./trained_model` usando `save_pretrained()` (adaptadores LoRA) e carregado automaticamente para o chat.

### Acompanhamento da perda (loss)

Durante o treino, a aplicação armazena os valores de loss de cada passo. Ao final, um gráfico da curva de loss é exibido na aba **"Análise"**. Esse gráfico é fundamental para avaliar se o modelo está aprendendo:

- **Loss decrescente** → aprendizado está ocorrendo.
- **Loss oscilante** → learning rate muito alta.
- **Loss estagnada** → modelo pode estar com poucas épocas ou dados mal formatados.

### Retreinando com novos parâmetros

O botão **"Treinar / Retreinar"** está sempre disponível. Se você modificar épocas, learning rate ou batch size e clicar no botão:

- O modelo antigo é apagado.
- Um novo treino é iniciado com os parâmetros atuais.
- O novo modelo é salvo e recarregado imediatamente.
- O chat passa a usar o novo modelo sem reiniciar a aplicação.

A **temperatura** e a **penalidade de repetição** são parâmetros de inferência exclusivos – alterá-los **não** dispara retreino e as mudanças são aplicadas instantaneamente nas respostas.

### Conversando com o modelo

- Após o treino, a aba "Chat" é desbloqueada.
- Digite sua pergunta sobre saúde.
- O modelo gera a resposta usando a temperatura e a penalidade de repetição configuradas.
- O histórico da conversa é mantido durante a sessão.

---

## Cache do modelo treinado

- O modelo ajustado é salvo em `./trained_model` com `save_pretrained()` – **não** usamos `joblib` ou serialização arbitrária.
- Na próxima inicialização, a aplicação carrega automaticamente o modelo do disco.
- Se o modelo não existir, a aplicação avisa e aguarda o usuário iniciar o treino.

---

## Preparando o ZIP de entrega

Segundo as instruções do trabalho, você deve entregar: **o código, o modelo treinado e o README**. Portanto, o arquivo `.zip` deve conter:

```
Vitalia.zip
├── app/
├── src/
├── analysis/
├── presentation/
├── trained_model/          # Adaptadores LoRA (obrigatório)
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

**Passos para criar o ZIP:**

1. Certifique-se de que `saude.jsonl` **NÃO** está na raiz.
2. Remova pastas como `__pycache__`, `.pytest_cache`, `venv`.
3. Execute (Linux/Mac):
   ```bash
   zip -r Vitalia.zip app/ src/ analysis/ presentation/ trained_model/ pyproject.toml Dockerfile docker-compose.yml .gitignore .dockerignore LICENSE README.md
   ```
4. No Windows, use o Explorer ou PowerShell com `Compress-Archive`.

> **Atenção:** A base `saude.jsonl` **nunca** deve ser incluída. O modelo treinado (`trained_model/`) **deve** ser incluído, pois o professor solicita explicitamente.

---

## Como rodar os testes (opcional)

Para validar a aplicação com um subconjunto de dados (100 exemplos), defina a variável de ambiente `TEST_MODE=1`:

```bash
TEST_MODE=1 streamlit run app/app.py
```

## Licença

MIT License – veja o arquivo [LICENSE](https://github.com/MBA-Data-Science-Unifor/Vitalia/blob/main/LICENSE).

---

## Apresentação

Os slides e o vídeo de demonstração (2–3 minutos) estão disponíveis na pasta `presentation/`. O link do vídeo (hospedado no Google Drive ou YouTube não listado) deve ser inserido abaixo.

**Link do vídeo:** [inserir link aqui]
