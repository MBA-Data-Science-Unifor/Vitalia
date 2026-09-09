# 🚀 Vitalia – Assistente Inteligente para Saúde e Bem-Estar

**Vitalia** é uma aplicação interativa desenvolvida em **Streamlit** que permite treinar e conversar com um modelo de linguagem (GPT‑2) especializado em saúde, alimentação, exercício, sono e bem‑estar. O projeto utiliza **LoRA** (Low‑Rank Adaptation) para um fine‑tuning eficiente, carregando uma base de 1.200 pares instrução‑resposta.

## 📖 Visão Geral

O projeto foi desenvolvido como trabalho final para a disciplina de [Nome da Disciplina] e tem como objetivo demonstrar o ciclo completo de um projeto de **fine‑tuning de LLMs**:

1. Carregamento e tokenização de dados.
2. Ajuste de hiperparâmetros (épocas, learning rate, LoRA, etc.).
3. Treino com validação e salvamento do melhor checkpoint.
4. Interface de chat amigável para testar o modelo treinado.

---

## ⚙️ Tecnologias Utilizadas

- **Python 3.12**
- **PyTorch** – backend do modelo.
- **Hugging Face Transformers** – carregamento do modelo base (`pierreguillou/gpt2-small-portuguese`).
- **PEFT (LoRA)** – fine‑tuning eficiente (~0.12% dos parâmetros treináveis).
- **Datasets** – gerenciamento dos dados.
- **Streamlit** – interface web interativa.
- **Matplotlib / Seaborn** – visualização das curvas de loss e perplexidade.
- **PyYAML** – gerenciamento centralizado de configurações.

---

## 📁 Estrutura do Projeto

```
📂 assistant-health/                     # Project root
│
├───📂 .venv/                            # Virtual environment (not committed to Git)
│
├───📂 src/                              # Reusable Python modules
│   │   __init__.py
│   │   model_loader.py                  # Functions for loading model/tokenizer
│   │   inference.py                     # respond() with optional context
│   │   data_utils.py                    # Formatting and tokenization helpers
│   │
│   └───📂 config/                       # Config loading/validation
│           config_loader.py
│
├───📂 notebooks/                        # Jupyter notebooks for experimentation
│       model_2.ipynb                    # Main training notebook
│       model_eval.ipynb                 # (optional) Evaluation notebook
│
├───📂 data/                             # All data files
│       saude.jsonl                      # Training data (health Q&A)
│       (other raw/external data)
│
├───📂 models/                           # Saved model artifacts
│       └───📂 lora_adapter/             # LoRA adapter (output_dir)
│               adapter_config.json
│               adapter_model.safetensors
│
├───📂 configs/                          # Configuration files (YAML)
│       config.yaml                      # Main hyperparameter config
│       (dev.yaml, prod.yaml – optional)
│
├───📂 tests/                            # Unit/integration tests
│       test_inference.py
│       test_data.py
│
├───📂 docs/                             # Additional documentation
│       ├─── architecture.md
│       └─── api_reference.md
│
├───📂 scripts/                          # Utility scripts (not notebooks)
│       run_training.py                  # If you convert notebook to script
│       download_data.py
│
├───📂 docker/                           # Docker‑related files
│       Dockerfile
│       docker-compose.yaml
│
├───📂 .github/                          # GitHub Actions CI/CD
│       └─── workflows/
│               train.yml
│
├─── pyproject.toml                      # Project metadata & dependencies
├─── uv.lock                             # Lock file for uv
├─── requirements.txt                    # Runtime deps
├─── requirements-dev.txt                # Dev deps (testing, linting)
├─── .gitignore                          # Git ignore rules
├─── .dockerignore                       # Docker ignore rules
└─── README.md                           # Project overview
```

> **⚠️ Atenção:** O arquivo `saude.jsonl` (base de dados) **não** está incluído no pacote de entrega, conforme orientação do professor.

---

## 🛠️ Pré‑requisitos

- **Python 3.12** ou superior.
- **Pip** e **virtualenv** (recomendado).
- Opcional, mas altamente recomendado: **GPU (CUDA)** para acelerar o treino (Google Colab ou local).

---

## 🚀 Instalação e Configuração

1. **Clone o repositório** (ou descompacte o ZIP):

   ```bash
   git clone https://github.com/seu-usuario/vitalia.git
   cd vitalia
   ```
2. **Crie e ative um ambiente virtual**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```
3. **Instale as dependências**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Coloque a base de dados**:

   - Coloque o arquivo `saude.jsonl` na raiz do projeto.
   - Ou ajuste o caminho no `config.yaml` (campo `data_path`).

---

## 📝 Configuração (YAML)

Todas as configurações do treino e da geração estão centralizadas no arquivo `config.yaml`.

```yaml
# --- Parâmetros de treino ---
epochs: 10
learning_rate: 0.0001
batch_size: 4
max_length: 256
weight_decay: 0.01
label_smoothing: 0.0          # 0.0 para replicar a perda do treino manual

# --- Parâmetros LoRA ---
lora_r: 4
lora_alpha: 8
lora_dropout: 0.1
lora_target_modules: ['c_attn']

# --- Dados ---
data_path: saude.jsonl
validation_split: 0.20
seed: 42

# --- Saída ---
output_dir: models/

# --- Parâmetros de geração (inferência) ---
generation:
  max_new_tokens: 70
  do_sample: false
  num_beams: 4
  repetition_penalty: 2.2
  no_repeat_ngram_size: 4

# --- Early Stopping ---
early_stopping_patience: 3
early_stopping_threshold: 0.001
```

**Principais parâmetros**:

- `learning_rate`: controla a velocidade de convergência.
- `label_smoothing`: `0.0` garante que a perda seja calculada como CrossEntropy padrão.
- `lora_r`: rank do LoRA – valores menores (4) reduzem overfitting.
- `num_beams`: beam search para respostas mais coerentes.

## 💻 Como Executar

### 1. Treinar o Modelo (Notebook)

O notebook `vitalia_trainer.ipynb` contém todo o pipeline de treino utilizando o `Trainer` do Hugging Face.

- **Localmente**: execute `jupyter notebook notebooks/vitalia_trainer.ipynb`.
- **Google Colab**: faça upload do notebook, ative a GPU (`Runtime > Change runtime type > GPU`) e execute as células.

Ao final do treino, o melhor modelo (com menor `eval_loss`) será salvo automaticamente na pasta `models/`.

### 2. Executar a Aplicação Streamlit

A interface de chat permite conversar com o modelo treinado e ajustar parâmetros de geração em tempo real.

```bash
streamlit run app/streamlit_app.py
```

A aplicação será aberta no seu navegador. Você pode digitar perguntas sobre saúde e receber respostas geradas pelo modelo.

## 📊 Estrutura dos Dados

O arquivo `saude.jsonl` deve seguir o formato:

```json
{"instruction": "Pergunta ou comando", "input": "Contexto adicional (opcional)", "output": "Resposta esperada"}
```

- **`instruction`**: a pergunta ou tarefa.
- **`input`**: campo opcional para informações contextuais (vazio em ~87% dos exemplos).
- **`output`**: a resposta alvo para o fine‑tuning.

## 📈 Resultados e Análise

O modelo alcançou uma **perda de validação de ~2.38** após 10 épocas, com curvas de treino e validação estáveis e próximas (indicando **baixo overfitting**).

**Exemplo de respostas geradas (após pós‑processamento e Beam Search):**

| Pergunta                           | Resposta Gerada                                                                                            |
| :--------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| *Como posso reduzir o estresse?* | O estresse no dia pode ser causado por uma série de fatores.                                              |
| *O que é hipertensão?*         | A hipertensão pode ser causada por fatores genéticos, sedentarismo e excesso de sal.                     |
| *Explique o que é vacinação.* | A vacinação é uma medida de prevenção contra doenças infecciosas, como a varíola e a febre amarela. |

> **Observação:** O modelo é pequeno (124M) e a base possui ruídos de tradução. Para melhorar a precisão factual, recomenda-se a limpeza da base ou o uso de técnicas de *few‑shot prompting*.

## 🔮 Próximos Passos

- **Aprimoramento da base**: corrigir erros de tradução e adicionar mais exemplos de alta qualidade.
- **Modelo maior**: testar o fine‑tuning com `pierreguillou/gpt2-medium-portuguese` ou modelos mais recentes.
- **Integração contínua**: adicionar testes automatizados e CI/CD.

---

## 👨‍💻 Créditos

Desenvolvido por **[Seu Nome]** como parte do projeto final da disciplina **[Nome da Disciplina]**.

- **Professor(a)**: [Nome do Professor]
- **Data da Entrega**: [Sábado, 12 de Setembro]

---

## 📄 Licença

Este projeto é de uso acadêmico. Consulte o arquivo `LICENSE` para mais informações.

---

## 🧩 Arquivo `requirements.txt` (para acompanhar o README)

Caso queira gerar o `requirements.txt`:

```txt
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
accelerate>=0.24.0
datasets>=2.14.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
scikit-learn>=1.3.0
torchinfo>=1.8.0
```
