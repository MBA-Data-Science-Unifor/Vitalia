# 🚀 Vitalia – Assistente Inteligente para Saúde e Bem-Estar

**Vitalia** é uma aplicação **Streamlit** que permite conversar com **três checkpoints GPT‑2 pequeno em português** (`pierreguillou/gpt2-small-portuguese`) fine‑tunados com **LoRA** para o domínio de saúde, alimentação, exercício, sono e bem‑estar.

Cada modelo corresponde a uma **época diferente do mesmo treino**, permitindo comparar visualmente a evolução da qualidade das respostas:

| Modelo           | Época | Val Loss | Perplexidade | Papel didático                             |
| :--------------- | :---: | :------: | :----------: | :----------------------------------------- |
| **Ruim**         |   1   |  2.807   |    16.55     | Underfitting severo — respostas desconexas |
| **Bom**          |   5   |  2.414   |    11.18     | Ponto de equilíbrio — ~94% do ganho        |
| **Ótimo**        |  10   |  2.384   |    10.85     | Menor val_loss, sem overfitting            |

O usuário pode **alternar entre os três checkpoints** e **ajustar parâmetros de geração** (`temperature`, `top_p`, `num_beams`, `repetition_penalty`, etc.) em tempo real na interface.

---

## 📖 Visão Geral

Pipeline completo de **fine‑tuning de LLMs** aplicado a um domínio específico:

1. **Preparação dos dados** – 1.200 pares instrução‑resposta em JSONL (`saude.jsonl`).
2. **Treino com LoRA** – notebook `analysis/model_2.ipynb` (Hugging Face `Trainer` + PEFT).
3. **Salvamento por época** – callback gera `epoca_1`, `epoca_2`, …, `epoca_10`.
4. **Inferência local** – app Streamlit carrega os adaptadores e permite comparar saídas.

---

## ⚙️ Tecnologias

| Camada                 | Tecnologia                                                              |
| :--------------------- | :---------------------------------------------------------------------- |
| Linguagem              | **Python 3.12**                                                         |
| Deep learning          | **PyTorch** (CPU; CUDA opcional)                                        |
| Modelo base            | **Transformers** – `pierreguillou/gpt2-small-portuguese`                |
| Fine‑tuning eficiente  | **PEFT (LoRA)** – ~0.12% dos parâmetros treináveis (147k de 124M)       |
| Interface web          | **Streamlit**                                                           |
| Configuração           | **PyYAML**                                                              |

---

## 📁 Estrutura do Projeto

```
📂 Assistant Health/
│
├───📂 analysis/                    # Experimentos e artefatos do treino
│   ├── model_2.ipynb               # Notebook principal de treino
│   ├── curvas_por_epoca.png        # Loss treino × validação
│   ├── lr_evolution.png            # Curva do cosine scheduler
│   └── resumo_pior_bom_melhor.csv  # Métricas consolidadas
│
├───📂 app/                         # Aplicação Streamlit
│   ├── app.py                      # Interface principal (Chat + Análise)
│   ├── app_functions.py            # Carregamento do modelo + inferência
│   ├── storage.py                  # Persistência (histórico + parâmetros)
│   ├── main_params.json            # Parâmetros default
│   └── chat_history.json           # Histórico (gerado em runtime)
│
├───📂 model/                       # ⬅️ Modelos locais pré‑treinados
│   ├───📂 modelo_ruim/epoca_1/
│   ├───📂 modelo_bom/epoca_5/
│   └───📂 modelo_otimo/epoca_10/
│       └── (adapter_config.json, adapter_model.safetensors, tokenizer.json)
│
├───📂 presentation/                # Slides e material de apresentação
│
├─── config.yaml                    # Configuração central
├─── requirements.txt               # Dependências do runtime
├─── requirements-dev.txt           # Dependências de desenvolvimento
└─── README.md
```

> **⚠️ Atenção:** o arquivo `saude.jsonl` (base de treino) **não** está incluído no pacote de entrega. O app não precisa dele — só o notebook de treino.

---

## 🛠️ Pré-requisitos

- **Python 3.12+**
- **pip** e **venv**
- GPU NVIDIA com CUDA **opcional** (acelera a geração; CPU funciona bem para o GPT‑2 small)

---

## 🚀 Instalação e Execução (pip + venv)

### 1) Clone o repositório

```bash
git clone https://github.com/seu-usuario/vitalia.git
cd vitalia
```

### 2) Crie e ative o ambiente virtual

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

### 3) Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**`requirements.txt`** (runtime do app):

```txt
streamlit>=1.28.0
transformers>=4.35.0
peft>=0.7.0
accelerate>=0.24.0
safetensors>=0.4.0
huggingface-hub>=0.20.0
pyyaml>=6.0
```

**`requirements-dev.txt`** (desenvolvimento e treino):

```txt
-r requirements.txt
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.0.0
datasets>=2.0.0
matplotlib>=3.8.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

Instale as duas em sequência:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

> 💡 **PyTorch**: `torch` não está listado explicitamente — é resolvido pelo `transformers` como dependência. Para GPU NVIDIA (CUDA 12.1):
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
>
> Para CPU-only (mais leve):
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 4) Coloque os modelos locais

Estrutura esperada:

```bash
model/
├── modelo_ruim/epoca_1/adapter_config.json
├── modelo_bom/epoca_5/adapter_config.json
└── modelo_otimo/epoca_10/adapter_config.json
```

Se você acabou de treinar no notebook, mova os checkpoints:

```powershell
# PowerShell (Windows)
New-Item -ItemType Directory -Force -Path "model\modelo_ruim", "model\modelo_bom", "model\modelo_otimo"

xcopy "modelo_lora_trainer\epoca_1\*"  "model\modelo_ruim\epoca_1\"  /E /I /Y
xcopy "modelo_lora_trainer\epoca_5\*"  "model\modelo_bom\epoca_5\"   /E /I /Y
xcopy "modelo_lora_trainer\epoca_10\*" "model\modelo_otimo\epoca_10\" /E /I /Y
```

```bash
# Linux / macOS
mkdir -p model/modelo_{ruim,bom,otimo}/epoca_{1,5,10}

cp -r modelo_lora_trainer/epoca_1/*  model/modelo_ruim/epoca_1/
cp -r modelo_lora_trainer/epoca_5/*  model/modelo_bom/epoca_5/
cp -r modelo_lora_trainer/epoca_10/* model/modelo_otimo/epoca_10/
```

> 💡 **Repare no `\*`** no final do `xcopy` — sem ele, o Windows cria uma subpasta aninhada (`modelo_ruim\epoca_1\epoca_1\...`).

### 5) Verifique o `config.yaml`

```yaml
base_model: "pierreguillou/gpt2-small-portuguese"

models:
  Ruim:  "model/modelo_ruim/epoca_1"
  Bom:   "model/modelo_bom/epoca_5"
  Ótimo: "model/modelo_otimo/epoca_10"
```

### 6) Rode o Streamlit

**A partir da raiz do projeto** (`Assistant Health/`):

```bash
streamlit run app/app.py
```

A aplicação abrirá em `http://localhost:8501`.

### 7) (Opcional) Teste rápido sem UI

Salve como `test_models.py` na raiz e rode `python test_models.py`:

```python
# test_models.py
import sys; sys.path.insert(0, "app")
from app_functions import load_config, resolve_path, load_model, respond, adapter_fingerprint

cfg = load_config()
for key in ["Ruim", "Bom", "Ótimo"]:
    path = resolve_path(cfg["models"][key])
    print(f"\n=== {key} → {path} ===")
    model, tok, dev = load_model(key, path, cfg["base_model"])
    print("fingerprint:", adapter_fingerprint(model))
    r = respond(model, tok, dev,
                instruction="O que é hipertensão?",
                prompt_template=cfg["prompt_template"],
                max_new_tokens=70, do_sample=False, num_beams=4,
                repetition_penalty=2.2, no_repeat_ngram_size=4)
    print("resposta:", r)
```

Saída esperada (cada modelo com **fingerprint diferente**):

```
=== Ruim → ...\model\modelo_ruim\epoca_1 ===
fingerprint: 9f96be90 (24 tensores)
resposta: Sim, o que é a falta de oxigênio?

=== Bom → ...\model\modelo_bom\epoca_5 ===
fingerprint: a1b2c3d4 (24 tensores)   ← diferente
resposta: A hipertensão pode ser causada por uma série de fatores...

=== Ótimo → ...\model\modelo_otimo\epoca_10 ===
fingerprint: e5f6a7b8 (24 tensores)   ← diferente
resposta: A hipertensão pode ser causada por uma série de fatores...
```

---

## 📝 Configuração (`config.yaml`)

Centraliza **caminhos dos modelos**, **template de prompt** e **defaults de geração**:

```yaml
base_model: "pierreguillou/gpt2-small-portuguese"

models:
  Ruim:  "model/modelo_ruim/epoca_1"
  Bom:   "model/modelo_bom/epoca_5"
  Ótimo: "model/modelo_otimo/epoca_10"

model_epochs:          # só para exibir na UI
  Ruim:  1
  Bom:   5
  Ótimo: 10

default_model: "Bom"

# Template IDÊNTICO ao usado no treino
prompt_template: |
  ### Pergunta:
  {instruction}

  ### Resposta:

generation:
  max_new_tokens: 70
  do_sample: false
  num_beams: 4
  repetition_penalty: 2.2
  no_repeat_ngram_size: 4
  temperature: 0.7
  top_p: 0.9

training_reference:
  epochs: 10
  learning_rate: 0.0001
  batch_size: 4
  max_length: 256
  lora_r: 4
  lora_alpha: 8
  lora_dropout: 0.1
  lora_target_modules: ['c_attn']
  data_path: "saude.jsonl"
  validation_split: 0.20
  seed: 42
```

> ⚠️ O **`prompt_template` precisa ser idêntico** ao do treino. Se mudar quebras de linha ou capitalização, o modelo gera lixo — ele aprendeu a continuar a partir de `### Resposta:\n`.

---

## 💻 Como Usar a Aplicação

### Aba **💬 Chat**

1. **Sidebar**:
   - **Escolha seu Modelo**: `Ruim (época 1)`, `Bom (época 5)`, `Ótimo (época 10)`
   - **Máx. de tokens novos**: 1–200 (default 70)
   - **Usar amostragem (do_sample)**: on/off (default off)
   - **Número de Hipóteses (Beams)**: 1–10 (default 4)
   - **Penalidade de Repetição**: 1.0–10.0 (default 2.2)
   - **Temperatura**: 0.1–2.0 (default 0.7, só ativo com sampling)
   - **n‑gramas sem repetição**: 0–6 (default 4)
   - **Amostragem de Núcleo (top_p)**: 0.1–1.0 (default 0.9, só ativo com sampling)
   - **💾 Salvar Parâmetros** / **↺ Resetar** / **🔄 Recarregar modelos do disco**
2. **Digite uma pergunta** (ex.: `O que é hipertensão?`) e pressione Enter.
3. Cada resposta exibe uma etiqueta:
   ```
   🤖 Ótimo (época 10) · device=cuda · fingerprint=9f96be90 (24 tensores)
   ```
   Se o **fingerprint** mudar entre Ruim/Bom/Ótimo, o adaptador está sendo trocado corretamente.

### Painel **🔍 Debug — modelos em cache**

No expander da sidebar: caminho resolvido do adaptador atual, última resposta e modelos em memória com seus fingerprints.

### Aba **📈 Análise**

Curvas de `train_loss` e `val_loss` do notebook + tabela comparativa Ruim/Bom/Ótimo.

---

## 🧪 Cenários de Teste

| Pergunta                                                | Ruim (ép. 1)                              | Bom (ép. 5)                                                     | Ótimo (ép. 10)                                                                                  |
| :------------------------------------------------------ | :---------------------------------------- | :-------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- |
| `Como posso reduzir o estresse no dia a dia?`           | Resposta desconexa                        | Coerente, genérica                                              | Coerente, específica                                                                            |
| `O que é hipertensão?`                                  | `Sim, o que é a falta de oxigênio?`       | `A hipertensão pode ser causada por uma série de fatores...`    | Similar ao Bom, com detalhes finais diferentes                                                  |
| `Quais são os benefícios de uma boa noite de sono?`     | Divaga sobre banco de areia               | `Um bom Night de sono pode ajudar a reduzir o estresse...`      | `A melhor noite de sono é um bom dia de trabalho e descanso.`                                   |
| `Explique o que é vacinação.`                           | Genérico (gripe, tuberculose)             | `O que é vacinação é a vacinação contra...`                     | `A vacinação é uma medida de prevenção contra doenças infecciosas, como a varíola e a febre amarela.` |

> 💡 **Dica de comparação justa:** mantenha os parâmetros de geração **iguais** e mude **apenas** o modelo.

### Erros comuns

| Sintoma                                                | Causa provável                                                                  | Correção                                                                                        |
| :----------------------------------------------------- | :------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------- |
| `Can't find 'adapter_config.json' at '...'`            | Pasta aninhada (xcopy criou `modelo_bom\epoca_5\epoca_5\`)                      | Use `xcopy "...\epoca_5\*"` com `\*` no final, ou ajuste o `config.yaml`                        |
| As 3 respostas são idênticas                           | Mesmo adaptador carregado (caminhos iguais no `config.yaml`)                    | Confirme os fingerprints no painel de debug — devem ser diferentes                              |
| Resposta vira eco da pergunta                          | Formatação da pergunta diferente do treino                                      | Use a mesma forma: `O que é hipertensão?`                                                       |
| `RuntimeError: Expected all tensors on same device`    | Modelo e inputs em devices diferentes                                           | Já tratado pelo `_resolve_device()` — reinicie o app                                            |
| App lento na 1ª pergunta                               | Cache vazio — baixando GPT‑2 + carregando LoRA                                  | Normal (5–15 s). Depois fica instantâneo (`@lru_cache`)                                         |

---

## 📊 Estrutura dos Dados de Treino

O arquivo `saude.jsonl` (usado **apenas no notebook**) segue o formato:

```json
{"instruction": "O que é hipertensão?", "input": "", "output": "A hipertensão é uma condição crônica..."}
```

- **`instruction`**: pergunta ou comando.
- **`input`**: contexto adicional (vazio em ~87% dos exemplos).
- **`output`**: resposta alvo.

Prompt final montado pelo notebook:

```
### Contexto:
{input}

### Pergunta:
{instruction}

### Resposta:
{output}<eos>
```

O bloco `### Contexto:` só aparece quando `input` não é vazio.

---

## 🔁 Reproduzindo o Treino

1. Coloque o `saude.jsonl` na raiz do projeto.
2. Abra `analysis/model_2.ipynb` no Jupyter ou no Google Colab (com GPU).
3. Execute todas as células.

Os checkpoints ficam em `modelo_lora_trainer/epoca_1..10/`. Mova para `model/` como descrito no passo 4.

**Configuração do treino** (`config.yaml` → seção `training_reference`):

| Parâmetro              |     Valor     | Justificativa                                   |
| :--------------------- | :-----------: | :---------------------------------------------- |
| `learning_rate`        |    `1e-4`     | Equilíbrio seguro para LoRA                     |
| `batch_size`           |     `4`       | Cabe em GPU de 4 GB                             |
| `max_length`           |    `256`      | Captura respostas inteiras                      |
| `lora_r`               |     `4`       | Baixo rank → menos overfitting em base pequena  |
| `lora_alpha`           |     `8`       | 2×r, escala padrão                              |
| `lora_dropout`         |    `0.1`      | Regularização                                   |
| `lora_target_modules`  | `['c_attn']`  | Camadas de atenção do GPT‑2 (Conv1D)            |
| `epochs`               |     `10`      | Com early stopping (patience 3)                 |

---

## 🔮 Próximos Passos

- **Aprimorar a base**: corrigir ruídos de tradução e adicionar mais exemplos de alta qualidade.
- **Modelos maiores**: testar `pierreguillou/gpt2-medium-portuguese` ou modelos mais recentes (Sabiá, Qwen).
- **Side‑by‑side**: mostrar as respostas dos 3 modelos em colunas paralelas.
- **Métricas automáticas**: BLEU/ROUGE contra respostas de referência.
- **Cache de respostas**: por (modelo, pergunta, parâmetros).

---

## 👨‍💻 Créditos

Desenvolvido por **[Seu Nome]** como projeto final da disciplina **[Nome da Disciplina]**.

- **Professor(a):** [Nome do Professor]
- **Data da Entrega:** [Sábado, 12 de Setembro]

---

## 📄 Licença

Uso acadêmico. Consulte o arquivo `LICENSE` para mais informações.

---

## 📚 Referências

- [PEFT — Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Streamlit Docs](https://docs.streamlit.io/)
- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)

---

## 🔑 Resumo das mudanças

**O que foi removido:**
- Toda a seção **Docker** (Dockerfile, docker-compose, `.dockerignore`, comandos `docker compose`, verificação pós-build, tabela de erros Docker).
- Duplicação de dependências (o `requirements.txt` aparecia como bloco separado e dentro do texto).

**O que foi adicionado/melhorado:**
- Seção **`requirements.txt` + `requirements-dev.txt`** com conteúdo sugerido e comando único de instalação.
- Bloco sobre instalação de `torch` (CPU vs CUDA) mais visível.
- Troubleshooting enxugado (removidas entradas específicas de Docker).
- Estrutura do projeto sem `Dockerfile`/`docker-compose.yml`/`.dockerignore`.
- Referências mantidas, seção “Próximos Passos” encurtada, formatação de tabelas padronizada.