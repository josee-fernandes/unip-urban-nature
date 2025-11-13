# UNIP - Classificação Binária: Urban vs Nature

Projeto de classificação de imagens usando redes neurais convolucionais para distinguir entre imagens urbanas e naturais.

## 📋 Índice

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Preparação do Dataset](#preparação-do-dataset)
- [Uso](#uso)
- [Parâmetros](#parâmetros)
- [Resultados](#resultados)
- [Reprodutibilidade](#reprodutibilidade)

## 🔧 Requisitos

### Software Necessário

- **Python**: 3.8 ou superior
- **Sistema Operacional**: Linux, macOS ou Windows (testado em Linux/WSL2)

### Bibliotecas Python

Todas as dependências estão listadas em `requirements.txt`:

- `tensorflow>=2.10` - Framework de deep learning (CPU-only, sem GPU)
- `numpy` - Computação numérica
- `pandas` - Manipulação de dados
- `matplotlib` - Visualização
- `scikit-learn` - Métricas e utilitários
- `scipy` - Operações científicas
- `opencv-python` - Processamento de imagens
- `tqdm` - Barras de progresso
- `seaborn` - Visualizações estatísticas
- `scikit-image` - Processamento de imagens

## 📦 Instalação e Reprodução do Zero

Este guia leva você desde o clone do repositório até a execução completa do treinamento.

### 1. Clone o Repositório

```bash
git clone <url-do-repositorio>
cd urban-vs-nature
```

### 2. Crie um Ambiente Virtual (venv)

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Instale as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota**: Este projeto foi configurado para usar apenas CPU (sem GPU) para maximizar compatibilidade. O TensorFlow será instalado na versão CPU-only automaticamente.

### 4. Configure a API do Kaggle (Opcional, mas Recomendado)

Para baixar os datasets automaticamente, você precisa configurar a API do Kaggle:

#### 4.1. Criar Conta no Kaggle

1. Acesse https://www.kaggle.com/ e crie uma conta gratuita
2. Faça login na sua conta

#### 4.2. Obter Credenciais da API

1. Acesse https://www.kaggle.com/settings
2. Role até a seção **"API"**
3. Clique em **"Create New Token"**
4. Isso baixará um arquivo `kaggle.json` no seu computador

#### 4.3. Configurar Credenciais

**Linux/macOS:**

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**

```bash
# Criar pasta: C:\Users\<seu_usuario>\.kaggle
# Mover kaggle.json para lá
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

#### 4.4. Verificar Instalação

```bash
kaggle datasets list
```

Se funcionar, você verá uma lista de datasets. Se der erro de autenticação, verifique se o arquivo `kaggle.json` está no lugar correto.

### 5. Baixar e Organizar o Dataset

#### 5.1. Aceitar Termos dos Datasets

Antes de baixar, você precisa aceitar os termos de uso:

1. Acesse https://www.kaggle.com/datasets/heonh0/daynight-cityview
   - Clique em **"Download"** ou **"New Notebook"** e aceite os termos
2. Acesse https://www.kaggle.com/datasets/heyitsfahd/nature
   - Clique em **"Download"** ou **"New Notebook"** e aceite os termos

#### 5.2. Baixar os Datasets

**Opção A: Via API do Kaggle (Recomendado)**

```bash
# Criar pasta temporária
mkdir -p data/temp

# Baixar dataset urbano
kaggle datasets download -d heonh0/daynight-cityview -p data/temp/urban

# Baixar dataset de natureza
kaggle datasets download -d heyitsfahd/nature -p data/temp/nature

# Descompactar
cd data/temp/urban && unzip *.zip && cd ../../..
cd data/temp/nature && unzip *.zip && cd ../../..
```

**Opção B: Download Manual**

1. Acesse https://www.kaggle.com/datasets/heonh0/daynight-cityview
2. Clique em **"Download"** (após aceitar os termos)
3. Repita para https://www.kaggle.com/datasets/heyitsfahd/nature
4. Descompacte os arquivos:

```bash
mkdir -p data/temp
unzip ~/Downloads/daynight-cityview.zip -d data/temp/urban
unzip ~/Downloads/nature.zip -d data/temp/nature
```

#### 5.3. Organizar o Dataset

Use o script fornecido para organizar automaticamente (ele copia apenas uma amostra para economizar espaço):

```bash
# Organizar dataset urbano (500 imagens)
python scripts/download_kaggle_dataset.py \
    --dataset urban \
    --kaggle_path data/temp/urban \
    --output data/raw/train \
    --max_images 500

# Organizar dataset de natureza (500 imagens)
python scripts/download_kaggle_dataset.py \
    --dataset nature \
    --kaggle_path data/temp/nature \
    --output data/raw/train \
    --max_images 500
```

**Nota**: Você pode ajustar `--max_images` conforme necessário. Recomendamos 500-1000 imagens por classe para começar.

#### 5.4. Verificar Estrutura

```bash
# Verificar estrutura
ls data/raw/train/
# Deve mostrar: urban/  nature/

# Contar imagens
find data/raw/train/urban -type f | wc -l
find data/raw/train/nature -type f | wc -l
```

#### 5.5. Limpeza (Opcional)

Após organizar, você pode remover as pastas temporárias:

```bash
rm -rf data/temp
```

### 6. Executar o Treinamento

Agora você está pronto para treinar o modelo:

```bash
# Treinamento básico
python src/train.py --data_dir data/raw/train --output outputs

# Ou com parâmetros customizados
python src/train.py \
    --data_dir data/raw/train \
    --output outputs \
    --model transfer \
    --height 128 \
    --width 128 \
    --batch_size 32 \
    --epochs 30 \
    --validation_split 0.2 \
    --seed 42
```

### 7. Verificar Resultados

Após o treinamento, os resultados estarão em `outputs/`:

```bash
ls outputs/
# Deve mostrar:
# - best_model.h5
# - train_history.png
# - confusion_matrix_val.png (ou confusion_matrix_test.png)
# - roc_curve_val.png (ou roc_curve_test.png)
# - classification_report_val.csv (ou classification_report_test.csv)
```

## 📋 Resumo Rápido (Para Quem Já Tem Tudo Configurado)

Se você já tem tudo configurado e só quer executar:

```bash
# 1. Ativar venv
source venv/bin/activate

# 2. Treinar
python src/train.py --data_dir data/raw/train --output outputs
```

## 📁 Estrutura do Projeto

```
urban-vs-nature/
├── data/
│   ├── raw/
│   │   ├── train/          # Imagens de treino (obrigatório)
│   │   │   ├── urban/       # Imagens urbanas
│   │   │   └── nature/      # Imagens naturais
│   │   └── test/            # Imagens de teste (opcional)
│   │       ├── urban/
│   │       └── nature/
│   └── README_dataset.md    # Documentação do dataset
├── src/
│   ├── train.py            # Script principal de treinamento
│   ├── model_utils.py       # Utilitários de modelo
│   └── viz_utils.py         # Utilitários de visualização
├── outputs/                 # Resultados gerados
│   ├── figures/             # Gráficos e visualizações
│   ├── logs/                # Logs de treinamento
│   └── results.csv          # Métricas
├── notebooks/               # Jupyter notebooks (opcional)
├── requirements.txt         # Dependências Python
├── README.md               # Este arquivo
└── LICENSE                 # Licença do projeto
```

## 📊 Preparação do Dataset

### Estrutura de Pastas Necessária

O projeto espera que as imagens estejam organizadas em subpastas por classe. Você tem duas opções:

#### Opção 1: Apenas Pasta Train (Divisão Automática)

Se você tem apenas uma pasta com todas as imagens, organize assim:

```
data/raw/train/
├── urban/
│   ├── imagem1.jpg
│   ├── imagem2.jpg
│   └── ...
└── nature/
    ├── imagem1.jpg
    ├── imagem2.jpg
    └── ...
```

O script dividirá automaticamente em treino (80%) e validação (20%) usando o parâmetro `--validation_split`.

#### Opção 2: Pastas Train e Test Separadas (Recomendado)

Para ter controle total sobre a divisão:

```
data/raw/
├── train/
│   ├── urban/
│   └── nature/
└── test/
    ├── urban/
    └── nature/
```

Neste caso, o script usará:

- **Train**: Para treinamento (será dividido em train/val internamente)
- **Test**: Para avaliação final (não usado durante treinamento)

### Formatos de Imagem Suportados

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)

### Onde Obter um Dataset?

**Nota**: Se você seguiu a seção de [Instalação e Reprodução do Zero](#-instalação-e-reprodução-do-zero), você já deve ter o dataset configurado. Esta seção é apenas para referência.

Os datasets recomendados são:

1. **Datasets do Kaggle (Recomendado)**:

   - [Day/Night City View](https://www.kaggle.com/datasets/heonh0/daynight-cityview) - Para imagens urbanas
   - [Nature](https://www.kaggle.com/datasets/heyitsfahd/nature) - Para imagens naturais
   - **Guia completo**: Veja `scripts/KAGGLE_DOWNLOAD_GUIDE.md` para instruções detalhadas
   - **Script automático**: Use `scripts/download_kaggle_dataset.py` para organizar automaticamente
   - **Dica**: Você pode baixar apenas uma amostra (ex: 500 imagens) usando `--max_images` para não ocupar muito espaço

2. **Criar seu próprio dataset**: Colete imagens manualmente e organize nas pastas

3. **Outros datasets públicos**:
   - [ImageNet](https://www.image-net.org/) - Requer filtragem
   - [Google Open Images](https://storage.googleapis.com/openimages/web/index.html)

**Importante**: Certifique-se de que as imagens estão balanceadas entre as classes (urban e nature) para melhor performance.

## 🚀 Uso

### Execução Básica

```bash
# Com modelo de transfer learning (MobileNetV2) - padrão
python src/train.py

# Com modelo CNN simples
python src/train.py --model simple
```

### Exemplos de Execução

#### 1. Treinamento com parâmetros padrão

```bash
python src/train.py --data_dir data/raw/train --output outputs
```

#### 2. Treinamento com modelo simples e configurações customizadas

```bash
python src/train.py \
    --model simple \
    --data_dir data/raw/train \
    --output outputs \
    --height 224 \
    --width 224 \
    --batch_size 64 \
    --epochs 50 \
    --validation_split 0.2 \
    --seed 42
```

#### 3. Treinamento com transfer learning (recomendado)

```bash
python src/train.py \
    --model transfer \
    --data_dir data/raw/train \
    --output outputs \
    --height 128 \
    --width 128 \
    --batch_size 32 \
    --epochs 30 \
    --validation_split 0.2 \
    --seed 42
```

## ⚙️ Parâmetros

| Parâmetro            | Tipo  | Padrão           | Descrição                                                                          |
| -------------------- | ----- | ---------------- | ---------------------------------------------------------------------------------- |
| `--data_dir`         | str   | `data/raw/train` | Caminho para pasta train com subpastas de classes                                  |
| `--output`           | str   | `outputs`        | Diretório de saída para modelos e resultados                                       |
| `--model`            | str   | `transfer`       | Tipo de modelo: `transfer` (MobileNetV2) ou `simple` (CNN simples)                 |
| `--height`           | int   | `128`            | Altura das imagens em pixels                                                       |
| `--width`            | int   | `128`            | Largura das imagens em pixels                                                      |
| `--batch_size`       | int   | `32`             | Tamanho do batch para treinamento                                                  |
| `--epochs`           | int   | `30`             | Número de épocas de treinamento                                                    |
| `--validation_split` | float | `0.2`            | Proporção para validação (0.0-1.0). Usado apenas se não houver pasta test separada |
| `--seed`             | int   | `42`             | Seed para reprodutibilidade (random_state)                                         |

### Sobre o Parâmetro `--seed` (Random State)

O parâmetro `--seed` (também chamado de `random_state`) garante **reprodutibilidade** dos resultados. Quando você executa o mesmo código com a mesma seed, você obterá:

- Mesma divisão train/val
- Mesma inicialização dos pesos da rede
- Mesma ordem de processamento das imagens

**Por que isso é importante?**

- Permite comparar diferentes modelos de forma justa
- Facilita debugging e reprodução de resultados
- Essencial para publicações científicas

**Valor padrão**: `42` (número clássico usado em ciência de dados)

Você pode alterar para qualquer número inteiro se quiser diferentes divisões aleatórias.

## 📈 Resultados

Após o treinamento, os seguintes arquivos serão gerados em `outputs/`:

### Arquivos Gerados

1. **`best_model.h5`** - Melhor modelo salvo durante treinamento
2. **`train_history.png`** - Gráficos de loss e accuracy durante treinamento
3. **`confusion_matrix_test.png`** ou **`confusion_matrix_val.png`** - Matriz de confusão
4. **`roc_curve_test.png`** ou **`roc_curve_val.png`** - Curva ROC
5. **`classification_report_test.csv`** ou **`classification_report_val.csv`** - Métricas detalhadas (precision, recall, F1, support)

### Métricas Calculadas

- **Accuracy**: Taxa de acerto geral
- **Precision**: Precisão por classe
- **Recall**: Revocação por classe
- **F1-Score**: Média harmônica de precision e recall
- **AUC-ROC**: Área sob a curva ROC (medida de qualidade do classificador)

### Interpretação dos Resultados

- **AUC > 0.9**: Excelente classificador
- **AUC 0.8-0.9**: Bom classificador
- **AUC 0.7-0.8**: Classificador aceitável
- **AUC < 0.7**: Classificador precisa melhorias

## 🔄 Reprodutibilidade

O projeto está configurado para ser totalmente reprodutível:

1. **Seeds fixos**: Todos os geradores aleatórios usam a mesma seed
2. **Divisão determinística**: A divisão train/val é sempre a mesma com a mesma seed
3. **Inicialização fixa**: Os pesos da rede são inicializados de forma determinística

Para garantir reprodutibilidade completa:

```bash
# Sempre use a mesma seed
python src/train.py --seed 42
```

## 🐛 Troubleshooting

### Erro: "No module named 'tensorflow'"

**Solução**: Certifique-se de que o ambiente virtual está ativado e as dependências foram instaladas:

```bash
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

### Erro: "Directory not found"

**Solução**: Verifique se a estrutura de pastas está correta. O script espera:

- `data/raw/train/urban/` e `data/raw/train/nature/` (obrigatório)
- `data/raw/test/urban/` e `data/raw/test/nature/` (opcional)

### Erro: "Out of memory"

**Solução**: Reduza o `batch_size`:

```bash
python src/train.py --batch_size 16
```

### Performance lenta

**Solução**:

- Reduza o tamanho das imagens: `--height 64 --width 64`
- Use o modelo simples: `--model simple`
- Reduza o número de épocas: `--epochs 10`

## 📝 Notas Adicionais

- **GPU**: Este projeto foi configurado para CPU-only para máxima compatibilidade. Se você tiver GPU e quiser usar, instale `tensorflow-gpu` em vez de `tensorflow`, mas isso não é necessário.
- **Sistema Operacional**: Testado em Linux/WSL2. Deve funcionar em macOS e Windows, mas pode haver pequenas diferenças de caminhos de arquivos.
- **Performance**: O modelo de transfer learning (MobileNetV2) geralmente oferece melhor performance que o modelo simples, especialmente com poucos dados.

## 📄 Licença

MIT
