# Towards Explainable Dialogue System (TXDS)

**Explaining intent classification using saliency techniques**

**Authors:** Joshi, Ratnesh Kumar, Arindam Chatterjee, and Asif Ekbal

**Conference:** Proceedings of the 18th International Conference on Natural Language Processing (ICON). 2021

## Overview

This repository contains the implementation of an explainable dialogue system that explains intent classification decisions using saliency techniques. The system uses bidirectional LSTM networks with attention mechanisms to classify dialogue intents and provides word-level importance scores for interpretability. The work includes implementations on three multi-domain datasets: ATIS, Multidog Airline, and Multidog Finance.

## Repository Structure

```
TXDS/
├── run.py                                    # Main training and evaluation script
├── IMDB_Review_Sentiment_Analyzer_Explainable.ipynb  # Sentiment analysis demo
├── xaiAtis.ipynb                            # ATIS dataset exploration
├── xaiMultidogoAirline.ipynb               # Multidog Airline exploration
├── xaiMultidogoFinance.ipynb               # Multidog Finance exploration
├── data/
│   ├── train.csv, val.csv, test.csv        # IMDB sentiment data
│   └── multidogoSentLevel/                 # Multi-domain intent data
│       ├── airline/                        # Airline domain
│       │   ├── train.tsv, val.tsv, test.tsv
│       ├── finance/                        # Finance domain
│       │   ├── train.tsv, val.tsv, test.tsv
│       ├── fastfood/, insurance/, media/, software/  # Other domains
│       └── ...
├── models/                                 # Trained model artifacts
│   ├── imdb_bi_lstm_big_tensorflow_model.json
│   ├── imdb_bi_lstm_big_tensorflow_model.h5
│   └── ...
└── Results/                                # Output visualizations
    ├── Results1.png
    ├── Results2.png
    └── Results3.png
```

## Datasets

### IMDB Movie Reviews (Sentiment Classification)
- **Size:** 50,000 reviews (25,000 train, 25,000 test split in code)
- **Task:** Binary sentiment classification (positive/negative)
- **Vocab Size:** 50,000 most frequent words
- **Format:** Integer-encoded sequences

### ATIS (Airline Travel Information System)
- **Size:** ~5,000 sentences across multiple intents
- **Domains:** Single domain (flight information)
- **Intents:** Flight booking, fare, airfare, flight service, ground service, etc.
- **Format:** TSV format with intent labels

### Multidogo Datasets (6 domains)
- **Airline:** Flight booking and travel information
- **Finance:** Financial services and transactions
- **FastFood:** Restaurant orders and queries
- **Insurance:** Insurance inquiries and policies
- **Media:** Entertainment and media recommendations
- **Software:** Software support and troubleshooting

Each domain contains sentence-level intent labels and multi-turn dialogue context.

## Prerequisites

```bash
# Core dependencies
pip install keras>=2.3.0
pip install tensorflow>=2.2.0
pip install tensorflow-gpu  # For GPU support

# ML and NLP libraries
pip install scikit-learn>=0.22.0
pip install numpy>=1.19.0
pip install scipy>=1.5.0

# Text processing
pip install textblob>=0.15.3
pip install nltk>=3.5

# Explainability
pip install deepexplain>=0.1.0
pip install lime>=0.1.1.33

# Data handling
pip install pandas>=1.0.0

# Visualization
pip install matplotlib>=3.0.0
pip install seaborn>=0.10.0
```

For GPU support:
```bash
pip install tensorflow-gpu>=2.2.0
# Or use CUDA-enabled PyTorch with DeepExplain backend
```

## Running the Code

### 1. IMDB Sentiment Classification with Explainability

#### Main Training & Inference Script

```bash
python run.py
```

**Key Hyperparameters in `run.py`:**

**Data Configuration:**
- **Vocabulary Size:** 50,000 most frequent words
- **Max Sequence Length:** 200 tokens
- **Number of Classes:** 2 (positive/negative sentiment)
- **Data Source:** IMDB dataset (auto-downloaded first run)

**Model Architecture:**
- **Embedding Layer:**
  - Input dimension: 50,000
  - Output dimension: 256
  - Name: `embedding_layer`

- **Bidirectional LSTM:**
  - Hidden units: 256
  - Return sequences: True
  - Merge mode: Concatenate (output size: 512)
  - Name: `bidi_lstm_layer`
  - Recurrent Dropout: 0.0 (p_W = 0)
  - Dropout: 0.0 (p_U = 0)
  - Regularizers: L2 with weight_decay = 0 (configurable)

- **Unidirectional LSTM:**
  - Hidden units: 256
  - Return sequences: False
  - Name: `lstm_layer`

- **Dense Layers:**
  - Dense 1: 256 units
  - Dense 2: 2 units (output layer)
  - Activation: Softmax

- **Dropout:** 0.5 after LSTM layers (name: `dropout`)

**Training Configuration:**
- **Loss Function:** Categorical crossentropy
- **Optimizer:** Adam (default learning rate 0.001)
- **Batch Size:** 25
- **Epochs:** 20
- **Validation Split:** Auto (test set used)
- **Metrics:** Accuracy

**Model Variants:**
The script saves 3 model versions:
1. `imdb_bi_lstm_big_tensorflow_model.json` + `.h5`: Full training checkpoint
2. `imdb_bi_lstm_big_tensorflow_model_NOT.json` + `.h5`: Alternative variant
3. `imdb_bi_lstm_big_tensorflow_model_ALL.json` + `.h5`: Complete model

**Explainability Configuration:**

The script computes word-level saliency/importance through:

**Saliency Computation:**
- Framework: DeepExplain (TensorFlow backend)
- Method: Integrated Gradients or Deconvolution
- Pooling: Sum across hidden dimension to get word importance
- Normalization: Min-max scaling to [0, 1]

**Polarity-Aware Scoring:**
```python
polarityHighThreshold = 0.4  # Strong polarity threshold
polarityLowThreshold = 0.0   # Minimum polarity threshold
```

Uses TextBlob sentiment polarity:
- Ranges from -1.0 (very negative) to +1.0 (very positive)
- Combined with saliency for refined interpretation

**Output Explanations:**
- Word lists with original relevance scores
- Normalized relevance scores
- Polarity-adjusted scores
- Visual output of important words per sample

#### Model Selection

Choose which trained model to evaluate by modifying `modelNum`:

```python
modelNum = 1  # Load model variant 1
# or
modelNum = 2  # Load model variant 2
# or
modelNum = 3  # Load all variants
```

#### Inference on Custom Text

The script includes a custom inference function:

```python
def getIndexArray(inputStr):
    # Preprocesses input: removes punctuation, stopwords
    # Converts words to indices
    # Pads to maxWords (200)
    words = preProcessQuery(inputStr)
    wordIndexList = np.array([word2Index[word] if word in word2Index else 0 for word in words])
    wordIndexArray = pad_sequences(..., maxlen=maxWords)
    return wordIndexArray

def getPrediction(inputStr):
    wordIndexArray = getIndexArray(inputStr)
    predictionScore = model.predict(wordIndexArray[0:1])
    prediction = getSentiment(predictionScore)
    return prediction, predictionScore
```

### 2. Multi-Domain Intent Classification

For multi-domain intent classification (ATIS, Multidog datasets):

#### Data Exploration & Analysis

```bash
# Explore ATIS dataset
jupyter notebook xaiAtis.ipynb

# Explore Airline domain
jupyter notebook xaiMultidogoAirline.ipynb

# Explore Finance domain
jupyter notebook xaiMultidogoFinance.ipynb
```

These notebooks provide:
- Data loading and exploration
- Intent distribution analysis
- Token/sequence length statistics
- Domain-specific insights

#### Training on Custom Datasets

To train on your own dataset (ATIS or Multidog format):

1. Prepare data in TSV format:
```
text<TAB>intent_label
```

2. Modify data loading section in `run.py`:
```python
# Load custom data
dataset = load_dataset('csv', data_files={'train': 'path/to/train.tsv', ...})
```

3. Adjust model hyperparameters:
- `topWords`: Vocab size for your domain
- `maxWords`: Max sequence length (adjust to dataset)
- `nb_classes`: Number of intents (not 2)

4. Run training:
```bash
python run.py
```

### 3. Interactive Explainability Demo

For interactive exploration:

```bash
jupyter notebook IMDB_Review_Sentiment_Analyzer_Explainable.ipynb
```

This notebook provides:
- Step-by-step sentiment analysis
- Word importance visualization
- Interactive testing on new reviews
- Saliency map visualization
- Detailed explanation generation

## Data Format

### Training Data Format

**IMDB (Sentiment):**
- Integer-encoded word sequences
- Auto-downloaded via Keras API
- Pad to 200 tokens

**ATIS/Multidog (Intent):**
```
text	intent
book a flight to new york	flight_booking
what's the fare to boston	fare_inquiry
i need help with luggage	baggage_service
```

Expected columns:
- `text`: Input sentence/query
- `intent`: Intent label (multi-class)

### Data Preprocessing Pipeline

**Steps in `run.py`:**
1. Load and decode word index
2. Remove stopwords (129 common words)
3. Remove punctuation
4. Convert to lowercase
5. Pad/truncate to `maxWords` (200)

**Stop Words:**
Includes: a, an, the, is, are, will, can, should, if, or, because, between, for, with, about, after, before, during, over, under, than, than, too, so, etc.

## Output Files

### Model Artifacts
- `imdb_bi_lstm_big_tensorflow_model.json`: Model architecture
- `imdb_bi_lstm_big_tensorflow_model.h5`: Model weights
- `imdb_bi_lstm_big_tensorflow_model_ALL.{json,h5}`: Complete model

### Results & Analysis
- Console output with sample-level predictions and saliency
- Word importance rankings
- Polarity-adjusted scores
- Correctness assessment

### Visualizations
- `Results1.png`, `Results2.png`, `Results3.png`: Generated saliency visualizations
- Heatmaps of word importance
- Attention weight distributions

## Key Hyperparameters & Configurations

| Parameter | Value | Notes |
|-----------|-------|-------|
| Top Words (Vocab) | 50,000 | Most frequent word types |
| Max Sequence Length | 200 | Pad/truncate to this length |
| Embedding Dimension | 256 | Word embedding size |
| LSTM Hidden Dimension | 256 | Hidden state size (BiLSTM outputs 512) |
| Dense Hidden Dimension | 256 | Hidden layer in classification head |
| Dropout Rate | 0.5 | Dropout after LSTM |
| Batch Size | 25 | Training batch size |
| Epochs | 20 | Number of training epochs |
| Learning Rate | 0.001 | Adam optimizer default |
| Optimizer | Adam | Adaptive learning rate |
| Loss | Categorical Crossentropy | Multi-class loss |

## Interpretability & Explanation Details

### Saliency Computation

The system computes saliency (word importance) through:

**Method 1: Gradient-Based Saliency**
- Computes gradients of output with respect to input embeddings
- Aggregates across embedding dimensions
- Identifies which words most influence the prediction

**Method 2: Polarity-Enhanced Saliency**
```python
# TextBlob polarity: [-1.0, +1.0]
base_saliency = gradient_magnitude
polarity = textblob_sentiment_polarity(word)

if abs(polarity) > 0.4:  # Strong sentiment
    if (prediction == positive AND polarity > 0) OR (prediction == negative AND polarity < 0):
        final_saliency = abs(polarity)  # Agree with prediction
    else:
        final_saliency = -abs(polarity)  # Contradict prediction
```

### Output Interpretation

For each test case:
1. **Original Relevance**: Raw importance scores
2. **Normalized Relevance**: Min-max scaled to [-1, 1]
   - Positive: Supports the prediction
   - Negative: Contradicts the prediction

### Word Ranking Example

```
Word Importance for Positive Classification:
1. excellent    +0.95   (strong positive word)
2. great        +0.87   (positive word)
3. amazing      +0.91   (positive word)
4. bad          -0.45   (contradicts prediction)
5. boring       -0.38   (contradicts prediction)
```

## Configuration Notes

### Hardware Requirements

- **GPU Memory:** 2GB+ (LSTM models are memory-efficient)
- **RAM:** 8GB+ (for data loading and preprocessing)
- **CPU:** Multi-core recommended (4+ cores)
- **Disk Space:** ~2GB (models + data + Word2Vec embeddings)

### Performance Estimates

- **Training Time:**
  - IMDB (50K samples): 30-60 minutes (GPU)/3-5 hours (CPU)
  - ATIS (5K samples): 5-10 minutes (GPU)/30-60 minutes (CPU)
  - MultiWOZ (10K samples): 15-30 minutes (GPU)/1-2 hours (CPU)

- **Inference Speed:**
  - Single sample: 100-500ms (CPU), 10-50ms (GPU)
  - Batch of 32: 2-5 seconds (CPU), 0.5-1 second (GPU)
  - Saliency computation: 2-3x slower than inference

### Optimization Tips

- **Memory Optimization:**
  - Reduce `maxWords` from 200 to 150
  - Reduce embedding dim from 256 to 128
  - Reduce LSTM units to 128

- **Speed Optimization:**
  - Increase batch size from 25 to 64-128
  - Reduce epochs from 20 to 10-15
  - Skip saliency computation during training

- **Better Accuracy:**
  - Increase LSTM units to 512
  - Use bi-directional attention layers
  - Ensemble multiple models
  - Add regularization (L1/L2)

## Example Usage

### Sentiment Analysis

```python
from run import model, tokenizer, word2Index

# Predict sentiment
text = "This movie is absolutely fantastic and I loved every moment of it!"
pred, score = getPrediction(text)
print(f"Prediction: {pred}")  # Output: Positive
print(f"Confidence: {score}")  # [0.05, 0.95]

# Get word-level explanations
words, relevances, normalized = getWordsAndRelevances(text)
for word, rel in zip(words, normalized):
    print(f"{word}: {rel:.3f}")
```

### Intent Classification

Adapt the code for multi-class:
```python
nb_classes = 10  # Number of intents
# Add final Dense layer with softmax activation
# Categorical crossentropy loss
```

## Training Curves & Metrics

Expected performance on IMDB dataset:
- **Accuracy:** 87-91%
- **Loss:** 0.25-0.35 (final epoch)
- **Training Time:** 45-60 minutes (single GPU)

Expected performance on ATIS:
- **Accuracy:** 93-95% (well-defined intents)
- **Training Time:** 5-10 minutes

## References

For detailed methodology and results, refer to the paper:

> Joshi, R. K., Chatterjee, A., & Ekbal, A. (2021). "Towards explainable dialogue system: Explaining intent classification using saliency techniques." *Proceedings of the 18th International Conference on Natural Language Processing (ICON)*, 2021.

## Code Architecture

```
Input Text
  ↓
Preprocessing (stopword removal, punctuation)
  ↓
Index Conversion (word2index mapping)
  ↓
Padding/Truncation (max 200 tokens)
  ↓
Embedding Layer (word2vec lookup, 256-dim)
  ↓
BiLSTM Layer (forward & backward passes)
  ↓
Dense Layers (2 layers, 256→2 units)
  ↓
Softmax Output (probability distribution)
  ↓
Prediction + Saliency Computation
  ↓
Explainability Report (word importance scores)
```

## Troubleshooting

### IMDB Download Issues
```python
# If auto-download fails, manually load:
import pickle
with open('path/to/imdbData.pickle', 'rb') as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f)
```

### Memory Issues
- Reduce `topWords` from 50K to 10K
- Reduce `maxWords` from 200 to 100
- Reduce batch size from 25 to 10

### Model Loading Issues
```python
# Ensure model path exists
import os
os.makedirs('models', exist_ok=True)
```

## Dependencies Version Notes

- **Keras:** 2.3.0+
- **TensorFlow:** 2.2.0+ (GPU support optional)
- **Scikit-learn:** 0.22.0+
- **DeepExplain:** Latest version
- **Python:** 3.6+ (recommend 3.8+)

## Contact & Contributions

For questions or issues:
- Open an issue in the repository
- Include code snippets, error messages, and dataset information
- State Python/TensorFlow versions used
