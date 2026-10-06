# Aspect-Based Sentiment Analysis Using GAN

An Aspect-Based Sentiment Analysis (ABSA) project for restaurant reviews, combining text preprocessing, exploratory data analysis, GAN-based data augmentation, and neural-network classifiers for **aspect-category** and **sentiment-polarity** prediction.

> **Repository note:** This README has been written from the actual files and notebooks in this repository. Where the project report describes a broader architecture than the code implements, this README documents the implementation that is present in the repository.

---

## Table of Contents

- [Overview](#overview)
- [What This Repository Does](#what-this-repository-does)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Aspect Categories](#aspect-categories)
- [Sentiment Polarities](#sentiment-polarities)
- [Workflow](#workflow)
- [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
- [2. Text Preprocessing](#2-text-preprocessing)
- [3. Baseline Aspect and Sentiment Models](#3-baseline-aspect-and-sentiment-models)
- [4. GAN-Based Augmentation](#4-gan-based-augmentation)
- [5. GloVe and BiLSTM Experiments](#5-glove-and-bilstm-experiments)
- [6. Augmented Google/BERT-Named Experiment](#6-augmented-googlebert-named-experiment)
- [Model Details](#model-details)
- [Training Configuration Found in the Repository](#training-configuration-found-in-the-repository)
- [Evaluation](#evaluation)
- [Reported Results](#reported-results)
- [Saved Artifacts](#saved-artifacts)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Important Reproducibility Notes](#important-reproducibility-notes)
- [Limitations and Implementation Notes](#limitations-and-implementation-notes)
- [Future Improvements](#future-improvements)
- [Project Status](#project-status)

---

## Overview

Aspect-Based Sentiment Analysis goes beyond assigning one sentiment to an entire review. It attempts to determine:

1. **Which aspect category** a review is associated with.
2. **Which sentiment polarity** is expressed.

This project focuses on restaurant reviews and uses five aspect categories:

- Food
- Service
- Ambience
- Price
- Anecdotes/Miscellaneous

The repository contains multiple experimental stages:

- exploratory data analysis,
- preprocessing,
- baseline neural classifiers,
- GAN-based synthetic data generation,
- GloVe-based embedding experiments,
- BiLSTM experiments,
- augmented-data classification.

The main implementation is notebook-based and uses TensorFlow/Keras and scikit-learn.

---

## What This Repository Does

At a high level, the project follows this pipeline:

```text
Restaurant Reviews
       |
       v
Exploratory Data Analysis
       |
       v
Text Preprocessing
       |
       v
Tokenization / Numerical Representation
       |
       +--------------------------+
       |                          |
       v                          v
Baseline Classifiers        GAN Augmentation
       |                          |
       |                          v
       |                    Synthetic Samples
       |                          |
       +------------+-------------+
                    |
                    v
       Aspect Category Prediction
                    +
          Sentiment Prediction
                    |
                    v
        Classification Metrics
```

The repository therefore contains both **baseline** and **GAN-augmented** experiments rather than one single end-to-end production pipeline.

---

# Repository Structure

```text
.
└── Code/
    ├── module_2_preprocessing.py
    ├── module_2_preprocessing.ipynb
    ├── eda.ipynb
    ├── module_3_CNN_baseline_model.ipynb
    ├── GANaugmentation.ipynb
    ├── module_4_CNN_google_bert_augmented.ipynb
    │
    ├── Restaurent_data/
    │   ├── restaurant_train_data.csv
    │   ├── augmented_data_restaurant.csv
    │   └── augmented_data_restaurant_bert.csv
    │
    ├── preprocessed_data/
    │   ├── label_encoder.pickle
    │   └── tokenizer.pickle
    │
    ├── trained_model/
    │   ├── best_model.h5
    │   └── restaurant_aspect_model.h5
    │
    └── .ipynb_checkpoints/
        └── notebook checkpoint files
```

### Main files

| File | Purpose |
|---|---|
| `eda.ipynb` | Exploratory data analysis and feature inspection |
| `module_2_preprocessing.py` | Reusable text preprocessing class |
| `module_2_preprocessing.ipynb` | Notebook version of preprocessing |
| `module_3_CNN_baseline_model.ipynb` | Baseline aspect and polarity classification |
| `GANaugmentation.ipynb` | GAN construction, training, synthetic sample generation, and augmented classification |
| `module_4_CNN_google_bert_augmented.ipynb` | Classification experiments on augmented restaurant data |
| `restaurant_train_data.csv` | Original training dataset |
| `augmented_data_restaurant.csv` | Augmented restaurant dataset |
| `augmented_data_restaurant_bert.csv` | Augmented dataset used by the BERT-named/GloVe experiments |
| `tokenizer.pickle` | Saved tokenizer artifact |
| `label_encoder.pickle` | Saved label encoder artifact |
| `best_model.h5` | Saved Keras model |
| `restaurant_aspect_model.h5` | Saved Keras aspect model |

---

# Dataset

The repository contains three CSV datasets.

## 1. Original dataset

`Code/Restaurent_data/restaurant_train_data.csv`

The inspected file contains:

- **3,044 rows**
- **5 columns**

Columns:

```text
id
text
aspect_term
aspect_category
polarity
```

The original dataset contains:

- 807 unique aspect terms
- 5 aspect categories
- 4 polarity classes

## 2. Augmented dataset

`Code/Restaurent_data/augmented_data_restaurant.csv`

The inspected file contains:

- **6,088 rows**
- **5 columns**

This is approximately twice the size of the original dataset and contains augmented examples.

## 3. Augmented/BERT-named dataset

`Code/Restaurent_data/augmented_data_restaurant_bert.csv`

The inspected file contains:

- **6,088 rows**
- **5 columns**

It has the same five-column schema:

```text
id
text
aspect_term
aspect_category
polarity
```

The notebooks use this file extensively for the GAN and augmented-model experiments.

---

# Aspect Categories

The repository uses five aspect categories:

| Category |
|---|
| `food` |
| `service` |
| `ambience` |
| `price` |
| `anecdotes/miscellaneous` |

These categories are used for the five-class aspect prediction problem.

---

# Sentiment Polarities

The repository contains four sentiment labels:

```text
positive
negative
neutral
conflict
```

The polarity model is therefore a four-class classification problem.

---

# Workflow

## 1. Exploratory Data Analysis

Notebook:

```text
Code/eda.ipynb
```

The EDA notebook examines:

- dataset shape,
- data types,
- missing values,
- descriptive statistics,
- text data,
- word frequency,
- aspect terms,
- aspect categories,
- polarity distribution,
- aspect-category/polarity relationships,
- word-cloud style text exploration,
- frequency plots and cross-tabulations.

The notebook specifically reports:

- **807 unique aspect terms** in the original dataset.
- **5 unique aspect categories**.
- **4 polarity classes**.
- Positive polarity has the highest frequency in the inspected EDA.

---

# 2. Text Preprocessing

Implementation:

```text
Code/module_2_preprocessing.py
```

The repository defines:

```python
class Data_Preprocessing
```

The preprocessing implementation includes:

1. Contraction expansion/decontraction.
2. Replacement of carriage-return/newline escape sequences.
3. Removal of non-alphabetic characters.
4. Porter stemming.
5. WordNet lemmatization.
6. Removal of short tokens according to the notebook's implemented condition.
7. Stop-word filtering.
8. Lowercasing and stripping.

The class exposes:

```python
Data_Preprocessing().preprocess_text(text_data)
```

### Important implementation note

The repository's preprocessing code is the authoritative implementation for this project. It should be used when reproducing the notebook experiments rather than substituting a different preprocessing pipeline.

---

# 3. Baseline Aspect and Sentiment Models

Notebook:

```text
Code/module_3_CNN_baseline_model.ipynb
```

Despite the notebook filename containing `CNN`, the inspected implementation does **not** contain a conventional `Conv1D`/convolutional neural-network layer.

The baseline models are Keras `Sequential` networks built primarily from `Dense` and `Dropout` layers.

## Aspect-category classifier

The baseline aspect classifier uses:

```text
Input: 6000-dimensional tokenized text representation
Dense(512, ReLU)
Dense(256, ReLU)
Dropout(0.3)
Dense(128, ReLU)
Dense(5, Softmax)
```

Compilation:

```text
Loss: categorical_crossentropy
Optimizer: Adam
Metric: accuracy
```

Training in the notebook:

```text
Epochs: 5
Validation/test split: 20%
Random state: 1
```

## Polarity classifier

The baseline polarity classifier uses:

```text
Input: 6000-dimensional tokenized text representation
Dense(512, ReLU)
Dropout(0.3)
Dense(4, Softmax)
```

Compilation:

```text
Loss: categorical_crossentropy
Optimizer: Adam
Metric: accuracy
```

Training in the notebook:

```text
Epochs: 6
Validation/test split: 20%
Random state: 1
```

Both classifiers use classification reports and confusion matrices for evaluation.

---

# 4. GAN-Based Augmentation

Notebook:

```text
Code/GANaugmentation.ipynb
```

This notebook contains the main GAN experimentation.

## Input representation

The GAN notebook:

- loads `augmented_data_restaurant_bert.csv`,
- extracts `text` and `aspect_category`,
- tokenizes text with Keras `Tokenizer`,
- uses a vocabulary limit of up to **10,000** in the main GAN experiment,
- converts text to a tokenized matrix,
- label-encodes aspect categories,
- converts labels to one-hot vectors,
- uses an **80/20** train-test split with `random_state=1`.

## Generator

The implemented generator is a fully connected network:

```text
Latent vector
    |
Dense(512)
    |
LeakyReLU
    |
BatchNormalization
    |
Dropout(0.3)
    |
Dense(1024)
    |
LeakyReLU
    |
BatchNormalization
    |
Dropout(0.3)
    |
Dense(output_dim, tanh)
```

The latent dimension used in the main GAN experiment is:

```text
100
```

## Discriminator

The discriminator is:

```text
Input
    |
Dense(512)
    |
LeakyReLU
    |
Dropout(0.3)
    |
Dense(256)
    |
LeakyReLU
    |
Dropout(0.3)
    |
Dense(1, sigmoid)
```

The discriminator uses:

```text
Loss: binary_crossentropy
Optimizer: Adam
Learning rate: 0.0002
Beta parameter: 0.5
```

## GAN training

The notebook trains the GAN using alternating discriminator and generator updates.

The inspected training function uses:

```text
Epochs: 5000
Batch size: 128
```

For each iteration:

1. Real samples are selected from the training matrix.
2. Random noise is generated.
3. The generator creates fake samples.
4. The discriminator is trained on real samples.
5. The discriminator is trained on generated samples.
6. The generator is trained through the combined GAN.
7. Progress is printed every 100 epochs.

## Synthetic data

The notebook includes:

```python
generate_synthetic_data(num_samples=500)
```

and generates:

```text
500 synthetic samples
```

These generated vectors are then combined with real training data for a downstream classifier.

---

# 5. GloVe and BiLSTM Experiments

`GANaugmentation.ipynb` also contains additional embedding/model experiments.

The notebook loads:

```text
GloVe: glove-wiki-gigaword-50
```

and constructs a **50-dimensional** embedding matrix.

A separate classifier experiment uses:

```text
Embedding
Bidirectional LSTM(128)
Bidirectional LSTM(64)
Dense(256, ReLU)
Dropout(0.3)
Dense(128, ReLU)
Dropout(0.3)
Dense(5, Softmax)
```

These experiments are present in the notebook as part of the model exploration.

They should therefore be treated as experimental components rather than assuming that they form the final production pipeline.

---

# 6. Augmented Google/BERT-Named Experiment

Notebook:

```text
Code/module_4_CNN_google_bert_augmented.ipynb
```

This notebook works with:

```text
augmented_data_restaurant_bert.csv
```

The notebook name contains `google_bert`, and the notebook includes commented BERT-related code, but the inspected executable classification pipeline primarily uses:

- Keras tokenization,
- a 6000-word vocabulary representation,
- Dense neural networks,
- Dropout,
- Softmax output layers.

Therefore, this repository should **not** be described as containing a fully fine-tuned BERT model unless the implementation is later changed.

## Aspect classifier

The notebook implements:

```text
Dense(512, ReLU)
Dense(256, ReLU)
Dropout(0.3)
Dense(128, ReLU)
Dense(5, Softmax)
```

and trains for:

```text
5 epochs
```

## Sentiment classifier

The notebook implements:

```text
Dense(512, ReLU)
Dropout(0.3)
Dense(4, Softmax)
```

and trains for:

```text
6 epochs
```

Both experiments use an 80/20 split with `random_state=1`.

---

# Model Details

## Baseline classification representation

The baseline notebooks use Keras:

```python
Tokenizer(num_words=6000)
```

and convert preprocessed text into a matrix representation using:

```python
tokenizer.texts_to_matrix(...)
```

The output is consequently a fixed-width 6000-dimensional representation for the baseline Dense classifiers.

## GAN representation

The main GAN experiment uses:

```python
Tokenizer(num_words=10000)
```

followed by:

```python
tokenizer.texts_to_matrix(...)
```

The resulting matrix dimension is used as the generator output dimension.

---

# Training Configuration Found in the Repository

| Component | Configuration |
|---|---|
| Baseline vocabulary limit | 6,000 |
| GAN vocabulary limit | 10,000 |
| Baseline split | 80% train / 20% test |
| GAN split | 80% train / 20% test |
| Random state | 1 |
| Baseline aspect epochs | 5 |
| Baseline polarity epochs | 6 |
| GAN training epochs | 5,000 |
| GAN batch size | 128 |
| GAN latent dimension | 100 |
| GAN discriminator learning rate | 0.0002 |
| GAN Adam beta | 0.5 |
| Synthetic samples generated | 500 |
| GAN downstream classifier epochs | 25 |
| GloVe embedding dimension | 50 |
| Main GAN activation | LeakyReLU / tanh / sigmoid |
| Classifier output | 5 aspect classes or 4 polarity classes |

---

# Evaluation

The notebooks evaluate classification using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

The notebooks also plot training/validation accuracy.

For the GAN downstream classifier, the notebook obtains a classification report after training the augmented classifier.

---

# Reported Results

The accompanying project report describes an overall test accuracy of approximately:

```text
91.03%
```

with:

```text
Macro F1:     0.91
Weighted F1:  0.91
Test samples: 1218
```

The report gives the following aspect-wise results:

| Aspect | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Ambience | 0.93 | 0.84 | 0.88 | 164 |
| Anecdotes/Miscellaneous | 0.89 | 0.95 | 0.92 | 400 |
| Food | 0.94 | 0.90 | 0.92 | 355 |
| Price | 0.94 | 0.88 | 0.91 | 113 |
| Service | 0.88 | 0.90 | 0.89 | 186 |

### Important distinction

These figures are **reported in the project document**. They should not automatically be interpreted as a guaranteed result of running the current repository from scratch, because the repository does not include every original training configuration, environment version, random seed, or generated training output required for exact reproduction.

---

# Saved Artifacts

The repository already contains serialized artifacts:

```text
Code/preprocessed_data/label_encoder.pickle
Code/preprocessed_data/tokenizer.pickle
```

and trained Keras models:

```text
Code/trained_model/best_model.h5
Code/trained_model/restaurant_aspect_model.h5
```

These files can be useful for inference or inspection without retraining, subject to compatibility with the TensorFlow/Keras environment that produced them.

---

# Installation

The repository does not currently include a `requirements.txt` or environment lock file.

The notebooks use packages including:

```text
numpy
pandas
scikit-learn
nltk
tqdm
tensorflow
keras
matplotlib
seaborn
gensim
xgboost
```

The notebooks also contain Colab-specific installation/mount commands.

A typical environment can be created with:

```bash
python -m venv venv
```

Activate it and install the packages used by the notebooks:

```bash
pip install numpy pandas scikit-learn nltk tqdm tensorflow keras matplotlib seaborn gensim xgboost
```

NLTK resources used by the preprocessing implementation may also need to be installed, particularly WordNet-related resources.

---

# Running the Project

Because the repository is notebook-oriented, the recommended order is:

### Step 1 — Explore the dataset

Open:

```text
Code/eda.ipynb
```

This notebook inspects the restaurant dataset and its aspect/polarity distributions.

### Step 2 — Preprocess text

Use:

```text
Code/module_2_preprocessing.py
```

or:

```text
Code/module_2_preprocessing.ipynb
```

### Step 3 — Run the baseline

Open:

```text
Code/module_3_CNN_baseline_model.ipynb
```

This trains:

- an aspect-category classifier,
- a sentiment-polarity classifier.

### Step 4 — Run GAN augmentation

Open:

```text
Code/GANaugmentation.ipynb
```

This notebook:

1. loads augmented restaurant data,
2. tokenizes the text,
3. creates the GAN,
4. trains the discriminator and generator,
5. generates synthetic vectors,
6. combines real and synthetic data,
7. trains a downstream aspect classifier,
8. evaluates the classifier.

### Step 5 — Run augmented classification experiments

Open:

```text
Code/module_4_CNN_google_bert_augmented.ipynb
```

This notebook performs additional aspect and polarity classification using the augmented dataset.

---

# Important Reproducibility Notes

The notebooks were written with Google Colab/local absolute paths in several places, for example paths beginning with:

```text
/content/gdrive/MyDrive/
```

and Windows paths such as:

```text
D:\college project\...
```

These paths will not work unchanged on another machine.

Before execution, update the dataset/import paths to the local repository paths.

For example:

```python
Code/Restaurent_data/augmented_data_restaurant_bert.csv
```

should be used instead of a machine-specific Google Drive path.

The repository also does not provide:

- `requirements.txt`
- a locked Python environment
- a complete configuration file
- a single command-line training entry point
- documented random seeds for every experiment
- a single consolidated inference script

Consequently, exact numerical reproduction may require reconstructing the original notebook environment.

---

# Limitations and Implementation Notes

## 1. The project is notebook-centric

Most of the experimentation lives in Jupyter notebooks rather than a single Python package.

## 2. The baseline notebook name is misleading

`module_3_CNN_baseline_model.ipynb` is named as a CNN model, but the inspected baseline classifiers use Dense/Dropout layers and do not implement a conventional convolutional layer.

## 3. The BERT-named notebook is not a full BERT implementation

`module_4_CNN_google_bert_augmented.ipynb` contains BERT-related commented code, but the executable classification pipeline inspected in the repository uses Keras tokenization and Dense networks.

## 4. GAN output is feature-level data

The GAN generates synthetic numerical vectors corresponding to the tokenized input representation. The repository does not show a complete text-decoding pipeline that converts those generated vectors back into natural-language restaurant reviews.

## 5. Several experiments coexist

The repository contains baseline, GAN, GloVe/BiLSTM, and augmented-data experiments. They should not be presented as one single sequential production architecture.

## 6. Environment compatibility

Some notebook code uses older Keras APIs such as `predict_classes`, and therefore may require modification when run with newer TensorFlow/Keras releases.

---

# Future Improvements

Based on the implementation and the accompanying project report, useful next steps include:

- Add a reproducible `requirements.txt`.
- Replace machine-specific paths with relative paths.
- Add a single Python training/inference entry point.
- Add explicit random seeds.
- Add automated data validation.
- Implement a genuine Transformer/BERT-based classifier if BERT is intended.
- Implement a genuine convolutional architecture if the project is to be called CNN-based.
- Integrate generated GAN samples with explicit class labels/conditions using a conditional GAN.
- Add early stopping and stronger regularization.
- Add model checkpoints and experiment configuration files.
- Add a clean inference script for new restaurant reviews.
- Add model explainability.
- Evaluate across multiple restaurant/domain datasets.
- Add multilingual experiments.
- Add tests for preprocessing and inference.

---

# Project Status

**Status: Research / academic project prototype**

The repository contains working experimental notebooks, datasets, preprocessing code, serialized preprocessing artifacts, and trained Keras model files.

It is suitable for:

- academic demonstration,
- ABSA experimentation,
- GAN-based augmentation research,
- restaurant-review sentiment analysis,
- further model development.

For production deployment, the project would benefit from packaging, dependency locking, path cleanup, reproducible training configuration, tests, and a unified inference pipeline.

---

## Summary

This repository implements an experimental **Aspect-Based Sentiment Analysis** system for restaurant reviews.

The core work covers:

```text
EDA
 ↓
Text preprocessing
 ↓
Tokenization
 ↓
Baseline aspect/sentiment classification
 ↓
GAN-based feature augmentation
 ↓
Synthetic sample generation
 ↓
Augmented classification
 ↓
Evaluation
```

The repository contains the actual experimental code and artifacts needed to continue development, while the accompanying project report provides the reported performance figures and broader research motivation.
