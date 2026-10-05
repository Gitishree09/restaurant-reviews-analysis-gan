# Aspect-Based Sentiment Analysis Using GAN Model

A GAN-based framework for **Aspect-Based Sentiment Analysis (ABSA)** designed to improve fine-grained sentiment classification when labelled data is limited. The project combines **Generative Adversarial Networks (GANs)** with contextual representations and aspect-sentiment feature learning to generate realistic synthetic samples, improve robustness, and classify sentiment associated with specific review aspects.

> **Project focus:** Restaurant review analysis  
> **Dataset:** Yelp restaurant reviews  
> **Model:** GAN-based ABSA / GAN-BERT-oriented architecture  
> **Reported test accuracy:** **91.03%**  
> **Macro F1-score:** **0.91**  
> **Weighted F1-score:** **0.91**

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Contributions](#key-contributions)
- [How the Model Works](#how-the-model-works)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Aspect Categories](#aspect-categories)
- [Sentiment Classes](#sentiment-classes)
- [Data Augmentation](#data-augmentation)
- [Model Configuration](#model-configuration)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Aspect-wise Performance](#aspect-wise-performance)
- [Training Behaviour](#training-behaviour)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Advantages](#advantages)
- [Limitations and Observations](#limitations-and-observations)
- [Future Scope](#future-scope)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Reproducibility Notes](#reproducibility-notes)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

## Overview

Traditional sentiment analysis generally assigns an overall sentiment such as positive, negative, or neutral to an entire sentence or document. This can miss the fact that a single review may express different opinions about different parts of the same entity.

**Aspect-Based Sentiment Analysis (ABSA)** addresses this problem by identifying specific aspects and determining the sentiment associated with each aspect.

For example:

> "The food was excellent, but the service was slow."

A conventional sentiment classifier may treat the sentence as a single positive/negative mixture. An ABSA system instead identifies:

| Aspect | Sentiment |
|---|---|
| Food | Positive |
| Service | Negative |

This project investigates the use of **Generative Adversarial Networks (GANs)** to improve ABSA, particularly in low-resource and data-sparse scenarios.

The proposed approach uses a generator to create synthetic aspect-sentiment feature representations and a discriminator to distinguish real samples from generated samples. The framework is intended to improve data augmentation, robustness, and generalization.

---

## Problem Statement

ABSA faces several challenges:

- Limited availability of labelled datasets.
- Data sparsity across domains and languages.
- Imbalanced sentiment and aspect distributions.
- Difficulty aligning an aspect with its correct sentiment expression.
- Implicit opinions and complex linguistic constructions.
- Noisy or ambiguous user-generated reviews.
- Reduced generalization when training data is limited.

The project addresses these challenges by using adversarial learning to generate additional aspect-sentiment representations and improve the robustness of the classification model.

---

## Objectives

The main objectives of this project are:

1. Build a GAN-based framework for Aspect-Based Sentiment Analysis.
2. Generate realistic aspect-sentiment representations for data augmentation.
3. Improve sentiment classification when labelled data is limited.
4. Improve the association between aspects and their corresponding sentiments.
5. Incorporate syntactic and contextual information into the ABSA pipeline.
6. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrices.
7. Study model performance across multiple restaurant-review aspect categories.

---

## Key Contributions

According to the project study, the major contributions are:

- A **GAN-based ABSA architecture** capable of generating realistic and contextually coherent aspect-sentiment pairs.
- Integration of **syntactic dependency parsing** with **multi-head attention** to model relationships between aspects and sentiments.
- A unified generative formulation covering major ABSA subtasks, including aspect extraction, opinion extraction, and sentiment classification.
- Evaluation on a restaurant-review dataset demonstrating strong classification performance.
- Improved robustness and data efficiency in settings with limited labelled data.

---

## How the Model Works

The proposed framework is based on the adversarial learning principle of GANs.

A GAN consists primarily of:

- **Generator (G)**
- **Discriminator (D)**

### Generator

The generator receives a random noise vector `z` and produces synthetic feature representations:

```text
z → Generator → Synthetic Aspect-Sentiment Representation
```

The goal of the generator is to produce representations that resemble real aspect-sentiment features.

### Discriminator

The discriminator receives both real and generated representations:

```text
Real Representation ───────┐
                           ├──→ Discriminator → Real / Fake
Generated Representation ──┘
```

The discriminator learns to distinguish between genuine and synthetic representations.

### Adversarial Training

The two components are trained competitively:

```text
                 ┌────────────────────┐
                 │ Random Noise Vector │
                 │        z            │
                 └─────────┬──────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Generator   │
                    │      G       │
                    └──────┬───────┘
                           │
                           ▼
                Synthetic Feature Vector
                           │
                           ▼
                 ┌──────────────────┐
Real Features ──►│  Discriminator D │
                 └────────┬─────────┘
                          │
                    Real / Fake
```

The generator attempts to fool the discriminator, while the discriminator attempts to correctly identify real and generated samples.

---

## Architecture

The project describes an ABSA architecture that combines adversarial learning with contextual and syntactic information.

The major components are:

1. **Text preprocessing**
2. **Aspect and sentiment representation**
3. **Contextual embeddings**
4. **Syntactic dependency information**
5. **Generator**
6. **Discriminator**
7. **Sentiment classification**
8. **Evaluation**

The study describes the use of contextual representations such as **BERT** and discusses dependency parsing and multi-head attention for capturing semantic and structural relationships.

### Conceptual Pipeline

```text
                 Restaurant Review
                        │
                        ▼
                Text Preprocessing
                        │
                        ▼
             Aspect / Sentiment Data
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
      Contextual Embedding   Dependency Structure
          (e.g. BERT)              │
              │                    │
              └─────────┬──────────┘
                        ▼
                 Feature Representation
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
        Generator             Real Samples
             │                     │
             ▼                     │
       Synthetic Samples           │
             │                     │
             └──────────┬──────────┘
                        ▼
                  Discriminator
                        │
                        ▼
             Classification / ABSA
                        │
                        ▼
            Aspect + Sentiment Output
```

---

## Dataset

The experiments use a customized and augmented restaurant-review dataset named:

```text
augmented_data_restaurant.csv
```

The source study describes the dataset as being collected from the **Yelp** platform.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total entries | 6,088 |
| Columns | 5 |
| Non-null ID values | 3,044 |
| Non-null aspect terms | 4,103 |
| Non-null aspect categories | 6,088 |
| Non-null polarity labels | 6,088 |

The dataset contains restaurant reviews annotated with aspect categories and sentiment polarity.

---

## Aspect Categories

The model works with five major aspect categories:

1. **Ambience**
2. **Anecdotes/Miscellaneous**
3. **Food**
4. **Price**
5. **Service**

These categories represent common areas discussed in restaurant reviews.

---

## Sentiment Classes

The document describes sentiment labels including:

- **Positive**
- **Negative**
- **Neutral**

The dataset description also mentions **conflict** in its annotation discussion, while the summarized dataset statistics list positive, negative, and neutral as the polarity classes used in the reported dataset overview.

---

## Data Augmentation

To address class imbalance and limited data, the project applies data augmentation techniques including:

- **Synonym replacement**
- **Back translation**
- GAN-based synthetic feature generation

The GAN component is intended to enrich the training distribution by producing synthetic aspect-sentiment representations.

This is particularly useful for:

- Low-resource settings
- Sparse classes
- Noisy review data
- Domain adaptation
- Improving generalization

---

## Model Configuration

The study reports the following parameter analysis and selected configuration:

| Parameter | Tested | Used |
|---|---|---|
| Dense layers | 2–4 | 3 Generator, 3 Discriminator |
| Optimizer | Adam, AdamW | Adam |
| Activation functions | ReLU, LeakyReLU, ELU, Swish, Tanh, Softmax | LeakyReLU, Tanh, Softmax |
| Tokenizer | WordPiece, BPE | WordPiece |
| Learning rate | `1e-3`, `5e-4`, `1e-4`, `2e-5` | Default Adam configuration |
| Batch size | 8, 16, 32, 64 | 16 (embedding), 32 (training) |
| Latent dimension | 64, 128, 256 | 128 |
| Epochs | 10–100 | 25 |
| Embedding strategy | CLS, pooled, average token embeddings | CLS token |
| Train/validation split | 70/30, 80/20, full training | Full training |

The reported evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

---

## Training

The GAN training process follows the adversarial learning setup.

### Step 1 — Prepare Data

Restaurant reviews are processed to obtain aspect-sentiment information.

### Step 2 — Create Representations

Aspect-sentiment pairs are converted into fixed-size feature representations using embedding techniques. The study discusses representations such as Word2Vec, GloVe, TF-IDF, positional encoding, and contextual transformer representations, with **BERT CLS representations** selected in the reported parameter configuration.

### Step 3 — Train the Discriminator

The discriminator is trained using:

- Real embeddings → label `1`
- Generated embeddings → label `0`

### Step 4 — Train the Generator

The generator attempts to create synthetic embeddings that the discriminator classifies as real.

### Step 5 — Alternate Optimization

Generator and discriminator updates are alternated until the generator produces increasingly realistic samples.

---

## GAN Loss Functions

The study describes the standard GAN minimax objective:

```text
min_G max_D E[x~pdata] [log D(x)]
              + E[z~pz] [log(1 - D(G(z)))]
```

### Discriminator Loss

```text
LD = -E[x~pdata][log D(x)]
     -E[z~pz][log(1 - D(G(z)))]
```

### Generator Loss

```text
LG = -E[z~pz][log D(G(z))]
```

When sentiment classification is incorporated, an additional cross-entropy loss is used for the supervised classification component.

---

## Evaluation Metrics

The project evaluates the model using several standard classification metrics.

### Accuracy

Measures the proportion of correctly classified instances among all instances.

### Precision

Measures how many predicted positive instances are actually positive.

### Recall

Measures how many actual positive instances are correctly identified.

### F1-Score

The harmonic mean of precision and recall:

```text
F1 = 2 × (Precision × Recall)
     -------------------------
       Precision + Recall
```

F1-score is especially useful when class distributions are imbalanced.

### Confusion Matrix

A confusion matrix provides a class-by-class view of correct predictions and misclassifications.

### Macro Average

Computes the metric independently for each class and then averages the results, treating all classes equally.

### Weighted Average

Computes the metric for each class and weights each class according to its number of instances.

---

## Results

The reported model achieves strong performance on the restaurant-review test set.

### Overall Performance

| Metric | Score |
|---|---:|
| Test accuracy | **91.03%** |
| Macro F1 | **0.91** |
| Weighted F1 | **0.91** |
| Test samples | **1,218** |

The document also reports an overall accuracy of approximately **91%** in its result analysis.

---

## Aspect-wise Performance

The reported aspect-level results are:

| Aspect | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Ambience | 0.93 | 0.84 | 0.88 | 164 |
| Anecdotes/Miscellaneous | 0.89 | 0.95 | 0.92 | 400 |
| Food | 0.94 | 0.90 | 0.92 | 355 |
| Price | 0.94 | 0.88 | 0.91 | 113 |
| Service | 0.88 | 0.90 | 0.89 | 186 |

### Best Performing Categories

The highest reported F1-score is **0.92**, achieved by:

- Anecdotes/Miscellaneous
- Food

### Areas for Improvement

The lowest aspect-level F1-score is reported for **Ambience (0.88)**.

Its recall of **0.84** suggests that some relevant ambience instances are missed.

The **Price** category also has relatively lower recall at **0.88**, despite its high precision of **0.94**.

---

## Training Behaviour

The model was trained for **25 epochs**.

The study reports:

- Training accuracy increasing from approximately **0.45** to **0.97**.
- Validation accuracy increasing from slightly above **0.55** to approximately **0.91**.
- Training loss decreasing from approximately **1.6** to below **0.1**.
- Validation loss reaching a minimum of approximately **0.35** around epoch 10 and later stabilizing around **0.4**.

The narrowing gap between training and validation accuracy indicates good generalization during training.

However, the later increase in validation loss suggests possible early signs of overfitting.

---

## Confusion Matrix Analysis

The reported confusion matrix covers five aspect categories:

- Ambience
- Anecdotes/Miscellaneous
- Food
- Price
- Service

Notable correct classifications include:

| Aspect | Correct Predictions |
|---|---:|
| Anecdotes/Miscellaneous | 382 |
| Food | 320 |
| Service | 168 |
| Ambience | 138 |
| Price | 100 |

The document reports some notable confusion patterns:

- **28 Food** instances were classified as Anecdotes/Miscellaneous.
- **12 Ambience** instances were classified as Anecdotes/Miscellaneous.
- **9 Ambience** instances were classified as Service.
- Price errors were primarily associated with Food and Service.

These errors are attributed largely to contextual and semantic ambiguity in restaurant reviews.

---

## Advantages

The proposed approach provides several potential advantages:

### 1. Data Augmentation

GAN-generated representations can enrich datasets where labelled examples are limited.

### 2. Robustness

Adversarial training exposes the model to challenging synthetic representations, potentially improving robustness to noisy or ambiguous inputs.

### 3. Fine-Grained Analysis

The framework focuses on sentiment associated with individual aspects rather than assigning only one sentiment to an entire review.

### 4. Low-Resource Learning

The approach is designed to be useful when large labelled datasets are unavailable.

### 5. Multi-Aspect Classification

The model can distinguish among multiple restaurant-review aspects such as food, service, ambience, and price.

---

## Limitations and Observations

The study identifies several areas where the system can be improved.

### Potential Overfitting

Validation loss begins increasing after approximately epoch 10 while training loss continues to decrease. This suggests possible overfitting in later training stages.

### Ambience Recall

The Ambience category has a relatively lower recall of **0.84**, indicating that some relevant instances are missed.

### Price Recall

The Price category has a recall of **0.88**, leaving room for improved identification of relevant samples.

### Synthetic Sample Quality

The practical usefulness of GAN-generated representations depends on their realism, diversity, coherence, and fidelity.

### Dataset Scope

The reported experiments focus on restaurant reviews, so performance across other domains is not established by this study.

---

## Future Scope

The document proposes several directions for future development.

### BERT / RoBERTa Integration

Future versions can further combine GANs with contextual transformer encoders such as:

- BERT
- RoBERTa

This could provide richer semantic representations.

### Conditional GANs

A **Conditional GAN (cGAN)** could generate samples conditioned on:

- Aspect labels
- Sentiment labels

This would provide more controlled synthetic data generation.

### Regularization

Potential techniques include:

- Early stopping
- Adversarial dropout
- Label smoothing

These methods may help reduce overfitting.

### Multi-Domain ABSA

The model could be evaluated on domains such as:

- Product reviews
- Hotel reviews
- Other customer-feedback datasets

### Multilingual ABSA

Extending the framework to multilingual datasets could test its cross-lingual robustness.

### Explainable AI

XAI techniques could be incorporated to understand how the generator and discriminator learn aspect-sentiment relationships.

### Human Evaluation

Generated samples could be evaluated for:

- Coherence
- Diversity
- Fidelity
- Linguistic quality

### Real-Time Learning

An online-learning extension could allow the model to adapt to changing sentiment trends on dynamic platforms such as social media and customer-feedback systems.

---

## Applications

A GAN-based ABSA system can potentially support applications such as:

- Restaurant review analytics
- Customer feedback analysis
- Food delivery review mining
- E-commerce review analysis
- Product feedback monitoring
- Service quality analysis
- Opinion mining
- Domain-specific sentiment dashboards

For restaurant reviews specifically, the system can help separate opinions about:

```text
Food
Service
Ambience
Price
Anecdotes / Miscellaneous
```

---

## Project Structure

A recommended GitHub structure for this project is:

```text
.
├── README.md
├── augmented_data_restaurant.csv
├── notebooks/
│   └── ABSA_GAN.ipynb
├── src/
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── discriminator.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_curves.png
└── requirements.txt
```

> **Note:** The source document does not provide this exact repository structure. The structure above is a suggested organization for publishing the project on GitHub.

---

## Reproducibility Notes

The source study reports the following configuration:

```text
Dataset:
    augmented_data_restaurant.csv

Domain:
    Restaurant reviews / Yelp

Aspects:
    Ambience
    Anecdotes/Miscellaneous
    Food
    Price
    Service

Tokenizer:
    WordPiece

Embedding strategy:
    CLS token

Optimizer:
    Adam

Generator dense layers:
    3

Discriminator dense layers:
    3

Batch size:
    32 for training
    16 for embedding

Latent dimension:
    128

Epochs:
    25
```

The paper does **not** provide all implementation details required for exact reproduction, such as a complete source-code listing, dependency versions, random seeds, exact preprocessing code, or a complete hyperparameter configuration file. These should therefore be added separately if available in the project repository.

---

## Expected Output

For a restaurant review, an ABSA system should conceptually produce aspect-specific sentiment information such as:

```text
Review:
"The food was excellent but the service was slow."

Predicted aspects:
    Food     → Positive
    Service  → Negative
```

The exact inference format depends on the implementation used in the repository.

---

## References

The source document cites the following works:

1. J. Li, S. Xu, and D. Yu, “ABSA-GAN: A GAN-based approach for aspect-level sentiment classification,” EMNLP, 2023.
2. D. Croce, G. Castellucci, D. Basili, and R. Troncy, “GAN-BERT: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples,” ACL, 2020, pp. 2114–2124.
3. R. Jain, S. Verma, and K. Gupta, “Aspect-Aware BERT-GAN for Multi-Dimensional Sentiment Analysis,” Journal of Intelligent & Fuzzy Systems, vol. 45, no. 2, 2023, pp. 987–996.
4. K. Lohith, S. Kumar, and R. Naik, “ABSA for Restaurant Reviews Using LDA-BERT-GAN Hybrid Model,” IEEE Access, vol. 11, 2023, pp. 34890–34901.
5. T. Hellwig, F. Meier, and J. Seifert, “Multilingual GAN-BERT for Restaurant Review Analysis,” Transactions on NLP & AI, vol. 3, no. 1, 2024.
6. Y. Zhou, Q. Li, and J. Han, “Aspect-Aware GAN-BERT with Contrastive Learning for Fine-Grained Sentiment Analysis,” COLING 2022, pp. 2785–2794.
7. V. Kumar and N. Rao, “Domain-Specific Fine-Tuning of GAN-BERT for Indian Restaurant Reviews,” ICAITPR, 2023.
8. H. Nguyen, T. Tran, and D. Bui, “NoisyGAN-BERT: Enhancing Robustness of ABSA under Noisy Annotations,” Applied Soft Computing, vol. 122, 2022, Art. no. 108226.
9. R. Patel and S. Joshi, “Cross-Platform GAN-BERT for Aggregated Sentiment Analysis on Food Delivery Apps,” Data Science and Management Journal, vol. 7, 2024.
10. Y. Zhang and K. Lee, “Hierarchical GAN-BERT for Multi-Aspect Review Classification,” Knowledge-Based Systems, vol. 289, 2024, Art. no. 110248.
11. I. Goodfellow et al., “Generative Adversarial Nets,” Advances in Neural Information Processing Systems (NeurIPS), vol. 27, 2014, pp. 2672–2680.
12. M. S. Saji, M. S. Shibily, and A. A. Shah, “Aspect based sentiment analysis using BERT: a survey,” Materials Today: Proceedings, vol. 85, pp. 1848–1853, 2024.
13. Y. Zhang and Q. Liu, “A survey on aspect-based sentiment analysis using pre-trained language models,” IEEE Transactions on Affective Computing, 2024.
14. K. Zhang et al., “A transformer-based approach for aspect-based sentiment analysis,” Scientific Reports, vol. 14, no. 1, Mar. 2024.
15. A. ElJundi, M. Tannir, and H. Hajj, “Aspect-based sentiment analysis using BERT for Arabic language,” ICNLSP, 2021.
16. A. Wu, L. Zhao, and Z. Zhang, “Conditional BERT contextual augmentation for aspect-based sentiment analysis,” arXiv preprint arXiv:2001.11316, 2020.
17. Z. Yang et al., “Boosting Aspect-Based Sentiment Analysis with Contextual Denoising and Semantic Augmentation,” Findings of NAACL, 2024.
18. H. Singh and M. Sharma, “Multimodal sentiment analysis for restaurant reviews using deep learning,” Neural Computing and Applications, 2025.
19. A. Amalia and E. Winarko, “Sentiment Analysis of Restaurant Reviews Using BERT-CNN on Indonesian Language,” Procedia Computer Science, vol. 179, 2021, pp. 865–872.
20. M. George and B. Srividhya, “An Ensemble BERT-Lexicon Framework for Aspect-Level Sentiment Analysis in the Hospitality Sector,” International Journal of Advanced Computer Science and Applications, vol. 12, no. 6, 2021.

---

## Acknowledgements

This README is based on the project document **“Aspect Based Sentiment Analysis Using GAN Model”** and summarizes the methodology, dataset, configuration, evaluation, results, limitations, and future scope described in that document.

---

## Project Status

**Research / Academic Project**

The reported results demonstrate promising performance for GAN-based ABSA on restaurant reviews, with a reported accuracy of approximately **91%** and strong F1 performance across the five evaluated aspect categories.

