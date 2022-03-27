# Explainable Sentence-Level Sentiment Analysis

This repository contains the source code of the final project for "Deep Natural Language Processing" course @ PoliTO – a.y. 2021-2022.

In this project, we start from the work of Li et al. [Explainable Sentence-Level Sentiment Analysis for Amazon Product Reviews](https://arxiv.org/abs/2111.06070) and we expand it by implementing two extensions:
- we test the model on an additional dataset
- we adapt the method to handle multilingual data

A detailed explanation of the developed solution, including both the description of the method and the experimental results, can be found in the [report](documents/report.pdf) file.

Slides can be found in the [presentation](documents/presentation.pdf) file.

## Task
Our task is to perform sentence-level sentiment analysis, so the model will have to determine if the review provided as input expresses a positive or negative sentiment.
In addition to this, we analyze the model's interpretability at two levels of granularity, document-level and corpus-level.

## Model overview
### Pre-processing
After loading the dataset, reviews text is cleaned by deleting analphabetic sign, lowercasing all words, applying lemmatization and stopwords removal.\
In view of the interpretability study, we define two sets of words: aspect terms and sentimental words which contain the top-160 nouns and adjectives/adverbs, respectively, with a higher TF-IDF score.\
Reviews are encoded into word vectors relying on a pre-trained BERT model which provides contextualized word embeddings that are then scaled by the corresponding sentiment score of the word. In particular, a sentiment lexicon is defined by aggregating the polarity weights provided by SentiWordNet.

### Network architecture
The main component of the network is a BiLSTM layer which processes the input word vectors in a bidirectional way, so for each word taking into consideration both its preceding and following token.\
Then BiLSTM output goes through an attention layer which generates for each word in the review a corresponding attention weight representing the relative importance of that word.\
Attention scores are used to scale the final hidden state of the BiLSTM layer that is sent to a dense layer followed by the final softmax layer that performs binary classification.

### Model explainability
Thanks to the introduced attention layer, it is possible to analyze both sentence-level attention weights and the attention weights distribution of the two sets of words defined before. By doing this, in the first case we can identify which words in a review have greater importance according to the network while, in the second case, it allows to understand which are the most relevant aspects in terms of products and features on which customers focus more.

## Model usage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gallipoligiuseppe/SentiModel/blob/main/amazon_musical_reviews.ipynb)\
The [dataset](data/musical_reviews.csv) used is the Amazon Musical Instruments Dataset.\
The model can be trained using the following command:
```
python main.py train --dataset music --encoder bert --n_epochs <n_epochs> --batch_size <batch_size> --dropout_rate <dropout_rate>
```
At the end of the training, the model is evaluated on the test set.\
To perform only evaluation, the following command can be used:
```
python main.py test --dataset music --encoder bert --from_pretrained <pretrained_path>
```
Commands to run this first set of experiments are grouped in [this](amazon_musical_reviews.ipynb) notebook file.

## Extension I – IMDb Movie Reviews Dataset
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gallipoligiuseppe/SentiModel/blob/main/imdb_movie_reviews.ipynb)\
We test the model on another [dataset](data/IMDb_movie_reviews.csv): since reviews were collected from IMDb website, HTML tags need to be filtered out before applying the pre-processing steps described before.\
The training of the model and its evaluation can be performed using the following commands:
```
python main.py train --dataset imdb --encoder bert --n_epochs <n_epochs> --batch_size <batch_size> --dropout_rate <dropout_rate>
```
```
python main.py test --dataset imdb --encoder bert --from_pretrained <pretrained_path>
```
Commands to run this second set of experiments are grouped in [this](imdb_movie_reviews.ipynb) notebook file.

## Extension II – Multilingual Amazon Reviews Corpus
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gallipoligiuseppe/SentiModel/blob/main/amazon_multi_reviews.ipynb)\
To enlarge the range of applicability of the model, we introduce additional pipeline steps that allow to handle reviews written in different languages.\
The [dataset](https://huggingface.co/datasets/amazon_reviews_multi) can be downloaded from the HuggingFace 🤗 hub.
We translate all reviews in a unique target language by leveraging pre-trained Opus-MT models, which are based on the Neural Machine Translation framework Marian. Since word embeddings are generated by a BERT model pre-trained on the English language, we choose English as target language. After reviews translation, the same pre-processing steps described before are applied.\
The training of the model and its evaluation can be performed using the following commands:
```
python main.py train --dataset home_multi --encoder bert --n_epochs <n_epochs> --batch_size <batch_size> --dropout_rate <dropout_rate>
```
```
python main.py test --dataset home_multi --encoder bert --from_pretrained <pretrained_path>
```
Commands to run this third set of experiments are grouped in [this](amazon_multi_reviews.ipynb) notebook file.

## Baselines
The model is evaluated against three baselines:
- SVM and Multinomial Naive Bayes: they implement a traditional machine learning approach using only TF-IDF scores as features
- LSTM: it uses the same architecture of the proposed solution with the only difference that the LSTM layer is not bidirectional

The evaluation of SVM and Naive Bayes baselines can be performed using the following commands:
```
python main.py baseline svm --dataset <dataset>
```
```
python main.py baseline nb --dataset <dataset>
```
The training and the evaluation of LSTM baseline can be performed using the following commands:
```
python main.py baseline lstm train --dataset <dataset> --encoder bert --n_epochs <n_epochs> --batch_size <batch_size> --dropout_rate <dropout_rate>
```
```
python main.py baseline lstm test --dataset <dataset> --encoder bert --from_pretrained <pretrained_path>
```

## Execution details
### Command line arguments
- --dataset music | imdb | home_multi – which dataset to use for the model training/evaluation
- --encoder bert | roberta – which word embeddings model to use
- --n_epochs – number of epochs
- --load_data – to load aspect terms, sentimental words and sentiment lexicon computed in a previous execution
- --from_pretrained – to load the weights of a model trained in a previous execution
- --max_seq_len – maximum sequence length to process
- --n_hidden_layers – number of BiLSTM layers
- --dropout_rate – dropout rate
- --lr – learning rate
- --batch_size – batch size
- --validation_step – how often to perform validation
- --n_samples – number of sentence-level attention samples

### Requirements
```
python 3.8
pandas 1.4.1
numpy 1.22.3
matplotlib 3.5.1
seaborn 0.11.2
Jinja2 3.1.0
beatifulsoup 4.10
nltk 3.7
scikit-learn 1.0.2
tensorflow 2.8.0
tensorflow-hub 0.12
tensorflow-text 2.8.1
datasets 2.0.0
easynmt 2.0.1
```

The required libraries can be easily installed through the [requirements](requirements.txt) file.

## Authors
- Salvatore Stefano Furnari
- Giuseppe Gallipoli
- Marco Tasca
