import logging

import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset
from SentiModel.dataset import *
from SentiModel.model import *
from SentiModel.model_utils import *
from SentiModel.utils import *
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

SEED = 31
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def main(args):
    try:
        if args.mode == 'train' or args.mode == 'test':
            bilstm(args)
        elif args.baseline == 'lstm':
            lstm(args)
        elif args.baseline == 'svm' or args.baseline == 'nb':
            baseline(args)
    except (FileNotFoundError, OSError):
        logging.error('File not found - Execution stopped')
        return


def bilstm(args):
    if args.dataset == 'music':
        df_reviews = pd.read_csv('./data/musical_reviews.csv', usecols=['reviewText', 'overall', 'summary'])
        dataset = AmazonDataset(df_reviews, SEED)
    elif args.dataset == 'imdb':
        df_reviews = pd.read_csv('./data/IMDb_movie_reviews.csv', names=['reviewText', 'label'], header=0)
        dataset = IMDbDataset(df_reviews, SEED)
    elif args.dataset == 'home_multi':
        df_reviews = load_dataset('amazon_reviews_multi', split='train').to_pandas()
        dataset = AmazonMultiDataset(df_reviews, SEED)
    else:
        logging.info('Not supported yet')
        return
    logging.info('Dataset preprocessing...')
    dataset.process()
    if args.load_data:
        dataset.load(ASPECT_PATH, SENTIM_PATH, LEXICON_PATH)
    else:
        dataset.create_aspect_sentim()
        dataset.create_lexicon()
        dataset.save(ASPECT_PATH, SENTIM_PATH, LEXICON_PATH)
    model = SentiModel(args.encoder, args.max_seq_len, args.n_hidden_layers, True, 256, args.dropout_rate, (dataset.aspect_terms,
                       dataset.sentim_words), dataset.sentiment_lexicon, MODEL_LOAD_PATH)
    opt = Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss=senti_loss, metrics=[accuracy, precision, recall, f1], run_eagerly=True)
    X_train, X_eval, X_test, y_train, y_eval, y_test = dataset.create_splits()
    if args.mode == 'train':
        callbacks = [TrainingCallback(MODEL_SAVE_PATH, args.validation_step)]
        logging.info(f'Training set size: {len(X_train)}')
        logging.info(f'Evaluation set size: {len(X_eval)}\n')
        logging.info('Start training...')
        history = model.fit(x=X_train, y=y_train, epochs=args.n_epochs, batch_size=args.batch_size, validation_data=(X_eval, y_eval),
                            validation_freq=args.validation_step, shuffle=True, callbacks=callbacks, verbose=1)
        logging.info('End training...')
        fig, axs = plt.subplots(1, 3, figsize=(20, 4))
        plot_metrics(history, 'loss', axs[0], 'Loss per epoch')
        plot_metrics(history, 'accuracy', axs[1], 'Accuracy per epoch')
        plot_metrics(history, 'precision', axs[2], 'Precision per epoch')
        filename = PLOTS_PATH+f'/loss_acc_prec.pdf'
        fig.savefig(filename, format='pdf')
        fig, axs = plt.subplots(1, 2, figsize=(13, 4))
        plot_metrics(history, 'recall', axs[0], 'Recall per epoch')
        plot_metrics(history, 'f1', axs[1], 'F1 score per epoch')
        filename = PLOTS_PATH+f'/rec_f1.pdf'
        fig.savefig(filename, format='pdf')
        aspect_terms_att, sentim_words_att = model.return_attention_weights()
        topK = 10
        fig, ax = plt.subplots(figsize=(10, 5))
        title = f'Aspect terms – top {topK} attention weights'
        plot_attention_weights(aspect_terms_att, topK, 'minmax', ax, title)
        filename = PLOTS_PATH+f'/att_weights_aspects.pdf'
        fig.savefig(filename, format='pdf')
        fig, ax = plt.subplots(figsize=(10, 5))
        title = f'Sentimental words – top {topK} attention weights'
        plot_attention_weights(sentim_words_att, topK, 'minmax', ax, title)
        filename = PLOTS_PATH+f'/att_weights_senti_words.pdf'
        fig.savefig(filename, format='pdf')
    elif args.from_pretrained is None:
        logging.info(f'No pretrained model was given, so initial weights will be used')
    logging.info(f'Test set size: {len(X_test)}')
    y_pred, y_pred_prob = evaluate_model(model, 'SentiModel', X_test, y_test, 'test')
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plot_confusion(y_test, y_pred, axs[0])
    plot_ROC(y_test, y_pred_prob, axs[1])
    filename = PLOTS_PATH+f'/confusion_roc_test.pdf'
    fig.savefig(filename, format='pdf')
    get_samples(model, args.max_seq_len, args.n_samples, X_test, SAMPLES_PATH)
    review_sample = X_test[31]
    input = tf.constant([review_sample])
    _, att_weights = model(input, training=False, return_attention=True)
    sentence_attention(review_sample, args.max_seq_len, att_weights, 'exp')


def lstm(args):
    if args.dataset == 'music':
        df_reviews = pd.read_csv('./data/musical_reviews.csv', usecols=['reviewText', 'overall', 'summary'])
        dataset = AmazonDataset(df_reviews, SEED)
    elif args.dataset == 'imdb':
        df_reviews = pd.read_csv('./data/IMDb_movie_reviews.csv', names=['reviewText', 'label'], header=0)
        dataset = IMDbDataset(df_reviews, SEED)
    elif args.dataset == 'home_multi':
        df_reviews = load_dataset('amazon_reviews_multi', split='train').to_pandas()
        dataset = AmazonMultiDataset(df_reviews, SEED)
    else:
        logging.info('Not supported yet')
        return
    logging.info('Dataset preprocessing...')
    dataset.process()
    if args.load_data:
        dataset.load(ASPECT_PATH, SENTIM_PATH, LEXICON_PATH)
    else:
        dataset.create_aspect_sentim()
        dataset.create_lexicon()
        dataset.save(ASPECT_PATH, SENTIM_PATH, LEXICON_PATH)
    model = SentiModel(args.encoder, args.max_seq_len, args.n_hidden_layers, False, 256, args.dropout_rate, (dataset.aspect_terms,
                            dataset.sentim_words), dataset.sentiment_lexicon, MODEL_LOAD_PATH)
    opt = Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss=senti_loss, metrics=[accuracy, precision, recall, f1], run_eagerly=True)
    X_train, X_eval, X_test, y_train, y_eval, y_test = dataset.create_splits()
    if args.lstm_mode == 'train':
        callbacks = [TrainingCallback(MODEL_SAVE_PATH, args.validation_step)]
        logging.info(f'Training set size: {len(X_train)}')
        logging.info(f'Evaluation set size: {len(X_eval)}\n')
        logging.info('Start training...')
        history = model.fit(x=X_train, y=y_train, epochs=args.n_epochs, batch_size=args.batch_size, validation_data=(X_eval, y_eval),
                            validation_freq=args.validation_step, shuffle=True, callbacks=callbacks, verbose=1)
        logging.info('End training...')
    elif args.from_pretrained is None:
        logging.info(f'No pretrained model was given, so initial weights will be used')
    logging.info(f'Test set size: {len(X_test)}')
    y_pred, y_pred_prob = evaluate_model(model, 'SentiModel_LSTM', X_test, y_test, 'test')


def baseline(args):
    if args.dataset == 'music':
        df_reviews = pd.read_csv('./data/musical_reviews.csv', usecols=['reviewText', 'overall', 'summary'])
        dataset = AmazonDataset(df_reviews, SEED)
    elif args.dataset == 'imdb':
        df_reviews = pd.read_csv('./data/IMDb_movie_reviews.csv', names=['reviewText', 'label'], header=0)
        dataset = IMDbDataset(df_reviews, SEED)
    elif args.dataset == 'home_multi':
        df_reviews = load_dataset('amazon_reviews_multi', split='train').to_pandas()
        dataset = AmazonMultiDataset(df_reviews, SEED)
    else:
        logging.info('Not supported yet')
        return
    logging.info('Dataset preprocessing...')
    dataset.process()
    X_train, X_eval, X_test, y_train, y_eval, y_test = dataset.create_splits()
    X_train_tfidf = dataset.vectorizer.fit_transform(X_train).toarray()
    X_eval_tfidf = dataset.vectorizer.transform(X_eval).toarray()
    X_test_tfidf = dataset.vectorizer.transform(X_test).toarray()
    if args.baseline == 'svm':
        model_SVM = SVC(C=1, kernel='rbf', degree=3, gamma='scale', random_state=SEED)
        logging.info('Start fitting...')
        model_SVM.fit(X_train_tfidf, y_train)
        logging.info('End fitting...')
        logging.info(f'Evaluation set size: {len(X_test)}')
        _ = evaluate_model(model_SVM, 'SVM', X_eval_tfidf, y_eval, 'evaluation')
        logging.info('\n#######################################################\n')
        logging.info(f'Test set size: {len(X_test)}')
        _ = evaluate_model(model_SVM, 'SVM', X_test_tfidf, y_test, 'test')
    elif args.baseline == 'nb':
        model_NB = MultinomialNB(alpha=1)
        logging.info('Start fitting...')
        model_NB.fit(X_train_tfidf, y_train)
        logging.info('End fitting...')
        logging.info(f'Evaluation set size: {len(X_test)}')
        _ = evaluate_model(model_NB, 'Naive Bayes', X_eval_tfidf, y_eval, 'evaluation')
        logging.info('\n#######################################################\n')
        logging.info(f'Test set size: {len(X_test)}')
        _ = evaluate_model(model_NB, 'Naive Bayes', X_test_tfidf, y_test, 'test')


def arg_parser(args=None):
    parser = argparse.ArgumentParser(description='BiLSTM + Attention for Sentiment Analysis')
    subparser = parser.add_subparsers(required=True, dest='mode')

    parser_aux = argparse.ArgumentParser(add_help=False)
    parser_aux_dataset = argparse.ArgumentParser(add_help=False)
    
    parser_aux_dataset.add_argument('--dataset', type=str, choices=['music', 'imdb', 'home_multi'], required=True, help='dataset (Amazon Music Reviews or IMDb Movie Reviews or Amazon Multilingual Home Reviews)')
    parser_aux.add_argument('--encoder', type=str, choices=['bert', 'roberta'], required=True, help='embedding model (BERT or RoBERTa)')
    parser_aux.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    parser_aux.add_argument('--load_data', action='store_true', default=False, help='load precomputed aspect/sentim/lexicon')
    parser_aux.add_argument('--from_pretrained', type=str, default=None, metavar='PRETRAINED PATH', help='path of pretrained model')
    parser_aux.add_argument('--max_seq_len', type=int, default=64, help='maximum sequence length to process')
    parser_aux.add_argument('--n_hidden_layers', type=int, default=1, help='number of hidden layers')
    parser_aux.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser_aux.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser_aux.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser_aux.add_argument('--validation_step', type=int, default=1, help='to perform validation after each validation_step epoch(s)')
    parser_aux.add_argument('--n_samples', type=int, default=5, help='number of sentence samples to save')

    subparser.add_parser('train', help='BiLSTM training', parents=[parser_aux_dataset, parser_aux])
    subparser.add_parser('test', help='BiLSTM test', parents=[parser_aux_dataset, parser_aux])
    parser_baseline = subparser.add_parser('baseline', help='LSTM, SVM, Naive Bayes')
    subparser_baseline = parser_baseline.add_subparsers(required=True, dest='baseline')
    parser_baseline_lstm = subparser_baseline.add_parser('lstm', help='LSTM')
    subparser_lstm = parser_baseline_lstm.add_subparsers(required=True, dest='lstm_mode')
    subparser_lstm.add_parser('train', help='LSTM training', parents=[parser_aux_dataset, parser_aux])
    subparser_lstm.add_parser('test', help='LSTM test', parents=[parser_aux_dataset, parser_aux])
    subparser_baseline.add_parser('svm', help='SVM', parents=[parser_aux_dataset])
    subparser_baseline.add_parser('nb', help='Naive Bayes', parents=[parser_aux_dataset])

    return parser.parse_args(args=args)
    

def init_path(args):
    timestamp = datetime.now().strftime('%d%H%M')
    
    global ROOT_PATH, OUTPUT_PATH, PLOTS_PATH, SAMPLES_PATH, ASPECT_PATH, SENTIM_PATH, LEXICON_PATH, MODEL_LOAD_PATH, MODEL_SAVE_PATH
    
    ROOT_PATH = './executions/'
    OUTPUT_PATH = ROOT_PATH+timestamp+'/'
    PLOTS_PATH = OUTPUT_PATH+'plots/'
    SAMPLES_PATH = OUTPUT_PATH+'/samples/'
    
    ASPECT_PATH = ROOT_PATH+f'aspect_set_{args.dataset}.pickle'
    SENTIM_PATH = ROOT_PATH+f'sentim_set_{args.dataset}.pickle'
    LEXICON_PATH = ROOT_PATH+f'sentim_lexicon_{args.dataset}.pickle'

    if (args.mode == 'train' or args.mode == 'test' or args.baseline == 'lstm') and args.from_pretrained is not None:
        MODEL_LOAD_PATH = ROOT_PATH+args.from_pretrained+'/model.h5'
    else:
        MODEL_LOAD_PATH = None
    MODEL_SAVE_PATH = OUTPUT_PATH+'model.h5'

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(SAMPLES_PATH, exist_ok=True)


if __name__ == '__main__':
    args = arg_parser()
    init_path(args)
    logging.basicConfig(filename=OUTPUT_PATH+'log.txt', filemode='w', format='%(message)s', level=logging.INFO)
    logging.info(f'Run with parameters:\n{args}\n')
    main(args)