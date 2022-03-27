import logging

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def senti_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, -1)
    pos_prob = y_pred[:, 1]
    SCE = SparseCategoricalCrossentropy(from_logits=False, reduction='sum_over_batch_size')
    l1 = SCE(y_true, y_pred)
    l2 = mean_squared_error(y_true, pos_prob, squared=False)
    loss = (l1 + l2)/2
    return loss

def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, -1)
    y_pred_lab = tf.math.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred_lab)
    return accuracy

def precision(y_true, y_pred):
    y_true = tf.reshape(y_true, -1)
    y_pred_lab = tf.math.argmax(y_pred, axis=1)
    precision =  precision_score(y_true, y_pred_lab, average='macro', zero_division=0)
    return precision

def recall(y_true, y_pred):
    y_true = tf.reshape(y_true, -1)
    y_pred_lab = tf.math.argmax(y_pred, axis=1)
    recall =  recall_score(y_true, y_pred_lab, average='macro', zero_division=0)
    return recall

def f1(y_true, y_pred):
    y_true = tf.reshape(y_true, -1)
    y_pred_lab = tf.math.argmax(y_pred, axis=1)
    f1 =  f1_score(y_true, y_pred_lab, average='macro', zero_division=0)
    return f1


def evaluate_model(model, model_name, X_test, y_test, test_type):
    logging.info(f'Start evaluating {model_name} on {test_type} set...')
    if model_name not in ['SVM', 'Naive Bayes']:
        pred_prob = model.predict(x=X_test, verbose=1)
        y_pred = tf.math.argmax(pred_prob, axis=1).numpy()
    else:
        y_pred = model.predict(X_test)
    n_correct = sum(y_test==y_pred)
    logging.info(f'\nPredicted {n_correct} correctly out of {len(y_test)} examples\n')
    logging.info(classification_report(y_test, y_pred, digits=3, zero_division=0))
    logging.info('End evaluation...')
    if model_name not in ['SVM', 'Naive Bayes']:
        return y_pred, pred_prob
    else:
        return y_pred


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_save_path, val_freq):
        super().__init__()
        self.model_save_path = model_save_path
        self.val_freq = val_freq


    def on_train_begin(self, logs=None):
        logging.info("Start training logging...")
        self.model.metrics_log = {}
  

    def on_train_end(self, logs=None):
        logging.info("Saving model...")
        self.model.save_weights(self.model_save_path)
        logging.info("End training logging...")


    def on_epoch_begin(self, epoch, logs=None):
        # to save only last epoch attention weights
        self.model.aspect_sentim_attention = {}, {}
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
        self.model.metrics_log[epoch+1] = {}
        for m in metrics:
            self.model.metrics_log[epoch+1][m] = []
        if (epoch + 1) % self.val_freq == 0:
            for m in metrics:
                self.model.metrics_log[epoch+1][f'val_{m}'] = []


    def on_train_batch_end(self, batch, logs=None):
        for m in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            self.model.metrics_log[list(self.model.metrics_log.keys())[-1]][m].append(logs[m])


    def on_test_batch_end(self, batch, logs=None):
        for m in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            self.model.metrics_log[list(self.model.metrics_log.keys())[-1]][f'val_{m}'].append(logs[m])