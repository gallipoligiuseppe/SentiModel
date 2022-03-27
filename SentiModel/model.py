import logging

import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense


class Attention(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(Attention, self).__init__(name=name)
        super(Attention, self).__init__(**kwargs)
  

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)


    def call(self, x, return_attention=False):
        u = K.tanh(K.dot(x, self.W) + self.b)
        global_contex = tf.reshape(u[:, 0], [-1, 1, 1])
        u_weighted = u * global_contex
        a = K.softmax(u_weighted, axis=1)
        output = x * a
        output = output[:, -1, :]
        if return_attention:
            a = tf.reshape(a, [-1, a.shape[1]])
            return output, a
        else:
            return output, None


class SentiModel(tf.keras.Model):
    def __init__(self, encoder, max_seq_len, n_hidden_layers, bidirectional, n_units, dropout_rate, aspect_sentim_set, sentiment_lexicon, pretrained_path):
        super().__init__()
        encoder_models = {'bert':('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'),
                          'roberta':('https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1', 'https://tfhub.dev/jeongukjae/roberta_en_cased_L-12_H-768_A-12/1')}

        self.max_seq_len = max_seq_len
        self.n_hidden_layers = n_hidden_layers
        self.aspect_sentim_set = aspect_sentim_set
        self.aspect_sentim_attention = {}, {}
        self.sentiment_lexicon = sentiment_lexicon

        self.preprocessor = tf_hub.load(encoder_models[encoder][0])
        self.tokenizer = tf_hub.KerasLayer(self.preprocessor.tokenize, name='tokenizer')
        self.bert_pack_inputs = tf_hub.KerasLayer(self.preprocessor.bert_pack_inputs, arguments=dict(seq_length=max_seq_len), name='tokenizer_pack_inputs')
        self.encoder = tf_hub.KerasLayer(encoder_models[encoder][1], trainable=False, name=encoder)

        self.hidden_layers = []
        self.dropouts = []
        for i in range(n_hidden_layers):
            if bidirectional:
                self.hidden_layers.append(Bidirectional(LSTM(units=n_units, return_sequences=True, recurrent_dropout=0), name=f'bilstm_{i+1}'))
            else:
                self.hidden_layers.append(LSTM(units=n_units, return_sequences=True, recurrent_dropout=0, name=f'lstm_{i+1}'))
            self.dropouts.append(Dropout(dropout_rate, name=f'dropout_{i+1}'))
        self.attention = Attention(name='attention')

        self.dense = Dense(units=n_units, activation='relu', name='dense')
        self.classifier = Dense(units=2, activation='softmax', name='classifier')

        if pretrained_path is not None:
            dummy_sample = tf.constant(['lorem ipsum'])
            _ = self(dummy_sample)
            self.load_weights(pretrained_path)
            logging.info('Model loaded correctly')
    

    def return_attention_weights(self, mode='avg'):
        if mode == 'max': f = np.max
        elif mode == 'sum': f = np.sum
        elif mode == 'avg': f = np.mean
        aspect_att_weights, sentim_att_weights = {}, {}
        for aspect, weights in self.aspect_sentim_attention[0].items():
            aspect_att_weights[aspect] = f(weights)
        for sentim, weights in self.aspect_sentim_attention[1].items():
            sentim_att_weights[sentim] = f(weights)
        return aspect_att_weights, sentim_att_weights
    

    def __update_attention_weights(self, inputs, att_weights):
        for sentence, att_sentence in zip(inputs.numpy().astype('str'), att_weights.numpy()):
            s_tokenized = sentence.split(' ')[:self.max_seq_len - 2]
            att_words = att_sentence[1:len(s_tokenized)+1]
            for w, att_weight in zip(s_tokenized, att_words):
                if w in self.aspect_sentim_set[0]:
                    if w not in self.aspect_sentim_attention[0]:
                        self.aspect_sentim_attention[0][w] = []
                    self.aspect_sentim_attention[0][w].append(att_weight)
                elif w in self.aspect_sentim_set[1]:
                    if w not in self.aspect_sentim_attention[1]:
                        self.aspect_sentim_attention[1][w] = []
                    self.aspect_sentim_attention[1][w].append(att_weight)

    
    def __weight_embeddings(self, inputs, embeddings):
        weights = []
        for sentence in inputs.numpy().astype('str'):
            s_tokenized = sentence.split(' ')[:self.max_seq_len - 2]
            s_weights = [1] + [self.sentiment_lexicon[w] for w in s_tokenized] + [1] * (self.max_seq_len - len(s_tokenized) - 1)
            weights.append(s_weights)
        weights = np.stack(weights, axis=0)
        weights = weights[:, :, np.newaxis]
        weighted_embeddings = embeddings * weights
        return weighted_embeddings
    

    def call(self, inputs, training=False, return_attention=False):
        tokenized_inputs = [self.tokenizer(inputs)]
        encoder_inputs = self.bert_pack_inputs(tokenized_inputs)
        outputs = self.encoder(encoder_inputs)
        embeddings = outputs['sequence_output']
        weighted_embeddings = self.__weight_embeddings(inputs, embeddings)
        
        x = weighted_embeddings
        for i in range(self.n_hidden_layers):
            x = self.hidden_layers[i](x)
            if training:
                x = self.dropouts[i](x, training=training)
        
        x, att_weights = self.attention(x, return_attention=training or return_attention)
        if training:
            self.__update_attention_weights(inputs, att_weights)
        x = self.dense(x)
        x = self.classifier(x)
        if return_attention:
            return x, att_weights
        else:
            return x