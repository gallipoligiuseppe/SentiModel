import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import tensorflow as tf


def plot_metrics(log, metric, ax, title):
    x = log.epoch
    y_values = (log.history[metric], log.history[f'val_{metric}'])
    ax.plot(x, y_values[0], label='training')
    ax.plot(x, y_values[1], label='validation')
    ax.legend(loc='lower right')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    ax.grid(linewidth=0.6)
    ax.set_title(title)


def plot_confusion(y_true, y_pred, ax):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['negative', 'positive'], columns=['negative', 'positive'])
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    cm_plot = sns.heatmap(data=cm_df, cmap='Blues', annot=True, ax=ax)


def plot_ROC(y_true, y_pred_prob, ax):
    y_pred_prob_pos = y_pred_prob[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob_pos)
    auc = roc_auc_score(y_true, y_pred_prob_pos, average='macro')
    ax.plot(fpr, tpr, color='darkorange', label=f'AUC = {auc:.2f}')
    ax.plot([0, 1], ls="--")
    ax.plot([0, 0], [1, 0], c=".6"), plt.plot([1, 1] , c=".6")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(linewidth=0.6)
    ax.set_title('Receiver Operating Characteristic')


def plot_attention_weights(word_att, topK, scaling, ax, title):
    word_att_sorted = sorted(word_att.items(), key=lambda p: p[1], reverse=True)
    words, att_weights = [], []
    for w, att in word_att_sorted:
        words.append(w)
        att_weights.append(att)

    if scaling == 'minmax':
        att_weights_scaled = [(w-np.min(att_weights))/(np.max(att_weights)-np.min(att_weights)+1e-6) for w in att_weights]
    elif scaling == 'exp':
        att_weights_scaled = [np.exp(100*w) for w in att_weights]
        att_weights_scaled = [(w-np.min(att_weights_scaled))/(np.max(att_weights_scaled)-np.min(att_weights_scaled)+1e-6) for w in att_weights_scaled]

    topK_words, topK_att_weights = words[:topK], att_weights_scaled[:topK]
    x = np.arange(topK)
    ax.bar(x, topK_att_weights)
    ax.set_xticks(x)
    ax.set_xticklabels(topK_words)
    ax.set_xlabel('word')
    ax.set_ylabel('attention weight')
    ax.grid(linewidth=0.6)
    ax.set_title(title)


def __background_gradient(df, row):
    cmap = 'vlag'
    if row.name == 'attention weight':
        weights = df.loc[row.name, :].copy().astype(float)
        return [f'background-color: {colors.rgb2hex(x)}' for x in plt.cm.get_cmap(cmap)(weights)]
    else: return ['' for value in row]


def sentence_attention(sentence, max_seq_len, att_weights, scaling):
    s_tokenized = sentence.split(' ')[:max_seq_len-2]
    att_weights = att_weights.numpy()[0][1:len(s_tokenized)+1]
    
    if scaling == 'minmax':
        att_weights_scaled = [(w-np.min(att_weights))/(np.max(att_weights)-np.min(att_weights)+1e-6) for w in att_weights]
    elif scaling == 'exp':
        att_weights_scaled = [np.exp(100*w) for w in att_weights]
        att_weights_scaled = [(w-np.min(att_weights_scaled))/(np.max(att_weights_scaled)-np.min(att_weights_scaled)+1e-6) for w in att_weights_scaled]
        
    df_s_att = pd.DataFrame(data=zip(s_tokenized, att_weights_scaled), columns=['word', 'attention weight'])
    df_s_att.index.name = 'ix'
    df_s_att.index = df_s_att.index + 1
    df_s_att = df_s_att.transpose()
    return df_s_att.style.apply(lambda row: __background_gradient(df_s_att, row), axis=1)


def get_samples(model, max_seq_len, n_samples, dataset, output_path):
    ix_samples = np.random.choice(len(dataset), size=n_samples, replace=False)
    for i, ix in enumerate(ix_samples):
        review_sample = dataset[ix]
        input = tf.constant([review_sample])
        _, att_weights = model(input, training=False, return_attention=True)
        sentence_att = sentence_attention(review_sample, max_seq_len, att_weights, 'exp')
        sentence_att_html = sentence_att.to_html(doctype_html=True)
        sentence_att_html = sentence_att_html.replace('utf-8">', 'utf-8">\n<link rel="stylesheet" href="https://cdn.jupyter.org/notebook/6.4.6/style/style.min.css">')
        sentence_att_html = sentence_att_html.replace('<body>', '<body class="rendered_html">')
        filename = output_path+f'/sample_{i}.html'
        f = open(filename, 'w')
        f.write(sentence_att_html)
        f.close()