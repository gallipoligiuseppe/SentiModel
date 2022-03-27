import logging

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import os
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import TfidfVectorizer
from easynmt import EasyNMT
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('sentiwordnet')


class Dataset():
    def __init__(self, dataset, seed):        
        self.dataset = dataset
        global SEED
        SEED = seed
        self.__lemmatizer = WordNetLemmatizer()
        ngram_range = (1, 1)
        min_df, max_df = 5, 1.0
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df,
                                          tokenizer=lambda s: s.split(' '), lowercase=False, preprocessor=lambda s: s)

        stopwords_list = set(stopwords.words('english') + ['could', 'would', 'may', 'might', 'shall', 'ought', "'s", 'ur', 'else', 'ever', 'sometimes'])
        self.to_exclude = set(['above', 'below', 'over', 'under', 'few', 'more', 'no', 'nor', 'not', 't', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
        self.__stopwords_filtered = stopwords_list - self.to_exclude
        
        self.aspect_terms = []
        self.sentim_words = []
        self.sentiment_lexicon = defaultdict(lambda: 1)


    def load(self, aspect_path, sentim_path, lexicon_path):
        self.aspect_terms = pickle.load(open(aspect_path, "rb"))
        self.sentim_words = pickle.load(open(sentim_path, "rb"))

        self.sentiment_lexicon = pickle.load(open(lexicon_path, "rb"))
        self.sentiment_lexicon = defaultdict(lambda: 1, self.sentiment_lexicon)
    

    def save(self, aspect_path, sentim_path, lexicon_path):
        pickle.dump(self.aspect_terms, open(aspect_path, "wb"))
        pickle.dump(self.sentim_words, open(sentim_path, "wb"))
        pickle.dump(dict(self.sentiment_lexicon), open(lexicon_path, "wb"))


    def __setup(self):
        pass


    def __tag_map(self, tag):
        return {'NOUN':wn.NOUN, 'VERB':wn.VERB, 'ADJ':wn.ADJ, 'ADV':wn.ADV}.get(tag, wn.NOUN)
        
    def __lemma_map(self, lemma):
        return {"n't":'not', 'dont':"don't", 'doesnt':"doesn't", 'wont':"won't", 'mustnt':"mustn't", 'cant':"can't", "'ve":'ve', "'ll":'will',
                "'d":'would','mike':'microphone', 'mic':'microphone', 'mics':'microphone'}.get(lemma, lemma)


    def __process_text(self, text, attach_pos=True):
        text = text.lower()
        text_cleaned = re.sub("[^a-zA-Z'’ ]+", "", text)
        text_tokenized = word_tokenize(text_cleaned)
        text_tagged = pos_tag(text_tokenized, tagset='universal')
        text_lemmatized = [(self.__lemmatizer.lemmatize(w[0], self.__tag_map(w[1])), w[1]) for w in text_tagged]
        text_filtered = [(self.__lemma_map(w[0]), w[1]) for w in text_lemmatized if w[0] not in self.__stopwords_filtered and (len(w[0]) > 3 or w[0] in self.to_exclude)]

        if attach_pos: output = ' '.join([f'{w[0]}/{w[1]}' for w in text_filtered])
        else: output = [w[0] for w in text_filtered]
        return output


    def __delete_tagging(self, text):
        tokens = text.split(' ')
        return [t.split('/')[0] for t in tokens]


    def process(self):
        self.__setup()
        self.dataset.info()

        self.dataset['reviewText_tagged'] = self.dataset['reviewText'].apply(self.__process_text)
        self.dataset['reviewText_tokenized'] = self.dataset['reviewText_tagged'].apply(self.__delete_tagging)
        self.dataset['reviewText_merged'] = self.dataset['reviewText_tokenized'].apply(lambda tokens: ' '.join(tokens))

        self.dataset['review_length'] = self.dataset['reviewText_tokenized'].apply(lambda tokens: len(tokens))
        ix_one_token = self.dataset[self.dataset['review_length']==1].index
        self.dataset.drop(labels=ix_one_token, inplace=True)

        avg_seq_len = np.mean(self.dataset['review_length'])
        logging.info(f'Average sequence length = {avg_seq_len:.0f}\n')
        self.dataset.drop(columns='review_length', inplace=True)

        ix_sample = np.random.choice(len(self.dataset))
        review_sample = self.dataset['reviewText'].values[ix_sample]
        logging.info(review_sample)
        logging.info(self.__process_text(review_sample))


    def create_aspect_sentim(self):
        tfidf_reviews = self.vectorizer.fit_transform(self.dataset['reviewText_tagged'])

        words_allfreq = []
        features = self.vectorizer.get_feature_names_out()

        for row in tfidf_reviews:
            row = row.toarray()[0]
            mask = row > 0
            words_allfreq.extend(list(zip(features[mask], row[mask])))
        words_allfreq = sorted(words_allfreq, key=lambda p: p[1], reverse=True)
        
        aspect_sentim_size = 160

        for w, _ in words_allfreq:
            word, tag = w.split('/')
            if tag in ['NOUN'] and len(self.aspect_terms) < aspect_sentim_size and word not in self.aspect_terms:
                self.aspect_terms.append(word)
            elif tag in ['ADJ', 'ADV'] and len(self.sentim_words) < aspect_sentim_size and word not in self.sentim_words:
                self.sentim_words.append(word)
            if len(self.aspect_terms) == aspect_sentim_size and len(self.sentim_words) == aspect_sentim_size: break
        del words_allfreq
    

    def create_lexicon(self):
        for w in self.vectorizer.get_feature_names_out():
            word = w.split('/')[0]
            word_scores = []
            synsets = swn.senti_synsets(word)
            for synset in synsets:
                if synset.synset.name().split('.')[0] == word:
                    # in [0.5, 1.5] not to have 0 in the range
                    word_compound = 0.5 * (synset.pos_score() - synset.neg_score() + 2)
                    word_scores.append(word_compound)
            if len(word_scores) == 0: word_compound = 1
            else: word_compound = np.round(np.mean(word_scores), 3)
            self.sentiment_lexicon[word] = word_compound

    
    def create_splits(self):
        X, y = self.dataset['reviewText_merged'].to_numpy(), self.dataset['label'].to_numpy()
        X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True, random_state=SEED)
        X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, stratify=y_other, shuffle=True, random_state=SEED)
        return X_train, X_eval, X_test, y_train, y_eval, y_test


class AmazonDataset(Dataset):
    def __init__(self, dataset, seed):
        super().__init__(dataset, seed)


    def __setup(self):
        # fill NaN reviewText with summary
        ix_noReviewText = pd.isna(self.dataset['reviewText'])
        self.dataset.loc[ix_noReviewText, 'reviewText'] = self.dataset.loc[ix_noReviewText, 'summary']
        self.dataset.drop(columns='summary', inplace=True)

        # positive if overall>=4
        self.dataset['label'] = self.dataset['overall'].apply(lambda value: int(value>=4))

        class_dist = self.dataset['label'].value_counts()/len(self.dataset)*100
        logging.info(f'Class 0 (negative): {class_dist[0]:.2f}%\nClass 1 (positive): {class_dist[1]:.2f}%')
    

    def process(self):
        self.__setup()
        super().process()


class IMDbDataset(Dataset):
    def __init__(self, dataset, seed):
        super().__init__(dataset, seed)
    

    def __parse_html(self, text):
        html_parser = BeautifulSoup(text, 'html.parser')
        text = html_parser.get_text()
        text = re.sub('\[[^]]*\]', '', text)
        return text


    def __setup(self):
        # reduce dataset 50k -> 25k
        self.dataset, _, _, _ = train_test_split(self.dataset, [None]*len(self.dataset), train_size=0.5, stratify=self.dataset['label'], shuffle=True, random_state=SEED)

        # remove HTML tags
        self.dataset['reviewText'] = self.dataset['reviewText'].apply(self.__parse_html) 

        self.dataset['label'] = self.dataset['label'].apply(lambda label: int(label=='positive'))

        class_dist = self.dataset['label'].value_counts()/len(self.dataset)*100
        logging.info(f'Class 0 (negative): {class_dist[0]:.2f}%\nClass 1 (positive): {class_dist[1]:.2f}%')
    

    def process(self):
        self.__setup()
        super().process()


class AmazonMultiDataset(Dataset):
    def __init__(self, dataset, seed):
        super().__init__(dataset, seed)
        self.mt_model = EasyNMT('opus-mt')


    def __setup(self):
        languages = ['en', 'fr', 'es', 'de']
        TRANSLATED_PATH = f'./executions/home_reviews_'+'-'.join(languages)+'.pickle'
        if os.path.isfile(TRANSLATED_PATH):
            self.dataset = pd.read_pickle(TRANSLATED_PATH)
        else:
            df_tmp = self.dataset
            # reviews of only one category = home products
            filter_mask = (df_tmp['product_category']=='home') & (df_tmp['language'].isin(languages))
            df_tmp = df_tmp.copy()[filter_mask]
            df_tmp.drop(columns=['review_id', 'product_id', 'reviewer_id', 'review_title', 'product_category'], inplace=True)
            # since ratings 1-5 in equal proportions, removed 3 stars to have an equal number of positive (4-5) and negative (1-2)
            # stratified sampling: same number of pos/neg reviews for each language
            ix_3 = df_tmp[df_tmp['stars']==3].index
            df_tmp.drop(labels=ix_3, inplace=True)
            df_tmp['label'] = df_tmp['stars'].apply(lambda value: int(value>=4))
            df_tmp['lang_label'] = df_tmp['language'] + '_' + df_tmp['label'].astype(str)

            n_target = 30000
            n_target_lang_label = int(n_target/(2*len(languages)))
            df_tmp_toconcat = []
            for lang_label in np.unique(df_tmp['lang_label'].values):
                ix_lang_label = df_tmp['lang_label']==lang_label
                df_tmp_toconcat.append(df_tmp[ix_lang_label].sample(n=n_target_lang_label, replace=False, random_state=SEED))
            self.dataset = pd.concat(df_tmp_toconcat)
            self.dataset = self.dataset.sample(frac=1, random_state=SEED)

            for lang in languages:
                ix_lang = self.dataset['language']==lang
                lang_reviews = self.dataset[ix_lang]['review_body'].tolist()
                if lang == 'en':
                    en_translations = lang_reviews
                else:
                    logging.info(f'Translating from {lang} to en...')
                    en_translations = self.mt_model.translate(lang_reviews, source_lang=lang, target_lang='en', show_progress_bar=True)
                self.dataset.loc[ix_lang, 'reviewText'] = en_translations
            self.dataset.to_pickle(TRANSLATED_PATH)

        class_dist = (self.dataset['lang_label'].value_counts()/len(self.dataset)*100).sort_index()
        logging.info('\nFormat: language_class (0 negative - 1 positive)')
        for label, perc in class_dist.items():
            logging.info(f'{label}: {perc:.2f}%')
    

    def process(self):
        self.__setup()
        super().process()