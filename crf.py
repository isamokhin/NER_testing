import spacy
from spacy.gold import biluo_tags_from_offsets
import string
import random
import json
from dateutil import parser
import sklearn_crfsuite
from sklearn.metrics import classification_report
from ukr_stemmer3 import UkrainianStemmer
import joblib


class NER_corpus():

    def __init__(self, lang, corpus_fname):
        self.lang = lang
        self.nlp = spacy.blank(self.lang)
        self.fname = corpus_fname
        self.corpus = self.get_ner_docs()
        self.train_corpus, self.dev_corpus = self.get_train_dev_corpora(self.corpus)
        self.train_pars = self.get_paragraphs(self.train_corpus)
        self.dev_pars = self.get_paragraphs(self.dev_corpus)
        self.train_tok_pars = self.get_tokenized_pars(self.train_pars)
        self.dev_tok_pars = self.get_tokenized_pars(self.dev_pars)
        self.train_labels = self.get_biluo_labels(self.train_pars)
        self.dev_labels = self.get_biluo_labels(self.dev_pars)

    def get_ner_docs(self):
        with open(self.fname) as f:
            corpus = json.load(f)
        if 'created' in corpus[0]:
            corpus = sorted(corpus,
                            key=lambda d:
                            parser.parse(d['created']))
        return corpus

    def get_train_dev_corpora(self, corpus, dev_share=0.1):
        sep = round(len(corpus) * (1 - dev_share))
        train_corpus, dev_corpus = corpus[:sep], corpus[sep:]
        return train_corpus, dev_corpus

    def make_paragraph_triads(self, text, ents):
        res = []
        pars = text.split('\n\n')
        cursor = 0
        for p in pars:
            p_start, p_end = cursor, cursor + len(p)
            par_ents = [(e['start_index'], e['end_index'],
                         e['type']) for e in ents
                        if (e['start_index'] >= p_start)
                        and (e['start_index'] < p_end)]
            triads = [(si - p_start, ei - p_start, ent_type)
                      for (si, ei, ent_type) in par_ents]
            par_tuple = (p, {'entities': triads})
            res.append(par_tuple)
            cursor += len(p) + 2
        return res

    def get_paragraphs(self, corpus):
        paragraphs = []
        for entry in corpus:
            title = entry['title']
            text = entry['text']
            entities = entry['entities']
            title_ents = [e for e in entities if e['source'] == 'title']
            text_ents = [e for e in entities if e['source'] == 'text']
            title_ner = self.make_paragraph_triads(title, title_ents)
            text_ner = self.make_paragraph_triads(text, text_ents)
            paragraphs.extend(title_ner + text_ner)
        return paragraphs

    def get_tokenized_pars(self, paragraphs):
        tok_paragraphs = []
        for par in paragraphs:
            doc = self.nlp(par[0])
            tok_paragraphs.append([t.text for t in doc])
        return tok_paragraphs

    def get_biluo(self, par):
        text = par[0]
        entities = par[1]['entities']
        doc = self.nlp(text)
        biluo_tags = biluo_tags_from_offsets(doc, entities)
        return biluo_tags

    def get_biluo_labels(self, paragraphs):
        return [self.get_biluo(par) for par in paragraphs]

def get_casing(word):
    casing = 'other'
    if len(word) == 0:
        casing = 'NONE'
        return casing
    num_digits = 0
    for char in word:
        if char.isdigit():
            num_digits += 1
    digit_fraction = num_digits / len(word)

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digit_fraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'all_lower'
    elif word.isupper():  # All upper case
        casing = 'all_upper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initial_upper'
    elif num_digits > 0:
        casing = 'contains_digit'

    return casing

def get_ner_features(word, prev_word, next_word):
    features = {
        'word': word,
        'word_stem': UkrainianStemmer(word).stem_word(),
        'word_pref2': word[:2],
        'word_pref4': word[:4],
        'word_suf4': word[-4:],
        'word_suf2': word[-2:],
        'prev_word': prev_word,
        'next_word': next_word,
        'prev_stem': UkrainianStemmer(prev_word).stem_word(),
        'next_stem': UkrainianStemmer(next_word).stem_word(),
        'casing': get_casing(word),
        'is_punct': word in string.punctuation,
        'is_after_punct': prev_word in string.punctuation,
        'is_before_punct': next_word in string.punctuation,
        'after_casing': get_casing(next_word),
        'before_casing': get_casing(prev_word),
    }
    return features

def biluo_to_bio(labels):
    new_labels = []
    tag_lookup = {
        'B': 'B',
        'I': 'I',
        'L': 'I',
        'U': 'B',
        'O': 'O'
    }
    for l in labels:
        if l == 'O' or l == '':
            new_labels.append('O')
            continue
        tag, label = l.split('-')
        new_tag = tag_lookup[tag]
        new_l = new_tag+'-'+label
        new_labels.append(new_l)
    return new_labels

def collect_the_features(tokens):
    features = []
    for i, line in enumerate(tokens):
        if i == 0:
            prev_word = '<S>'
        else:
            prev_word = tokens[i-1]
        if i == len(tokens) - 1:
            next_word = '</S>'
        else:
            next_word = tokens[i+1]
        this_word = tokens[i]
        features.append(get_ner_features(this_word, prev_word, next_word))
    return features

def crf_get_features(paragraphs):
    features = []
    for par in paragraphs:
        f = collect_the_features(par)
        features.append(f)
    return features

def unlabel(label):
    if label == 'O':
        return 'O'
    else:
        return 'N'

def main():
    langs = 'uk ru pl ro de fr it es hu'
    for lang in langs.split(' '):
        print('CRF for language', lang)
        print('')
        try:
            fname = f'ner_finals/ner_{lang}_enriched.json'
            ner = NER_corpus(lang, fname)
        except:
            fname = f'ner_finals/ner_{lang}_final.json'
            ner = NER_corpus(lang, fname)

        train_features = crf_get_features(ner.train_tok_pars)
        dev_features = crf_get_features(ner.dev_tok_pars)
        train_labels = ner.train_labels
        dev_labels = ner.dev_labels

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(train_features, train_labels)

        predicted = crf.predict(dev_features)
        dev_labels_short = [s.split('-')[-1] for t in dev_labels for s in t]
        predicted_short = [s.split('-')[-1] for t in predicted for s in t]
        dev_unlabeled = [unlabel(l) for l in dev_labels_short]
        predicted_unlabeled = [unlabel(l) for l in predicted_short]

        print('-' * 100)
        print(classification_report(dev_unlabeled, predicted_unlabeled))
        print('-' * 100)
        print(classification_report(dev_labels_short, predicted_short))
        print('=' * 100)
        joblib.dump(crf, f'ner_{lang}_crf.joblib')
        print('')

if __name__ == '__main__':
    main()