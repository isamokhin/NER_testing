import spacy
#from spacy.gold import biluo_tags_from_offsets
import string
import random
import json
from dateutil import parser
import sklearn_crfsuite

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
        'word_pref3': word[:2],
        'word_pref4': word[:4],
        'word_suf3': word[-3:],
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
    #sentence = [e.split(' ') for e in sentence.split('\n')]
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

