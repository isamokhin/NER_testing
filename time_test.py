import plac
import json
import logging
from dateutil import parser
import time
import spacy
import joblib
from crf import collect_the_features
from spacy.gold import offsets_from_biluo_tags

logger = logging.getLogger('time_test')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('time_test.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def crf_ner(text, nlp, crf):
    crf_ents = []
    cursor = 0
    for p in text.split('\n\n'):
        spacy_doc = nlp(p)
        tokens = [t.text for t in spacy_doc]
        features = collect_the_features(tokens)
        tags = crf.predict([features])[0]
        for i, t in enumerate(tags):
            if t.startswith('I'):
                tags[i] = tags[i].replace('I-', 'B-')
        crf_spans = offsets_from_biluo_tags(spacy_doc, tags)
        crf_ents.extend([(start_char + cursor,
                          end_char + cursor,
                          label)
                           for (start_char, end_char, label) in crf_spans
                           ])
        cursor += (len(p) + 2)
    return crf_ents

@plac.annotations(
    data_path=("Path to test data", "option", "d", str),
    spacy_model_path=("Path to spacy model", "option", "m", str),
    crf_model_path=("Path to CRF model", "option", "c", str),
)
def main(data_path, spacy_model_path, crf_model_path):
    if not data_path:
        print('Please provide path to test data')
        return
    lang = spacy_model_path.split('_')[2][:2]
    with open(data_path) as f:
        ner_corpus = json.load(f)
    if 'enriched' in data_path:
        ner_corpus = sorted(ner_corpus,
            key=lambda doc: parser.parse(doc['created']))
    n_iter = 10
    nlp = spacy.load(spacy_model_path)
    loop_total = 0
    for i in range(n_iter):
        loop_start = time.time()
        for entry in ner_corpus:
            text = entry['text']
            doc = nlp(text)
        loop_total += time.time() - loop_start
    spacy_avg = loop_total / n_iter
    logger.info(f'Spacy model {spacy_model_path} predicts 1000 docs in {spacy_avg} seconds')
    crf = joblib.load(crf_model_path)
    nlp = spacy.blank('uk')
    loop_total = 0
    for i in range(n_iter):
        loop_start = time.time()
        for entry in ner_corpus:
            text = entry['text']
            crf_ents = crf_ner(text, nlp, crf)
        loop_total += time.time() - loop_start
    crf_avg = loop_total / n_iter
    logger.info(f'CRF model {crf_model_path} predicts 1000 docs in {crf_avg} seconds')


if __name__ == '__main__':
    plac.call(main)