import plac
import json
from NER_testing import NERTest
from dateutil import parser

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
    test_data = ner_corpus[round(len(ner_corpus) * 0.9):]
    print(f'\nUsing {data_path} as path to the data.')
    print(f'There are {len(test_data)} docs in test data.')
    print(f'Using {spacy_model_path} as path to the spacy model.')
    print(f'Using {crf_model_path} as path to the CRF model.')
    if lang in ('uk', 'ru'):
        sem_test = NERTest(method_name='semantrum', lang=lang)
        print('')
        sem_test.print_comparisons(test_data)
        sem_df = sem_test.get_full_stats(test_data)
        sem_df.to_csv(f'test_semantrum_{lang}.csv', index=False)
        print(f'Saved Semantrum full test results to test_semantrum_{lang}.csv.')
    spacy_test = NERTest(method_name='spacy', lang=lang, model_path=spacy_model_path)
    crf_test = NERTest(method_name='crf', lang=lang, model_path=crf_model_path)
    print('')
    spacy_test.print_comparisons(test_data)
    print('')
    crf_test.print_comparisons(test_data)
    spacy_df = spacy_test.get_full_stats(test_data)
    spacy_df.to_csv(f'test_{spacy_model_path[:-1]}.csv', index=False)
    print(f'Saved Spacy full test results to test_{spacy_model_path[:-1]}.csv.')
    crf_df = crf_test.get_full_stats(test_data)
    crf_df.to_csv(f'test_{crf_model_path}.csv', index=False)
    print(f'Saved CRF full test results to test_{crf_model_path}.csv.')


if __name__ == '__main__':
    plac.call(main)