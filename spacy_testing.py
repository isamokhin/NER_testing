import plac
import json
from NER_testing import NERTest

@plac.annotations(
    data_path=("Path to test data", "option", "d", str),
    spacy_model_path=("Path to spacy model", "option", "m", str),
)
def main(data_path, spacy_model_path):
    lang = spacy_model_path.split('_')[2][:2]
    if not data_path:
        print('Please provide path to test data')
        return
    with open(data_path) as f:
        ner_corpus = json.load(f)
    test_data = ner_corpus[round(len(ner_corpus) * 0.9):]
    spacy_test = NERTest(lang=lang, method_name='spacy', model_path=spacy_model_path)
    print(f'\nUsing {data_path} as path to the data.')
    print(f'There are {len(test_data)} docs in test data.')
    print(f'Using {spacy_model_path} as path to the spacy model.')
    print('')
    spacy_test.print_comparisons(test_data)
    print('')
    spacy_df = spacy_test.get_full_stats(test_data)
    spacy_df.to_csv(f'test_{spacy_model_path[:-1]}.csv', index=False)
    print(f'Saved Spacy full test results to test_{spacy_model_path[:-1]}.csv.')


if __name__ == '__main__':
    plac.call(main)