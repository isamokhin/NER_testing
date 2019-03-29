import spacy
import json
from itertools import chain
from dateutil import parser
import pandas as pd
import plac


class NERTest:

    def __init__(self,
                 method_name,
                 model_path=None,
                 exclude=None):

        self.methods_dict = {
            'semantrum': False,
            'spacy': True
        }

        self.method_name = method_name
        self.model_path = model_path
        if not exclude:
            self.exclude = []
        else:
            self.exclude = exclude

        if self.method_name not in self.methods_dict:
            raise Exception('Please provide valid method.')

        if self.methods_dict[self.method_name]:
            if not model_path:
                raise Exception('Please provide model path for', self.method_name)
            else:
                self._load_model(self.method_name, self.model_path)
        self.ner_func = self.get_ner_function(self.method_name)
        self.ent_list = ['PERSON', 'ORG', 'LOC',
                         'MEDIA', 'PRODUCT', 'MISC']
        self.ent_list = [e for e in self.ent_list
                         if e not in self.exclude]

    def print_comparisons(self, corpus):
        stats1 = self.test_on_golden_corpus(corpus, False, True)
        stats2 = self.test_on_golden_corpus(corpus, False, False)
        stats3 = self.test_on_golden_corpus(corpus, True, True)
        stats4 = self.test_on_golden_corpus(corpus, True, False)
        self.print_results(stats1)
        self.print_results(stats2)
        self.print_results(stats3)
        self.print_results(stats4)

    def _load_model(self, method_name, model_path):
        if method_name == 'spacy':
            import spacy
            self.nlp = spacy.load(model_path)
        else:
            pass

    def get_ner_function(self, method_name):
        func = lambda x: x
        if method_name == 'spacy':
            func = self.spacy_ner
        elif method_name == 'semantrum':
            func = self.semantrum_ner
        return func

    def semantrum_ner(self, doc):
        """
        Converts Semantrum entities into common standard.
        """
        ent_dict = {'LOC': doc['geo_pos'],
                    'PERSON': doc['pers_pos'],
                    'ORG': doc['brand_pos']}
        sem_ents = []
        for k in ent_dict:
            if not ent_dict[k]:
                continue
            if k in self.exclude:
                continue
            flattened = chain.from_iterable(ent_dict[k])
            for l in flattened:
                if l[1] == 'title':
                    continue
                ent_tuple = (l[2], l[3], k)
                sem_ents.append(ent_tuple)
        return sem_ents

    def spacy_ner(self, doc):
        """
        Provide correct spans for each spacy doc.
        """
        text = doc['text']
        spacy_ents = []
        cursor = 0
        for p in text.split('\n\n'):
            spacy_doc = self.nlp(p)
            spacy_ents.extend([(ent.start_char + cursor,
                                ent.end_char + cursor,
                                ent.label_)
                               for ent in spacy_doc.ents
                               if ent.label_ not in self.exclude
                               ])
            cursor += (len(p) + 2)
        return spacy_ents

    def overlap(self, span1, span2, unlabeled=False):
        if unlabeled:
            res = (span1[0] < span2[1]) and (span1[1] > span2[0])
        else:
            res = ((span1[0] < span2[1])
                   and (span1[1] > span2[0])
                   and (span1[2] == span2[2]))
        return res

    def full_overlap(self, span1, span2, unlabeled=False):
        if unlabeled:
            res = (span1[0] == span2[0]) and (span1[1] == span2[1])
        else:
            res = ((span1[0] == span2[0])
                   and (span1[1] == span2[1])
                   and (span1[2] == span2[2]))
        return res

    def span_intersection(self, spans1, spans2, unlabeled=False, full=False):
        """
        Given two lists of spans, return list of spans
        that are present in both lists.
        """
        spans1 = sorted(list(spans1), key=lambda t: t[0])
        spans2 = sorted(list(spans2), key=lambda t: t[0])
        overlappen = [((999999999, 9999999, 'None'), (999999999, 999999999, 'None'))]
        if full:
            ofunc = self.full_overlap
        else:
            ofunc = self.overlap
        while spans1:
            s1 = spans1.pop(0)
            for s2 in spans2:
                if (ofunc(s1, s2, unlabeled=unlabeled)
                        and not ofunc(s1, overlappen[-1][1], unlabeled=unlabeled)):
                    overlappen.append((s1, s2))
                    break
                else:
                    continue
        return overlappen[1:]

    def get_overlaps(self, doc,
                     unlabeled=False, full=False):
        """
        Compute overlaps with the golden set of spans
        for two NER functions.
        """
        spans = set(self.ner_func(doc))
        gold_spans = set([(ent['start_index'], ent['end_index'], ent['type'])
                          for ent in doc['entities'] if ent['source'] == 'text'])
        overlap = self.span_intersection(spans, gold_spans,
                                         unlabeled=unlabeled, full=full)
        intersection = len(overlap)
        spans_size = len(spans)
        gold_size = len(gold_spans)
        return (intersection, spans_size,
                gold_size)

    def get_overlap_stats(self, doc, full=False):
        """
        Given two sets of spans, return numbers
        for each type of entity.
        """
        spans = set(self.ner_func(doc))
        gold_spans = set([(ent['start_index'], ent['end_index'], ent['type'])
                          for ent in doc['entities'] if ent['source'] == 'text'])
        type_dict = {}
        for etype in self.ent_list:
            type_spans = {s for s in spans
                          if s[2] == etype}
            type_golds = {s for s in gold_spans
                          if s[2] == etype}
            type_overlap = self.span_intersection(type_spans, type_golds,
                                                  unlabeled=False, full=full)
            type_dict[etype] = (len(type_overlap),
                                len(type_spans), len(type_golds))
        return type_dict

    def test_by_entity(self, corpus, full=False):
        """
        Compute stats for each entity type
        for the whole corpus.
        """
        type_stats = {t: dict() for t in self.ent_list}
        for etype in self.ent_list:
            type_stats[etype]['type_total_overlap'] = 0
            type_stats[etype]['type_total'] = 0
            type_stats[etype]['type_gold_total'] = 0
        for doc in corpus:
            type_dict = self.get_overlap_stats(doc, full=full)
            for etype in self.ent_list:
                (t_intersection, t_spans_size,
                 t_gold_size) = type_dict[etype]
                type_stats[etype]['type_total_overlap'] += t_intersection
                type_stats[etype]['type_total'] += t_spans_size
                type_stats[etype]['type_gold_total'] += t_gold_size
        for etype in self.ent_list:
            if type_stats[etype]['type_total'] == 0:
                type_stats[etype]['type_total'] = 1
            if type_stats[etype]['type_gold_total'] == 0:
                type_stats[etype]['type_gold_total'] = 1
            type_stats[etype]['precision'] = (
                    type_stats[etype]['type_total_overlap']
                    /
                    type_stats[etype]['type_total'])
            type_stats[etype]['recall'] = (
                    type_stats[etype]['type_total_overlap']
                    /
                    type_stats[etype]['type_gold_total'])
            if (type_stats[etype]['recall'] == 0 or
                    type_stats[etype]['precision'] == 0):
                type_stats[etype]['f1'] = 0
            else:
                type_stats[etype]['f1'] = self.f1(
                    type_stats[etype]['precision'],
                    type_stats[etype]['recall']
                )
            type_stats[etype]['full'] = full
        return type_stats

    def f1(self, precision, recall):
        return 2 * ((precision * recall) / (precision + recall))

    def test_on_golden_corpus(self, corpus,
                              unlabeled=False, full=False):
        """
        Compute overlaps with golden sets for the whole corpus.
        """
        total_overlap = 0
        total = 0
        total_gold_ents = 0
        for doc in corpus:
            (intersection, spans_size,
             gold_size) = (
                self.get_overlaps(doc, unlabeled=unlabeled, full=full))
            total_overlap += intersection
            total += spans_size
            total_gold_ents += gold_size
        if total == 0:
            total = 1
        if total_gold_ents == 0:
            total_gold_ents = 1
        stat_dict = {
            'precision': total_overlap / total,
            'recall': total_overlap / total_gold_ents,
            'unlabeled': unlabeled,
            'full': full
        }
        stat_dict.update({
            'method_f1': self.f1(stat_dict['precision'], stat_dict['recall']),
        })
        return stat_dict

    def print_results(self, stat_dict):
        method = self.method_name.title()
        if stat_dict['unlabeled']:
            lab_str = 'unlabeled'
        else:
            lab_str = 'labeled'
        if stat_dict['full']:
            full_str = 'full'
        else:
            full_str = 'not full'
        precision = stat_dict['precision']
        recall = stat_dict['recall']
        method_f1 = stat_dict['method_f1']

        print(f'{method} NER has precision {round(precision, 2)}, '
              f'recall {round(recall, 2)}, and F1 score {round(method_f1, 2)} ({lab_str}, {full_str} overlaps)')

    def get_full_stats(self, corpus):
        """
        Get everything we can get.
        """
        stats1 = self.test_on_golden_corpus(corpus, False, True)
        stats2 = self.test_on_golden_corpus(corpus, False, False)
        stats3 = self.test_on_golden_corpus(corpus, True, True)
        stats4 = self.test_on_golden_corpus(corpus, True, False)
        type_stats1 = self.test_by_entity(corpus, False)
        type_stats2 = self.test_by_entity(corpus, True)
        full_res = []
        for stat in [stats1, stats2, stats3, stats4]:
            res_list = [self.method_name, self.model_path, self.exclude,
                        stat['unlabeled'], stat['full'], 'all_types',
                        round(stat['precision'], 3), round(stat['recall'], 3),
                        round(stat['method_f1'], 3)]
            full_res.append(res_list)
        for stat in [type_stats1, type_stats2]:
            for ent in stat:
                ent_dict = stat[ent]
                res_list = [self.method_name, self.model_path, self.exclude,
                            False, ent_dict['full'], ent,
                            round(ent_dict['precision'], 3), round(ent_dict['recall'], 3),
                            round(ent_dict['f1'], 3)]
                full_res.append(res_list)
        col_names = ['method_name', 'model_path', 'entities_excluded',
                     'is_unlabeled', 'is_full_overlap', 'entity_type',
                     'precision', 'recall', 'f1']
        res_df = pd.DataFrame(full_res, columns=col_names)
        return res_df


@plac.annotations(
    data_path=("Path to test data", "option", "d", str),
    spacy_model_path=("Path to spacy model", "option", "m", str),
)
def main(data_path, spacy_model_path):
    if not data_path:
        print('Please provide path to test data')
        return
    with open(data_path) as f:
        ner_enriched = json.load(f)
    ner_enriched = sorted(ner_enriched,
                          key=lambda doc: parser.parse(doc['created']))
    test_data = ner_enriched[round(len(ner_enriched) * 0.9):]
    sem_test = NERTest('semantrum')
    spacy_test = NERTest('spacy', spacy_model_path)
    print(f'\nUsing {data_path} as path to the data.')
    print(f'There are {len(test_data)} docs in test data.')
    print(f'Using {spacy_model_path} as path to the spacy model.')
    print('')
    sem_test.print_comparisons(test_data)
    print('')
    spacy_test.print_comparisons(test_data)
    print('')
    sem_df = sem_test.get_full_stats(test_data)
    if '_uk' in data_path:
        lang = 'uk'
    elif '_ru' in data_path:
        lang = 'ru'
    else:
        lang = 'xx'
    sem_df.to_csv(f'semantrum_test_{lang}.csv', index=False)
    print(f'Saved Semantrum full test results to semantrum_test_{lang}.csv.')
    spacy_df = spacy_test.get_full_stats(test_data)
    spacy_df.to_csv(f'spacy_test_{lang}.csv', index=False)
    print(f'Saved Spacy full test results to spacy_test_{lang}.csv.')


if __name__ == '__main__':
    plac.call(main)
