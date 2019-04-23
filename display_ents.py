import gzip
import json
from spacy import displacy
from spacy.gold import biluo_tags_from_offsets, spans_from_biluo_tags
from itertools import chain

def read_corpus(fname, first_n = None):
    corpus = []
    if fname.endswith('.json'):
        with open(fname) as f:
            corpus = json.load(f)
        return corpus
    count = 0
    with gzip.open(fname, "rb") as f:
        for i,line in enumerate(f):
            postdata = json.loads(line.decode('utf-8'))
            test_text = postdata['text']
            if test_text is None:
                continue
            corpus.append(postdata)
            count += 1
            if first_n:
                if count >= first_n:
                    return corpus
    return corpus


options = {'colors':
               {'PERSON': '#0097ff',
                'LOC': '#53b151',
                'ORG': '#ff5d5d',
                'MEDIA': '#ff5df8',
                'PRODUCT': '#ffcc5d',
                'MISC': '#4efaff'}
           }

def semantrum_ner(doc):
    """
    """
    ent_dict = {'LOC': doc['geo_pos'],
                'PERSON': doc['pers_pos'],
                'ORG': doc['brand_pos']}
    title_ents, text_ents = [], []
    for k in ent_dict:
        if not ent_dict[k]:
            continue
        flattened = chain.from_iterable(ent_dict[k])
        for l in flattened:
            ent_tuple = (l[2], l[3], k)
            if l[1] == 'title':
                title_ents.append(ent_tuple)
            else:
                text_ents.append(ent_tuple)
    return title_ents, text_ents

def sem_to_ents(sem_doc, nlp):
    title_ents, text_ents = semantrum_ner(sem_doc)
    source_dict = {'title': title_ents,
                   'text': text_ents}
    docs = []
    for source in ('title', 'text'):
        sem_ents = source_dict[source]
        cursor = 0
        for p in sem_doc[source].split('\n\n'):
            p_len = len(p)
            p_ents = [(s - cursor, e - cursor, etype) for (s, e, etype)
                      in sem_ents
                      if (s > cursor) and (e <= (cursor + p_len))]
            doc = nlp(p)
            biluo_ents = biluo_tags_from_offsets(doc, p_ents)
            doc.ents = spans_from_biluo_tags(doc, biluo_ents)
            docs.append(doc)
            cursor += (p_len + 2)
    return docs

def render_sem_ents(sem_doc, nlp):
    docs = sem_to_ents(sem_doc, nlp)
    for doc in docs:
        if not doc.ents:
            print(doc)
            continue
        displacy.render(doc, style="ent", options=options)


def render_spacy(sem_doc, nlp):
    title = sem_doc['title']
    text = sem_doc['text']
    doc = nlp(title)
    if doc.ents:
        displacy.render(doc, style="ent")
    else:
        print(doc)
    for p in text.split('\n\n'):
        doc = nlp(p)
        if not doc.ents:
            print(doc)
            continue
        displacy.render(doc, style="ent", options=options)