## Інструмент тестування моделей NER.

Поки що перевірено для української та російської мови, оскільки для них є і натренована модель spacy,
і NER від Семантрума.

Для перевірки spacy моделі потрібно вказати шлях до моделі.
Скачати моделі для української і російської можна так:

```
scp -r [username]@10.0.30.1/resource/WORK_HOME/samokhin/ner_train/ner_model_uk_p/ .
```
Дані можна взяти тут:

```
scp -r [username]@10.0.30.1/resource/WORK_HOME/samokhin/ner_data/ner_uk_enriched.json .
```

Приклад використання інструменту:

```
python NER_testing.py -d ner_uk_enriched.json -m ner_model_uk_p
```

Результат для української мови (детальніше зберігається в csv):

```
Semantrum NER has precision 0.34, recall 0.22, and F1 score 0.27 (labeled, full overlaps)
Semantrum NER has precision 0.83, recall 0.54, and F1 score 0.65 (labeled, not full overlaps)
Semantrum NER has precision 0.36, recall 0.23, and F1 score 0.28 (unlabeled, full overlaps)
Semantrum NER has precision 0.94, recall 0.61, and F1 score 0.74 (unlabeled, not full overlaps)

Spacy NER has precision 0.81, recall 0.82, and F1 score 0.82 (labeled, full overlaps)
Spacy NER has precision 0.88, recall 0.88, and F1 score 0.88 (labeled, not full overlaps)
Spacy NER has precision 0.86, recall 0.87, and F1 score 0.87 (unlabeled, full overlaps)
Spacy NER has precision 0.95, recall 0.96, and F1 score 0.95 (unlabeled, not full overlaps)
```