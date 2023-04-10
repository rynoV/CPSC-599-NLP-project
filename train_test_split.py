import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from imblearn.under_sampling import RandomUnderSampler
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from spacy.tokens import DocBin


def resample(X, y):
    rus = RandomUnderSampler(random_state=0)
    X_i = np.arange(len(X)).reshape((-1, 1))
    X_i, y = rus.fit_resample(X_i, y)
    return X.loc[X_i.reshape((-1, ))], y


def get_x_y(data):
    X = []
    y = []
    for doc in data:
        for s in doc.spans['sc']:
            X.append(doc)
            y.append(s.label_)
    X = pd.Series(X)
    return X, y


def manual_balance(X):
    totals = Counter(s.label_ for doc in X for s in doc.spans['sc'])
    summary = pd.DataFrame([{
        'count': count,
        'link': link
    } for link, count in totals.items()])
    summary = summary.set_index('link')
    min_count = summary.min()['count']
    summary['count_r'] = summary['count']
    Xb = []
    for doc in X:
        spans = doc.spans['sc']
        if len(spans) == 0:
            continue
        counts = Counter(s.label_ for s in spans)
        count_r = summary['count_r'].loc[list(counts.keys())]
        df = pd.DataFrame([{
            'link': k,
            'count_r': c
        } for k, c in counts.items()]).set_index('link')
        count_r -= df['count_r']
        if all(count_r >= min_count):
            summary['count_r'].loc[count_r.index] = count_r
        else:
            Xb.append(doc)
    return Xb


if __name__ == '__main__':
    data_dir = sys.argv[1]
    min_examples = int(sys.argv[2])
    test_size = float(sys.argv[3])
    nlp = spacy.blank('en')
    db = DocBin().from_disk(os.path.join(data_dir, 'data.spacy'))
    docs = pd.Series(list(db.get_docs(nlp.vocab)))
    n = len(docs)
    print(f'Splitting {n} docs')
    label_count = Counter(s.label for doc in docs for s in doc.spans['sc'])
    print(f'Initial label count: {len(label_count)}')
    label_indices = {}
    i = 0
    for l, count in label_count.items():
        if count >= min_examples:
            label_indices[l] = i
            i += 1
    print(f'Label count after filtering: {len(label_indices)}')
    labels = []
    for doc in docs:
        row = np.zeros(len(label_indices))
        indices = [
            label_indices[s.label] for s in doc.spans['sc']
            if s.label in label_indices
        ]
        spans = doc.spans['sc']
        doc.spans['sc'] = []
        for i in range(len(spans)):
            s = spans[i]
            if s.label in label_indices:
                doc.spans['sc'].append(s)
                indices.append(label_indices[s.label])
        row[indices] = 1
        labels.append(row)
    labels = np.array(labels)

    doc_indices = np.arange(n).reshape((-1, 1))
    docs_train, _, docs_test, _ = iterative_train_test_split(
        doc_indices,
        labels,
        test_size=test_size,
    )
    docs_train = docs.loc[docs_train.reshape((-1, ))]
    docs_test = docs.loc[docs_test.reshape((-1, ))]

    # docs_train, y_train = get_x_y(docs_train)
    # docs_test, y_test = get_x_y(docs_test)
    # docs_train, y_train = resample(docs_train, y_train)
    # docs_test, y_test = resample(docs_test, y_test)
    docs_train = manual_balance(docs_train)
    docs_test = manual_balance(docs_test)

    train_db = DocBin(docs=docs_train)
    print(f'{len(train_db)} training documents')
    train_db.to_disk(
        os.path.join(data_dir, 'split',
                     f'train-{min_examples}-{test_size*100:.0f}.spacy'))
    test_db = DocBin(docs=docs_test)
    print(f'{len(test_db)} testing documents')
    test_db.to_disk(
        os.path.join(data_dir, 'split',
                     f'test-{min_examples}-{test_size*100:.0f}.spacy'))
