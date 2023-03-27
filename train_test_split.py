from util import change_extension
from collections import defaultdict
from urllib.parse import urlparse, urlunparse
from itertools import chain
import numpy as np
import re
import json
import sys
import glob
import os

def normalize_link_schemes(all_links):
    link_paths = defaultdict(list)
    for l in set(all_links):
        parse = urlparse(l)
        p = parse.netloc + parse.path + parse.fragment + parse.query + parse.params
        if p != '':
            link_paths[p].append((l, parse))
    change_scheme = {}
    for links in link_paths.values():
        if len(links) > 1:
            has_scheme = lambda scheme, ls: filter(lambda l: l[1].scheme == scheme, ls)
            for httpsl, _ in has_scheme('https', links):
               for l, p in links:
                   if p.scheme == 'http':
                       change_scheme[l] = httpsl
    return [change_scheme.get(l) or l for l in all_links]

def normalize_relative_link(link):
    if link.startswith('./') or link.startswith('../'):
        return re.sub(r'^(\.?\./)+', '', link)
    return link

def normalize_links(all_link_data, min_examples):
    all_links = []
    # indices of links back to their linkdatas
    link_linkdata = []
    for i, linkdata in enumerate(all_link_data):
        for k, v in linkdata.items():
            all_links.append(v['link'].lower())
            link_linkdata.append((i, k))
    all_links = normalize_link_schemes(all_links)
    all_links = [urlunparse(urlparse(l)) for l in all_links]
    all_links = [normalize_relative_link(l) for l in all_links if l != '']
    assert len(all_links) == len(link_linkdata)
    counter = defaultdict(int)
    for link in all_links:
        counter[link] += 1
    for (i, k), link in zip(link_linkdata, all_links):
        all_link_data[i][k]['link'] = link if counter[link] >= min_examples else None
    return all_link_data

def train_test_split(paths, test_size=0.33, min_examples=5):
    """Algorithm:

    - For each class, count its total number of examples, calculate
      its ideal count for the training and test sets, (store in
      vectors totals, trainideals, testideals)

    - Init vector of zeros 'traincounts'. While the training set does
      not have at least the ideal count for each class, and the
      training set's max size has not been reached, repeat:

      - Iterate through each document and count the number of examples
        for each class for the document (store in vector counts)

      - Take the absolute difference between (traincounts + counts) and
        trainideals, sum that in v

      - Subtract (traincounts + counts) from totals, and for each value
        x (with index i) less than testideals[i], compute 1/totals[i],
        sum these all together in m. This is intended to penalize more
        heavily data splits which leave fewer examples to the classes
        with fewer examples

      - Let n be the number of non-zero elements in (traincounts + counts)

      - Let score = 1/v - m + n. The highest scoring document for the
        iteration is added to the training set, and counts is added to
        traincounts

    - Add the remaining documents to the test set

    """
    docs_data = []
    linkdatapaths = []
    for p in paths:
        linkdatapath = change_extension(p, '.linkdata.json')
        linkdatapaths.append(linkdatapath)
        with open(linkdatapath, 'r') as f:
            link_data = json.load(f)
            docs_data.append(link_data)
    docs_data = normalize_links(docs_data, min_examples)
    classes = {}
    class_index = 0
    totals = []
    for v in (v for d in docs_data for v in d.values()):
        link = v['link']
        if link is not None:
            if link not in classes:
                classes[link] = class_index
                class_index += 1
                totals.append(1)
            else:
                totals[classes[link]] += 1
    doc_counts = {}
    for doc_ind, d in enumerate(docs_data):
        doc_data = np.zeros(len(classes))
        for v in d.values():
            link = v['link']
            if link is not None:
                doc_data[classes[link]] += 1
        doc_counts[doc_ind] = doc_data
    totals = np.array(totals)
    testideals = np.floor(totals * test_size)
    trainideals = totals - testideals
    traincounts = np.zeros(len(totals))
    traindata = []
    traindata_max = len(paths) * (1-test_size)
    while len(traindata) < traindata_max and np.any(traincounts < trainideals):
        best = (0, float('-inf'))
        if len(doc_counts) <= 0:
            print('Bug in train test split')
            break
        init = True
        for doc_ind, counts in doc_counts.items():
            newcounts = traincounts + counts
            if init and not np.any(newcounts > trainideals):
                best = (doc_ind, float('-inf'))
                break
            init = False
            v = np.sum(np.absolute(trainideals - newcounts))
            m = 0
            for i, x in enumerate(totals - newcounts):
                if x < testideals[i]:
                    m += 1 / totals[i]
            n = np.count_nonzero(newcounts)
            score = 1 / v - m + n
            if score > best[1]:
                best = (doc_ind, score)
        doc = best[0]
        counts = doc_counts[doc]
        del doc_counts[doc]
        traincounts = traincounts + counts
        print(traincounts)
        print(trainideals)
        print(len(doc_counts))
        traindata.append((linkdatapaths[doc], docs_data[doc]))
    testdata = [(linkdatapaths[doc_ind], docs_data[doc_ind]) for doc_ind in doc_counts.keys()]
    return traindata, testdata

def save_paths(paths, name, min_examples, test_size):
    for path, linkdata in paths:
        path = os.path.join('split', f'{name}-{min_examples}-{test_size*100:.0f}', path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(linkdata, f)

if __name__ == '__main__':
    data_dir = sys.argv[1]
    min_examples = int(sys.argv[2])
    test_size = float(sys.argv[3])
    pattern = os.path.join(data_dir, '**', '*' + '.txt')
    all_txt = glob.glob(pattern, recursive=True)
    train, test = train_test_split(all_txt, test_size=test_size, min_examples=min_examples)
    save_paths(train, 'train', min_examples, test_size)
    save_paths(test, 'test', min_examples, test_size)
