from util import sliding_window, change_extension
from collections import defaultdict
from urllib.parse import urlparse
from itertools import chain
import multiprocessing
import nltk
import spacy
from spacy.tokens import Token, DocBin
import numpy as np
import en_core_web_sm
import re
import json
import sys
import glob
import os

link_wrap_pattern = re.compile(r"__WRAPSTART(?P<uuid>.+?)__\s(?P<content>.+?)\s__WRAPEND__", flags=re.DOTALL)
link_wrap_split_pattern = re.compile(r"(__WRAPSTART.+?__\s.+?\s__WRAPEND__)", flags=re.DOTALL)

def split_at_links(text):
    split = link_wrap_split_pattern.split(text)
    processed = []
    offset = 0
    for i, s in enumerate([s for s in split if s != '']):
        match = link_wrap_pattern.match(s)
        content = s
        uuid = None
        # Note: we add spaces around the link content to ensure spacy
        # identifies the content as tokens properly
        if match:
            content = ' ' + match['content'] + ' '
            uuid = match['uuid']
        end = offset + len(content)
        # Ensure we account for the extra spaces in the span
        # start/end, otherwise spacy won't like the indices
        processed.append((content, uuid, (offset + 1, end - 1)))
        offset = end
    return processed

def save_spacy_data(path, name):
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    db = DocBin()
    with open(path[1], 'r') as f:
        linkdata = json.load(f)
    with open(path[0], 'r') as f:
        text = f.read()
    split = split_at_links(text)
    joined = ''.join([s for s, _, _ in split])
    doc = nlp(joined)
    spans = []
    for s, uuid, (start, end) in (p for p in split if p[1] is not None):
        content_link = linkdata.get(uuid)
        if content_link is None:
            raise ValueError('UUID could not be found in link data')
        link = content_link['link']
        if link is not None:
            content = doc.char_span(start, end, label=link)
            link_text = content.text
            if len(re.sub('\W', '', link_text)) > 2:
                for t in content:
                    t._.content = content_link
                spans.append(content)
    if spans == []:
        print('Empty spans for:', path[0])
        return
    padded_sents = chain([None], doc.sents, [None])
    i = 0
    scount = 0
    for prev, current, nxt in sliding_window(padded_sents, 3):
        start_sent = prev or current
        start = start_sent.start
        end_sent = nxt or current
        end = end_sent.end
        new_doc = doc[start:end].as_doc()
        new_doc_spans = []
        offset = start_sent.start_char
        while i < len(spans):
            s = spans[i]
            if s.start >= start and s.end <= end:
                new_span = new_doc.char_span(s.start_char - offset, s.end_char - offset, label=s.label)
                new_doc_spans.append(new_span)
                i += 1
            else:
                break
        if len(new_doc_spans) > 0:
            # print('Adding subdocument:', new_doc)
            # print('Spans:', new_doc_spans)
            scount += len(new_doc_spans)
            new_doc.spans['sc'] = new_doc_spans
            db.add(new_doc)
    assert scount == len(spans), f'Number of spans across subdocuments not equal to total'
    out_path = change_extension(path[0], f'.{name}.spacy')
    # print('Writing:', out_path)
    db.to_disk(out_path)

def save_for_paths(x):
    i, (name, n, paths) = x
    # print(f'[{name}] {i+1} / {n}: {paths[0]}')
    save_spacy_data(paths, name)

def linkdata_path_to_txt(path):
    return change_extension(change_extension(path, ''), '.txt')

def get_paths(name, split_suffix, data_dir):
    split_dir = os.path.join('split', f'{name}-{split_suffix}')
    pattern = os.path.join(split_dir, data_dir, '**', '*' + '.linkdata.json')
    linkdata = glob.glob(pattern, recursive=True)
    return list(map(lambda p: (linkdata_path_to_txt(os.path.relpath(p, split_dir)), p), linkdata))

if __name__ == '__main__':
    print('Making spacy data...')
    Token.set_extension('content', default=None)
    data_dir = sys.argv[1]
    split_suffix = sys.argv[2]
    train = get_paths('train', split_suffix, data_dir)
    test = get_paths('test', split_suffix, data_dir)
    with multiprocessing.Pool(processes=8) as pool:
        train_iter = enumerate(('train', len(train), paths) for paths in train)
        test_iter = enumerate(('test', len(test), paths) for paths in test)
        [*pool.imap(save_for_paths, chain(train_iter, test_iter), 100)]
