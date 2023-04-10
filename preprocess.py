import glob
import json
import multiprocessing
import os
import re
import sys
from collections import defaultdict
from itertools import chain
from urllib.parse import urlparse, urlunparse

import spacy
from spacy.tokens import DocBin, Token

from util import change_extension, sliding_window


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
            has_scheme = lambda scheme, ls: filter(
                lambda l: l[1].scheme == scheme, ls)
            for httpsl, _ in has_scheme('https', links):
                for l, p in links:
                    if p.scheme == 'http':
                        change_scheme[l] = httpsl
    return [change_scheme.get(l) or l for l in all_links]


def normalize_relative_link(link):
    if link.startswith('./') or link.startswith('../'):
        return re.sub(r'^(\.?\./)+', '', link)
    return link


def normalize_links(all_link_data):
    all_links = []
    # indices of links back to their link_datas
    link_link_data = []
    for i, link_data in enumerate(all_link_data):
        for k, v in link_data.items():
            all_links.append(v['link'].lower())
            link_link_data.append((i, k))
    all_links = normalize_link_schemes(all_links)
    all_links = [urlunparse(urlparse(l)) for l in all_links]
    all_links = [normalize_relative_link(l) for l in all_links if l != '']
    assert len(all_links) == len(link_link_data)
    for (i, k), link in zip(link_link_data, all_links):
        all_link_data[i][k]['link'] = link
    return all_link_data


link_wrap_pattern = re.compile(
    r"__WRAPSTART(?P<uuid>.+?)__\s(?P<content>.+?)\s__WRAPEND__",
    flags=re.DOTALL)
link_wrap_split_pattern = re.compile(r"(__WRAPSTART.+?__\s.+?\s__WRAPEND__)",
                                     flags=re.DOTALL)


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


def make_new_span(doc, s, offset):
    return doc.char_span(s.start_char - offset,
                         s.end_char - offset,
                         label=s.label)


def save_spacy_data(args):
    path, linkdata, nlp = args
    # print('Processing', path[0])
    db = DocBin()
    with open(path, 'r') as f:
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
        content = doc.char_span(start, end, label=link)
        link_text = content.text
        if len(re.sub(r'\W', '', link_text)) > 2:
            for t in content:
                t._.content = content_link
            spans.append(content)
    if spans == []:
        print('Empty spans for:', path)
        return db
    padded_sents = chain([None], doc.sents, [None])
    i = 0
    spans_used = set()
    # Try to use sentences as context
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
                spans_used.add(s)
                new_span = make_new_span(new_doc, s, offset)
                new_doc_spans.append(new_span)
                i += 1
            else:
                break
        if len(new_doc_spans) > 0:
            # print('Adding subdocument:', new_doc)
            # print('Spans:', new_doc_spans)
            new_doc.spans['sc'] = new_doc_spans
            db.add(new_doc)
    # For spans crossing the parsed sentence boundaries, just use a given amount
    # of tokens as context. Note: this isn't perfect for preventing false negatives
    spans_unused = list(set(spans).difference(spans_used))
    for i, s in enumerate(spans_unused):
        if s not in spans_used:
            spans_used.add(s)
            context = 10
            start = max(0, s.start - context)
            offset = doc[start].idx
            end = min(len(doc) - 1, s.end + context)
            new_doc = doc[start:end].as_doc()
            new_spans = [make_new_span(new_doc, s, offset)]
            for ss in (s for s in spans_unused if s not in spans_used
                       and s.start >= start and s.end <= end):
                spans_used.add(ss)
                new_spans.append(make_new_span(new_doc, ss, offset))
            new_doc.spans['sc'] = new_spans
            db.add(new_doc)
    assert len(spans_used) == len(
        spans
    ), f'Number of spans across subdocuments not equal to total: {len(spans_used)}, {len(spans)}, {path[0]}'
    # print('Writing:', out_path)
    return db


def linkdata_path_to_txt(path):
    return change_extension(change_extension(path, ''), '.txt')


def remove_duplicates(docs):
    seen = set()
    new_docs = []
    for doc in docs:
        spans = doc.spans['sc']
        spans_set = set(spans)
        spans_used = set()
        no_duplicates = True
        for tokens in sliding_window(doc, 8):
            ts = (' '.join(t.text.strip() for t in tokens)).lower()
            all_spans_used = False
            for span in filter(lambda s: any(t in s for t in tokens), spans):
                if (ts, span.label) in seen:
                    # print('Duplicate found:', ts)
                    no_duplicates = False
                    all_spans_used = spans_used == spans_set
                else:
                    seen.add((ts, span.label))
                    spans_used.add(span)
            if all_spans_used:
                break
        if no_duplicates:
            new_docs.append(doc)
    return new_docs


if __name__ == '__main__':
    print('Making spacy data...')
    Token.set_extension('content', default=None)
    data_dir = sys.argv[1]
    pattern = os.path.join(data_dir, '**', '*' + '.txt')
    all_txt = glob.glob(pattern, recursive=True)
    docs_data = []
    for p in all_txt:
        link_data_path = change_extension(p, '.linkdata.json')
        with open(link_data_path, 'r') as f:
            link_data = json.load(f)
            docs_data.append(link_data)
    docs_data = normalize_links(docs_data)
    db = DocBin()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.initialize()
    with multiprocessing.Pool(processes=None) as pool:
        for d in pool.map(save_spacy_data, ((path, docs_data[i], nlp)
                                            for i, path in enumerate(all_txt)),
                          100):
            db.merge(d)
    print(f'Total of {len(db)} examples before duplicates')
    docs = list(db.get_docs(nlp.vocab))
    docs = remove_duplicates(docs)
    db = DocBin(docs=docs)
    print(f'Total of {len(db)} examples after duplicates')
    db.to_disk(os.path.join('data', 'data.spacy'))