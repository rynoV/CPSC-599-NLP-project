# [[file:experiment.org::+begin_src python][No heading:1]]
import nltk
import spacy
from spacy.tokens import Token
import en_core_web_sm
import re
import json
from spacy.tokens import DocBin

nlp = en_core_web_sm.load()

Token.set_extension('wrapper', default=False, force=True)
Token.set_extension('content', default=None, force=True)
with open('data/testing/processed/linear_model.linkdata.json', 'r') as f:
    linkdata = json.load(f)
with open('data/testing/processed/linear_model.txt', 'r') as f:
    text = f.read()
# No heading:1 ends here

# [[file:experiment.org::+begin_src python][No heading:2]]
link_wrap_pattern = re.compile(r"(?P<wrapstart>__WRAPSTART(?P<uuid>.+?)__)\s(?P<content>.+?)\s(?P<wrapend>__WRAPEND__)", flags=re.DOTALL)
link_wrap_split_pattern = re.compile(r"(__WRAPSTART.+?__\s.+?\s__WRAPEND__)", flags=re.DOTALL)
split = link_wrap_split_pattern.split(text)
link_data = []
processed = []
offset = 0
for i, s in enumerate([s for s in split if s != '']):
    match = link_wrap_pattern.match(s)
    content = s
    uuid = None
    if match:
        content = ' ' + match['content'] + ' '
        uuid = match['uuid']
    end = offset + len(content)
    processed.append((content, uuid, (offset + 1, end - 1)))
    offset = end
# No heading:2 ends here

# [[file:experiment.org::+begin_src python][No heading:3]]
joined = ''.join([s for s, _, _ in processed])
doc = nlp(joined)
with doc.retokenize() as retokenizer:
    for s, uuid, (start, end) in (p for p in processed if p[1] is not None):
        content = doc.char_span(start, end)
        print(s, (start, end), content)
        retokenizer.merge(content)
        content_link = linkdata[uuid]
        if content_link is None:
            raise ValueError('UUID could not be found in link data')
        for t in content:
            t._.content = content_link
# print('DOC: ', doc)
# for t in doc:
#     if t._.content:
#         print('TOK: ', t, t._.content)
# No heading:3 ends here

# [[file:experiment.org::+begin_src python][No heading:4]]
nlp = spacy.blank("en")

db = DocBin()
doc = nlp(joined)

spans = []
for s, uuid, (start, end) in (p for p in processed if p[1] is not None):
    content_link = linkdata[uuid]
    if content_link is None:
        raise ValueError('UUID could not be found in link data')
    content = doc.char_span(start, end, label=content_link['link'])
    for t in content:
        t._.content = content_link
    spans.append(content)

doc.spans['sc'] = spans

db.add(doc)
db.to_disk("./train.spacy")
# No heading:4 ends here
