import spacy
from spacy.tokens import DocBin
import sys
import glob
import os

if __name__ == '__main__':
    data_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    pattern = os.path.join(data_dir, '**', '*.' + dataset_name + '.spacy')
    all_inputs = glob.glob(pattern, recursive=True)
    db = DocBin()
    out = dataset_name + '.spacy'
    print('Making:', out)
    for path in all_inputs:
        db.merge(DocBin().from_disk(path))
    print('Writing:', out)
    db.to_disk(out)
