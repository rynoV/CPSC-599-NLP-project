#!/bin/sh

fd -e spacy . $1 --exec rm {}
python preprocess.py $1 "$2-$3"
python make_dataset.py $1 train
python make_dataset.py $1 test
