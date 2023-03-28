import os
import collections
from itertools import islice

def change_extension(path, ext):
    name = os.path.splitext(os.path.basename(path))[0]
    dirname = os.path.dirname(path)
    return os.path.join(dirname, name + ext)

def sliding_window(iterable, n):
    """sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG

    From https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    it = iter(iterable)
    window = collections.deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)
