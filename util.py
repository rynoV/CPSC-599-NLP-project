import os

def change_extension(path, ext):
    name = os.path.splitext(os.path.basename(path))[0]
    dirname = os.path.dirname(path)
    return os.path.join(dirname, name + ext)
