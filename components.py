from spacy.language import Language
import torch

@Language.component("remove_trf_data")
def remove_trf_data(doc):
    doc._.trf_data = None
    torch.cuda.empty_cache()
    return doc