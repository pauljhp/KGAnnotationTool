from transformers import AlbertTokenizer, AutoTokenizer, BertTokenizer
import torch
import utils
from typing import Sequence, Iterable, Union, Any


DEVICE = torch.cuda.is_available()

class Tokenizer:
    def __init__(self, embedding: str="bert-base-cased"):
        print("initializing tokenizer")
        if "albert" in embedding:
            tokenizer = AlbertTokenizer.from_pretrained(embedding)
        elif "bert" in embedding:
            tokenizer = AutoTokenizer.from_pretrained(embedding)
        else:
            raise ValueError("unrecogized embedding")
        self.tokenizer = tokenizer
    
    def tokenize(self, input: str):
        """tokenize sentence"""
        return self.tokenizer.tokenize(input)

def list_overlap(parentlist, childlist):
    assert len(childlist), "childlist cannot be empty"
    window_size = len(childlist)
    assert window_size <= len(parentlist), \
        "childlist cannot be longer than parentlist!"
    parent = utils.SlidingList(parentlist)
    for i, chunk in parent.sliding_window_iter(
                        window_size=window_size, 
                        enumeration=True
                    ):
        if list(chunk) == list(childlist):
            return chunk, (i, i + window_size)