from collections import deque
from typing import Iterable, Sequence, Any, Union, Optional


class SlidingList(list):
    def __init__(self, data: Optional[Iterable[Any]]=None, *args):
        """You can either initialize by passing an iterable into data,
        or unpack the iterable and wrap into into *args
        :param data: Iterable, optional.
        """
        self._data = data
    
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, newdata: Iterable[Any]):
        self._data = newdata

    def sliding_window_iter(self, window_size: int, enumeration: bool=False):
        assert len(self.data) >= window_size, \
            "window size cannot be longer than self.data!"
        for i, _ in enumerate(self.data):
            if len(self.data) - i >= window_size:
                if enumeration:
                    yield i, self.data[i: i + window_size]
                else:
                    yield self.data[i: i + window_size]

def is_numeric(char: Optional[Union[str, int]]=None):
    """takes single char"""
    if char:
        if isinstance(char, int):
            return True
        char_str = str(char)
        assert len(char_str) == 1, "is_numeric only takes one char!"
        if ord(char_str) in range(48, 58):
            return True
        else:
            return False
    else:
        return False

def str_is_int(input: Union[int, str]):
    if isinstance(input, int):
        return True
    else:
        assert isinstance(input, str), "only str or int values allowed"
    for i in input:
        if not is_numeric(i):
            return False
    return True