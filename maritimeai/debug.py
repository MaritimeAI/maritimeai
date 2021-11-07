from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass


def handler_object(o):
    return None


def sizemem(o, handlers={}, verbose=False):
    handler_dict = lambda x: chain.from_iterable(x.items())
    handlers_all = {
            tuple: iter,
            list: iter,
            deque: iter,
            dict: handler_dict,
            set: iter,
            frozenset: iter,
    }
    handlers_all.update(handlers)
    seen = set()
    size_default = getsizeof(0)

    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = getsizeof(o, size_default)

        if verbose:
            print(s, type(o), repr(o))  # , file=stderr)

        for t, h in handlers_all.items():
            if isinstance(o, t):
                s += sum(map(sizeof, h(o)))
                break
        return s
    return sizeof(o)
