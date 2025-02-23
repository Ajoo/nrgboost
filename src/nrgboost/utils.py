from dataclasses import dataclass
from typing import Callable
import tempfile
import os
from joblib import dump, load


def dump_memmap(array, filename=None):
    if filename is None:
        temp_folder = tempfile.mkdtemp()
        filename = os.path.join(temp_folder, 'array.mmap')
    if os.path.exists(filename):
        os.unlink(filename)

    _ = dump(array, filename)
    return load(filename, mmap_mode='r')


@dataclass(frozen=True, slots=True)
class _Indexer:
    base: Callable

    def __getitem__(self, index):
        return self.base(index)


@dataclass(frozen=True, slots=True)
class _VariableLengthIndexer(_Indexer):
    length: Callable

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        return self.base(index)

    def __len__(self):
        return self.length()


@dataclass(frozen=True, slots=True)
class _FixedLengthIndexer(_Indexer):
    length: int

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        return self.base(index)

    def __len__(self):
        return self.length
