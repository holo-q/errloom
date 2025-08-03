import typing
from typing import List, TypeAlias, Union

if typing.TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

Data: TypeAlias = Union['DatasetDict', 'Dataset', 'IterableDatasetDict', 'IterableDataset']
APIChat: TypeAlias = List[dict[str, str]]


def is_data_type(obj) -> bool:
    """Lazy isinstance check for Data type alias without import overhead"""
    try:
        from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
        return isinstance(obj, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict))
    except ImportError:
        return False
