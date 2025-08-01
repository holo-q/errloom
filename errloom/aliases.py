from typing import List, TypeAlias, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

Data: TypeAlias = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
APIChat: TypeAlias = List[dict[str, str]]
