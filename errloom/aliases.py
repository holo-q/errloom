from typing import TypeAlias, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

Data: TypeAlias = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
