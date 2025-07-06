from typing import TypeAlias, Union

import torch._dynamo

from errloom.utils.logging_utils import setup_logging
from datasets import IterableDataset, Dataset, IterableDatasetDict, DatasetDict

torch._dynamo.config.suppress_errors = True  # type: ignore
__version__ = "0.1.0"

Data: TypeAlias = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]

setup_logging()
