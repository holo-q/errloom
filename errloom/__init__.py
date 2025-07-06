import torch._dynamo

from .utils import setup_logging

torch._dynamo.config.suppress_errors = True # type: ignore

__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

