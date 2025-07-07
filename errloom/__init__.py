import torch._dynamo

from errloom.comm import CommModel
from errloom.utils.logging_utils import setup_logging
from errloom import discovery
from errloom.attractor import Attractor
from errloom.loom import Loom

torch._dynamo.config.suppress_errors = True  # type: ignore
__version__ = "0.1.0"

setup_logging()

discovery.crawl_package(
    'errloom',
    base_classes=[Attractor, Loom, CommModel],
    check_has_attr=['__holo__']
)