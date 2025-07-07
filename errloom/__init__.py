import torch._dynamo

from errloom.comm import CommModel
from errloom import discovery
from errloom.attractor import Attractor
from errloom.loom import Loom
from errloom.utils.log import setup_logging

torch._dynamo.config.suppress_errors = True  # type: ignore
__version__ = "0.1.0"

setup_logging('info')

discovery.crawl_package(
    'errloom',
    base_classes=[Attractor, Loom, CommModel],
    check_has_attr=['__holo__']
)

# you must reference all the classes of the library directly
# if there is an update to the library and the class is now missing
# you must investigate what is the new path and name of the class
# this means that we get closer to truth together and is crucial
# for exponential acceleration without plateaus