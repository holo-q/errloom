__version__ = "0.1.0"


from errloom.argp import errlargs
from errloom.comm import CommModel
from errloom import discovery
from errloom.attractor import Attractor
from errloom.holoware import HoloSpan
from errloom.loom import Loom
from errloom.utils.log import setup_logging

setup_logging(
    level=errlargs.log_level,
    print_path=errlargs.log_more)

discovery.crawl_package(
    'errloom',
    base_classes=[Attractor, Loom, CommModel, HoloSpan],
    check_has_attr=['__holo__']
)

# you must reference all the classes of the library directly
# if there is an update to the library and the class is now missing
# you must investigate what is the new path and name of the class
# this means that we get closer to truth together and is crucial
# for exponential acceleration without plateaus
