__version__ = "0.1.0"

from errloom.lib import log
from errloom.tapestry import Rollout

log.setup_logging(
    level="info",
    highlight=True,
    print_paths=True,
    print_threads=False,
    reset_log_columns=False,
)

from errloom.argp import errlargs
from errloom.comm import CommModel
from errloom.lib import discovery
from errloom.attractor import Attractor
from errloom.holoware.holoware import Span
from errloom.loom import Loom
from errloom.paths import userdir
import logging

logger = logging.getLogger(__name__)

# TODO use storage
log.setup_logging(
    level="debug" if errlargs.debug else errlargs.log_level,
    highlight=True,
    print_paths=True or errlargs.print_paths,  # TODO userconf override flag
    print_threads=errlargs.print_threads,  # TODO userconf override flag
    reset_log_columns=errlargs.reset_log_columns,
)

# TODO make this a command line option (list of special loggers to enable)
# log.disable_logger("errloom.tapestry")
log.disable_logger("errloom.lib.discovery")
log.disable_logger("errloom.holoware.holoware_parser")

logger.debug("🔍 Starting package discovery for errloom...")
discovery.crawl_package_fast(
    "errloom",
    base_classes=[Attractor, Loom, CommModel, Span],
    method_patterns=["__holo__"],
    skip_patterns=[
        "errloom.gui",  # GUI modules - not needed for core functionality
        "errloom.deploy",  # Deployment modules - heavy imports
        "errloom.tui",  # Terminal UI modules
        "errloom.training",  # Training modules - heavy ML imports
        "errloom.interop.vast",  # Vast-specific interop modules
    ],
)

# you must reference all the classes of the library directly
# if there is an update to the library and the class is now missing
# you must investigate what is the new path and name of the class
# this means that we get closer to truth together and is crucial
# for exponential acceleration without plateaus

# from errloom.holoware.holoware_handlers import HolowareHandlers

# for name,Class in discovery.get_all_classes().items():
#     if issubclass(Class, Span):
#         if type(Class).__name__ not in HolowareHandlers.__dict__:
#             log(f"HoloSpan class {name} has no handler in HolowareHandlers")

def test_rollout_conversation():
    roll = Rollout({})
    roll.new_context()
    roll.add_frozen("system", """
You are an expert in information theory and symbolic compression.
Your task is to compress text losslessly into a non-human-readable format optimized for density.
Abuse language mixing, abbreviations, and unicode symbols to aggressively compress the input while retaining ALL information required for full reconstruction.
        """)
    roll.add_frozen(ego="user", content="Foo1")
    roll.add_frozen(ego="assistant", content="Bar2")
    roll.new_context()
    roll.add_frozen("system", "It's still a test!")
    roll.add_frozen(ego="user", content="Foo1")
    roll.add_frozen(ego="assistant", content="Bar2")

    logger.info(roll.to_rich())

