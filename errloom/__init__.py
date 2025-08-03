__version__ = "0.1.0"


from errloom.argp import errlargs
from errloom.comm import CommModel
from errloom.lib import discovery
from errloom.attractor import Attractor
from errloom.holoware.holoware import Span
from errloom.loom import Loom
from errloom.lib import log
from errloom.paths import userdir

# Set up persistence file path
persistence_file = str(userdir / "persistence.json")

log.setup_logging(
    level='debug' if errlargs.debug else errlargs.log_level,
    highlight=True,
    persistence_file=persistence_file,
    print_paths=True or errlargs.print_paths, # TODO userconf override flag
    print_threads=errlargs.print_threads, # TODO userconf override flag
    reset_log_columns=errlargs.reset_log_columns)

# TODO make this a command line option (list of special loggers to enable)
log.disable_logger("errloom.tapestry")

# Debug: Let's see what classes are discovered
import logging
logger = logging.getLogger(__name__)
logger.info("🔍 Starting package discovery for errloom...")

discovery.crawl_package_fast(
    'errloom',
    base_classes=[Attractor, Loom, CommModel, Span],
    method_patterns=['__holo__'],
    skip_patterns=[
        'errloom.gui',           # GUI modules - not needed for core functionality
        'errloom.deploy',        # Deployment modules - heavy imports
        'errloom.tui',           # Terminal UI modules
        'errloom.training',      # Training modules - heavy ML imports
        'errloom.interop.vast',  # Vast-specific interop modules
    ]
)

# Debug: Check if BingoAttractor is in the index
class_index = discovery.get_class_index()
logger.info(f"📋 Class index contains {len(class_index)} classes:")
for class_name in sorted(class_index.keys()):
    if 'Bingo' in class_name or 'Attractor' in class_name:
        logger.info(f"   🎯 Found: {class_name} -> {class_index[class_name]}")

# Debug: Try to get BingoAttractor class
bingo_class = discovery.get_class('BingoAttractor')
if bingo_class:
    logger.info(f"✅ BingoAttractor successfully loaded: {bingo_class}")
else:
    logger.error(f"❌ BingoAttractor could not be loaded!")
    # Let's see what attractor classes are available
    all_classes = discovery.get_all_classes()
    attractor_classes = [name for name in all_classes.keys() if 'Attractor' in name]
    logger.info(f"🔍 Available Attractor classes: {attractor_classes}")

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