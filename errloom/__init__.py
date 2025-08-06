__version__ = "0.1.0"

from errloom.lib import log
from errloom.tapestry import Rollout

# To make logs palatable during argp setup
log.setup_logging(
    level="info",
    highlight=False,
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

# Now we can actually setup logging for real
# Determine base level: --debug bare wins; else --log-level (default info)
_base_level_str = 'debug' if (isinstance(errlargs.debug, str) and errlargs.debug == '') else errlargs.log_level
log.setup_logging(
    level=_base_level_str,
    highlight=False,
    enable_file_logging=True,
    print_time=False,
    print_paths=errlargs.print_paths,  # TODO userconf override flag
    print_threads=errlargs.print_threads,  # TODO userconf override flag
    reset_log_columns=errlargs.reset_log_columns
)

# Defaults
# log.disable_logger("errloom.tapestry")
# log.disable_logger("errloom.lib.discovery")
# log.disable_logger("errloom.holoware.holoware_parser")

# Initialize logging early based on args
from errloom.lib.log import logging_level_from_str, parse_logs_arg, apply_logger_levels

# Collect per-logger overrides from --logs
overrides: dict[str, int] = {}

# --logs can appear multiple times
if errlargs.logs:
    default_level_for_logs = logging_level_from_str(errlargs.log_level, logging.INFO)
    for expr in errlargs.logs:
        parsed = parse_logs_arg(expr, default_level_for_logs)
        # later occurrences override earlier
        overrides.update(parsed)

# --debug with value: modules at DEBUG (does NOT force global DEBUG if value provided)
if isinstance(errlargs.debug, str) and errlargs.debug not in (None, ''):
    mods = [m.strip() for m in errlargs.debug.split(",") if m.strip()]
    for m in mods:
        overrides[m] = logging.DEBUG

# Apply overrides
apply_logger_levels(overrides)

# l1 = logging.getLogger("errloom.tapestry")
# l2 = logging.getLogger("errloom.tapestry")
# l3 = logging.getLogger("errloom.holoware.holoware_handlers")
# l4 = log.getLogger("errloom.holoware.holoware_handlers")

# print(l1, l2, l1 == l2)
# print(l3, l4, l3 == l4)
# exit(0)

# TODO make this a command line option (list of special loggers to enable)

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
