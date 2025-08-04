from rich.rule import Rule

from errloom.lib import log
from errloom.tapestry import Rollout
from tests.base import ErrloomTest

logger = log.getLogger(__name__)

class LogAlignmentTests(ErrloomTest):
    def test_rule(self):
        logger.info(Rule(style="dim"))
        logger.info(Rule(style="cyan"))
        logger.info(Rule(style="red"))

    def test_rollout_conversation(self):
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



