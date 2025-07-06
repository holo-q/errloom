import pytest
import logging
from errloom.utils.logging_utils import setup_logging

@pytest.fixture(autouse=True)
def rich_logging():
    """Fixture to ensure rich logging is set up for tests."""
    setup_logging()
    yield
    # No teardown needed unless we want to revert logging config 