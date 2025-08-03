from abc import ABC

from testslide import TestCase

from errloom.lib import log

class ErrloomTest(TestCase, ABC):
    def setUp(self):
        super().setUp()
        # log.push(f"\[{self._testMethodName}]")

    def tearDown(self):
        super().tearDown()
        log.clear_stack() # If a test fails, clear the stack to avoid polluting other tests (TODO ErrloomTest that we inherit everywhere)
