import logging
from abc import ABC

# Setup logging for tests
# setup_logging(level="DEBUG", print_path=True)

from errloom.holoware.holophore import Holophore
from errloom.holoware.holoware import Holoware, ClassSpan
from errloom.tapestry import Rollout
from errloom.lib import log
from tests.base import ErrloomTest
from errloom.holoware.holoware import TextSpan

logger = log.getLogger(__name__)

# Mock classes for testing
# ----------------------------------------

class MockLoom:
    def __init__(self, sample_text="mocked_sample"):
        self.sample_text = sample_text

    def sample(self, rollout):
        return self.sample_text


class HoloClass:
    """A versatile mock class for testing holoware execution."""
    def __init__(self, *kargs, **kwargs):
        self.init_args = (kargs, kwargs)
        self.init_called = True
        self.holo_init_called = False
        self.holo_end_called = False
        self.holo_called = False
        self.last_holo_args = None
        self.last_holo_init_args = None
        self.last_holo_end_args = None

    def __holo_init__(self, holophore, span):
        self.holo_init_called = True
        self.last_holo_init_args = (holophore, span)

    def __holo__(self, holophore, span):
        self.holo_called = True
        self.last_holo_args = (holophore, span)
        return f"Holo! kargs={span.kargs}, kwargs={span.kwargs}"

    def __holo_end__(self, holophore, span):
        self.holo_end_called = True
        self.last_holo_end_args = (holophore, span)

class HoloTest(ErrloomTest, ABC):
    def setUp(self) -> None:
        super().setUp()
        self.loom = MockLoom()
        self.env = {
            "HoloTest": HoloClass,
            "my_var":   "injected_value",
        }
        self.holoware: Holoware | None = None
        self.holophore: Holophore | None = None
        self.rollout: Rollout | None = None

    def run_holoware(self, code: str) -> tuple[Holoware, Holophore]:
        """Helper function to parse and run a holoware string."""
        holoware = Holoware.parse(code)
        logger.info("=== PARSED: ===")
        logger.info(holoware.to_rich())

        # logging.getLogger("errloom.holoware").setLevel(logging.DEBUG)
        self.holoware = holoware
        self.rollout = Rollout(row={})
        self.holophore = Holophore(loom=self.loom, rollout=self.rollout, env=self.env)
        # Force dry mode for attractors that check it
        setattr(self.holophore, "dry", True)

        logging.getLogger().setLevel(logging.DEBUG)

        logger.info("=== EXECUTING: ===")
        holoware(self.holophore)

        logger.info("=== RESULT: ===")
        self.rollout.to_api_chat()
        logging.getLogger().setLevel(logging.INFO)
        logger.info(self.rollout.to_rich())

        return holoware, self.holophore

# Tests
# ----------------------------------------

class HolowareExecutionTest(HoloTest):
    def test_holoware_run_simple_text(self):
        code = "Hello, world!"
        holoware, holophore = self.run_holoware(code)

        self.assertEqual(len(holophore.contexts), 1)
        context = holophore.contexts[0]
        # Assert directly on fragments: single user text fragment rendered in rich
        self.assertEqual(len(context.fragments), 1)
        self.assertEqual(context.fragments[0].content, "Hello, world!")

    def test_holoware_run_ego_change(self):
        code = "<|o_o|>User message.<|@_@|>Assistant response."
        holoware, holophore = self.run_holoware(code)

        self.assertEqual(len(holophore.contexts), 1)
        context = holophore.contexts[0]
        frags = context.fragments
        # Based on rich output, we get two text fragments only (roles normalized on render)
        self.assertEqual(len(frags), 2)
        self.assertIn("User message.", frags[0].content)
        self.assertIn("Assistant response.", frags[1].content)
        self.assertEqual(holophore.ego, "assistant")

    def test_holoware_run_obj_span(self):
        code = "<|o_o|>Value is <|my_var|>."
        holoware, holophore = self.run_holoware(code)

        self.assertEqual(len(holophore.contexts), 1)
        context = holophore.contexts[0]
        frags = context.fragments
        # From rich, content split across lines: "Value is <obj..." then "."
        self.assertGreaterEqual(len(frags), 2)
        self.assertIn("<obj id=my_var>injected_value</obj>", "".join(f.content for f in frags))

    def test_holoware_run_class_lifecycle(self):
        code = "<|o_o|><|HoloTest|>"
        holoware, holophore = self.run_holoware(code)

        self.assertEqual(len(holophore.span_bindings), 1)
        instance = list(holophore.span_bindings.values())[0]

        self.assertIsInstance(instance, HoloClass)
        self.assertTrue(instance.init_called)
        self.assertTrue(instance.holo_init_called)
        self.assertTrue(instance.holo_called)
        self.assertTrue(instance.holo_end_called)

        context = holophore.contexts[0]
        frags = context.fragments
        # Expect at least one fragment with class output
        self.assertGreaterEqual(len(frags), 1)
        self.assertTrue(any("Holo! kargs=[], kwargs={}" in f.content for f in frags))

    def test_holoware_run_class_with_args(self):
        code = "<|o_o|><|HoloTest karg1 karg2 key1=val1|>"
        holoware, holophore = self.run_holoware(code)

        instance = list(holophore.span_bindings.values())[0]

        self.assertEqual(instance.init_args[0], ("karg1", "karg2"))
        self.assertEqual(instance.init_args[1], {"key1": "val1"})

        self.assertIsInstance(instance.last_holo_init_args[0], Holophore)
        self.assertIsInstance(instance.last_holo_init_args[1], ClassSpan)

        self.assertIsInstance(instance.last_holo_args[0], Holophore)
        self.assertIsInstance(instance.last_holo_args[1], ClassSpan)

        self.assertIsInstance(instance.last_holo_end_args[0], Holophore)
        self.assertIsInstance(instance.last_holo_end_args[1], ClassSpan)

    def test_holoware_run_sample_span(self):
        code = "<|@_@ goal=test|>"
        holoware, holophore = self.run_holoware(code)

        context = holophore.contexts[0]
        frags = context.fragments
        # assistant content fragment containing wrapped sample
        self.assertGreaterEqual(len(frags), 1)
        self.assertTrue(any(f.content == "<test>mocked_sample</test>" for f in frags))

    def test_holoware_run_context_reset(self):
        code = "<|o_o|>First context.<|+++|>Second context."
        holoware, holophore = self.run_holoware(code)

        self.assertEqual(len(holophore.contexts), 2)
        frags0 = holophore.contexts[0].fragments
        self.assertTrue(any(f.content == "First context." for f in frags0))

        frags1 = holophore.contexts[1].fragments
        self.assertTrue(any(f.content == "Second context." for f in frags1))
        self.assertEqual(holophore.ego, "system")

    def test_holoware_with_body(self):
        code = """
        <|o_o|>
        <|BodyHoloTest|>
            I am a body.
        """

        class BodyHoloTest(HoloClass):
            def __holo__(self, holophore, span):
                self.holo_called = True
                self.last_holo_args = (holophore, span)
                # Fix: Check if body exists and has spans before accessing
                if span.body and span.body.spans:
                    text_span = span.body.spans[0]  # Changed index to 0 since TextSpan is first
                    if hasattr(text_span, 'text'):
                        return f"Body text: {text_span.text}"
                return "Body text: fallback"

        self.env["BodyHoloTest"] = BodyHoloTest
        holoware, holophore = self.run_holoware(code)

        instance = list(holophore.span_bindings.values())[0]
        self.assertTrue(instance.holo_called)

        context = holophore.contexts[0]
        frags = context.fragments
        self.assertTrue(any("Body text: I am a body." in f.content for f in frags))


COMPRESSOR_HOL = """<|+++|>
You are an expert in information theory and symbolic compression.
Your task is to compress text losslessly into a non-human-readable format optimized for density.
Abuse language mixing, abbreviations, and unicode symbols to aggressively compress the input while retaining ALL information required for full reconstruction.

<|o_o|>
<|BingoAttractor|>
    Compress the following text losslessly in a way that fits a Tweet, such that you can reconstruct it as closely as possible to the original.
    Abuse of language  mixing, abbreviation, symbols (unicode and emojis) to aggressively compress it, while still keeping ALL the information to fully reconstruct it.
    Do not make it human readable.
<|text|original|input|data|>

<|@_@|>
<|@_@:compressed <>compress|>


<|+++|>
You are an expert in information theory and symbolic decompression.
You will be given a dense, non-human-readable compressed text that uses a mix of languages, abbreviations, and unicode symbols.
Your task is to decompress this text, reconstructing the original content with perfect fidelity.

<|o_o|>
Please decompress this content:
<|compressed|>

<|@_@|>
<|@_@:decompressed <>decompress|>


<|===|>
You are a precise content evaluator.
Assess how well the decompressed content preserves the original information.
Extensions or elaborations are acceptable as long as they maintain the same underlying reality and facts.

<|o_o|>
Please observe the following text objects:
<|original|>
<|decompressed|>
<|BingoAttractor|>
    Compare the two objects and analyze the preservation of information and alignment.
    Focus on identifying actual losses or distortions of meaning.
Output your assessment in this format:
<|FidelityCritique|>

<|@_@|>
<|@_@ <>think|>
<|@_@ <>json|>

<|FidelityAttractor original decompressed|>
"""

class CompressorHolowareExecutionTest(HoloTest):
    def setUp(self) -> None:
        super().setUp()
        self.loom = MockLoom()
        # Update the main env with the mock values needed for the compressor test
        self.env.update({
            "text": "This is the original text.",
            "original": "This is the original text.",
            "compressed": "th_is_s_th_0r1g_txt",
            "decompressed": "This is the original text, decompressed.",
            "BingoAttractor": MockBingoAttractor,
            "FidelityCritique": MockFidelityCritique,
            "FidelityAttractor": MockFidelityAttractor,
        })


    def test_holoware_run_compressor_holoware(self):
        holoware, holophore = self.run_holoware(COMPRESSOR_HOL)

        # Assert that the correct contexts are marked for training
        self.assertEqual(holoware.trained_contexts, [0, 1])

        # 1. Check that all 3 contexts were created
        self.assertEqual(len(holophore.contexts), 3)

        # 2. Check that all class instances were created and bound
        self.assertEqual(len(holophore.span_bindings), 4)

        bingo_attractor1 = list(holophore.span_bindings.values())[0]
        bingo_attractor2 = list(holophore.span_bindings.values())[1]
        fidelity_critique = list(holophore.span_bindings.values())[2]
        fidelity_attractor = list(holophore.span_bindings.values())[3]

        self.assertIsInstance(bingo_attractor1, MockBingoAttractor)
        self.assertIsInstance(bingo_attractor2, MockBingoAttractor)
        self.assertIsInstance(fidelity_critique, MockFidelityCritique)
        self.assertIsInstance(fidelity_attractor, MockFidelityAttractor)

        # 3. Check arguments for FidelityAttractor
        self.assertEqual(fidelity_attractor.original_id, "original")
        self.assertEqual(fidelity_attractor.decompressed_id, "decompressed")

        # 4. Check that __holo__ methods were called
        self.assertTrue(bingo_attractor1.holo_called)
        self.assertTrue(bingo_attractor2.holo_called)
        self.assertTrue(fidelity_critique.holo_called)
        self.assertTrue(fidelity_attractor.holo_called)


        # 5. Check the final rendered output for each context
        context0_content = "".join(frag.content for frag in holophore.contexts[0].fragments)
        context1_content = "".join(frag.content for frag in holophore.contexts[1].fragments)
        context2_content = "".join(frag.content for frag in holophore.contexts[2].fragments)

        # Context 0: Compression
        self.assertIn("BINGO: Compress the following text losslessly", context0_content)
        self.assertIn("<obj id=text>This is the original text.</obj>", context0_content)
        self.assertIn("<compress>mocked_sample</compress>", context0_content)
        self.assertNotIn("<decompress>", context0_content)
        self.assertNotIn("BINGO: Compare the two objects", context0_content)


        # Context 1: Decompression
        self.assertIn("<obj id=compressed>th_is_s_th_0r1g_txt</obj>", context1_content)
        self.assertIn("<decompress>mocked_sample</decompress>", context1_content)
        self.assertNotIn("<compress>", context1_content)
        self.assertNotIn("FidelityCritique", context1_content)

        # Context 2: Evaluation
        self.assertIn("<obj id=original>This is the original text.</obj>", context2_content)
        self.assertIn("<obj id=decompressed>This is the original text, decompressed.</obj>", context2_content)
        self.assertIn("BINGO: Compare the two objects and analyze", context2_content)
        self.assertIn('{"mock_schema": "compact"}', context2_content)
        self.assertIn("<think>mocked_sample</think>", context2_content)
        self.assertIn("<json>mocked_sample</json>", context2_content)
        self.assertNotIn("<compress>", context2_content)


class MockFidelityCritique(HoloClass):
    def __holo__(self, holophore, span):
        self.holo_called = True
        self.last_holo_args = (holophore, span)
        return '{"mock_schema": "compact"}'

class MockBingoAttractor(HoloClass):
    def __holo__(self, holophore, span):
        self.holo_called = True
        self.last_holo_args = (holophore, span)
        if span.body and span.body.spans:
            text_span = span.body.first_span_by_type(TextSpan)
            if hasattr(text_span, 'text'):
                return f"BINGO: {text_span.text.strip()}"
        return "BINGO: <no body>"

class MockFidelityAttractor(HoloClass):
    def __init__(self, original_id, decompressed_id, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.original_id = original_id
        self.decompressed_id = decompressed_id

    def __holo__(self, holophore, span):
        # This class doesn't produce content, it's for scoring.
        # We set flags to confirm it was processed.
        self.holo_called = True
        self.last_holo_args = (holophore, span)
        return None
