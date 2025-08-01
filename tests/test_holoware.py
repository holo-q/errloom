import pytest

# Setup logging for tests
# setup_logging(level="DEBUG", print_path=True)

from errloom.holophore import Holophore
from errloom.holoware import Holoware, ClassSpan
from errloom.tapestry import Rollout


# Mock classes for testing
# ----------------------------------------

class MockLoom:
    def __init__(self, sample_text="mocked_sample"):
        self.sample_text = sample_text

    def sample(self, rollout):
        return self.sample_text


class HoloTest:
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

    def __holo_init__(self, phore, span):
        self.holo_init_called = True
        self.last_holo_init_args = (phore, span)

    def __holo__(self, phore, span):
        self.holo_called = True
        self.last_holo_args = (phore, span)
        return f"Holo! kargs={span.kargs}, kwargs={span.kwargs}"

    def __holo_end__(self, phore, span):
        self.holo_end_called = True
        self.last_holo_end_args = (phore, span)


# Pytest Fixtures
# ----------------------------------------

@pytest.fixture
def mock_loom():
    return MockLoom()


@pytest.fixture
def mock_env():
    """Provides a mock environment with a test class."""
    return {
        "HoloTest": HoloTest,
        "my_var":   "injected_value"
    }


def run_ware(code: str, loom: MockLoom, env: dict) -> Holophore:
    """Helper function to parse and run a holoware string."""
    ware = Holoware.parse(code)
    rollout = Rollout(row={})
    phore = Holophore(loom=loom, rollout=rollout, env=env)
    return ware(phore)


# Tests
# ----------------------------------------

def test_holoware_run_simple_text(mock_loom, mock_env):
    code = "Hello, world!"
    phore = run_ware(code, mock_loom, mock_env)

    assert len(phore.contexts) == 1
    context = phore.contexts[0]
    assert context is not None
    assert len(context.messages) == 1
    assert context.messages[0]['role'] == 'system'
    assert context.messages[0]['content'] == 'Hello, world!'


def test_holoware_run_ego_change(mock_loom, mock_env):
    code = "<|o_o|>User message.<|@_@|>Assistant response."
    phore = run_ware(code, mock_loom, mock_env)

    assert len(phore.contexts) == 1
    context = phore.contexts[0]
    assert len(context.messages) == 2
    assert context.messages[0]['role'] == 'user'
    assert context.messages[0]['content'] == 'User message.'
    assert context.messages[1]['role'] == 'assistant'
    assert context.messages[1]['content'] == 'Assistant response.'
    assert phore.ego == 'assistant'


def test_holoware_run_obj_span(mock_loom, mock_env):
    code = "<|o_o|>Value is <|my_var|>."
    phore = run_ware(code, mock_loom, mock_env)

    assert len(phore.contexts) == 1
    context = phore.contexts[0]
    assert len(context.messages) == 1
    assert context.messages[0]['role'] == 'user'
    expected_content = "Value is <obj id=my_var>injected_value</obj>."
    assert context.messages[0]['content'] == expected_content


def test_holoware_run_class_lifecycle(mock_loom, mock_env):
    code = "<|o_o|><|HoloTest|>"
    phore = run_ware(code, mock_loom, mock_env)

    assert len(phore.span_bindings) == 1
    instance = list(phore.span_bindings.values())[0]

    assert isinstance(instance, HoloTest)
    assert instance.init_called
    assert instance.holo_init_called
    assert instance.holo_called
    assert instance.holo_end_called

    context = phore.contexts[0]
    assert len(context.messages) == 1
    assert "Holo! kargs=[], kwargs={}" in context.messages[0]['content']


def test_holoware_run_class_with_args(mock_loom, mock_env):
    code = "<|o_o|><|HoloTest karg1 karg2 key1=val1|>"
    phore = run_ware(code, mock_loom, mock_env)

    instance = list(phore.span_bindings.values())[0]

    assert instance.init_args[0] == ('karg1', 'karg2')
    assert instance.init_args[1] == {'key1': 'val1'}

    assert isinstance(instance.last_holo_init_args[0], Holophore)
    assert isinstance(instance.last_holo_init_args[1], ClassSpan)

    assert isinstance(instance.last_holo_args[0], Holophore)
    assert isinstance(instance.last_holo_args[1], ClassSpan)

    assert isinstance(instance.last_holo_end_args[0], Holophore)
    assert isinstance(instance.last_holo_end_args[1], ClassSpan)


def test_holoware_run_sample_span(mock_loom, mock_env):
    code = "<|@_@ goal=test|>"
    phore = run_ware(code, mock_loom, mock_env)

    context = phore.contexts[0]
    assert len(context.messages) == 1
    assert context.messages[0]['role'] == 'assistant'
    assert context.messages[0]['content'] == '<test>mocked_sample</test>'


def test_holoware_run_context_reset(mock_loom, mock_env):
    code = "<|o_o|>First context.<|+++|>Second context."
    phore = run_ware(code, mock_loom, mock_env)

    assert len(phore.contexts) == 2
    assert phore.contexts[0].messages[0]['content'] == 'First context.'
    assert phore.contexts[0].messages[0]['role'] == 'user'

    assert phore.contexts[1].messages[0]['content'] == 'Second context.'
    assert phore.contexts[1].messages[0]['role'] == 'system'
    assert phore.ego == 'system'


def test_holoware_with_body(mock_loom, mock_env):
    class BodyHoloTest(HoloTest):
        def __holo__(self, phore, span):
            self.holo_called = True
            self.last_holo_args = (phore, span)
            text_span = span.body.spans[1]
            return f"Body text: {text_span.text}"

    mock_env["BodyHoloTest"] = BodyHoloTest
    code = """<|o_o|>
<|BodyHoloTest|>
    I am a body.
"""
    phore = run_ware(code, mock_loom, mock_env)

    instance = list(phore.span_bindings.values())[0]
    assert instance.holo_called

    context = phore.contexts[0]
    assert "Body text: I am a body." in context.messages[0]['content']

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

def test_holoware_run_compressor_holoware(mock_loom, mock_env):

    # --- Setup Mocks ---
    mock_env["BingoAttractor"] = HoloTest
    mock_env["FidelityCritique"] = HoloTest
    mock_env["FidelityAttractor"] = HoloTest

    mock_env["text"] = "This is the original text."
    mock_env["original"] = "This is the original text."
    mock_env["compressed"] = "th_is_s_th_0r1g_txt"
    mock_env["decompressed"] = "This is the original text, decompressed."

    # --- Run and Assert ---
    phore = run_ware(COMPRESSOR_HOL, mock_loom, mock_env)

    # Check that all 3 contexts were created
    assert len(phore.contexts) == 3

    # Check that the class instances were created and bound
    assert len(phore.span_bindings) == 4
    instance_types = [type(inst) for inst in phore.span_bindings.values()]
    assert instance_types.count(HoloTest) == 4

    # Check some content from the last context
    last_context = phore.contexts[2]
    assert len(last_context.messages) > 3

    # Ego(user) -> Text -> Obj -> Obj -> Class -> Text ...
    user_turn = last_context.messages[1]
    assert user_turn['role'] == 'user'
    assert "<obj id=original>This is the original text.</obj>" in user_turn['content']
    assert "<obj id=decompressed>This is the original text, decompressed.</obj>" in user_turn['content']
    assert "Holo! kargs=[], kwargs={}" in user_turn['content'] # From BingoAttractor

    assistant_turn = last_context.messages[2]
    assert assistant_turn['role'] == 'assistant'
    assert "<think>mocked_sample</think>" in assistant_turn['content']
    assert "<json>mocked_sample</json>" in assistant_turn['content']
    assert "Holo! kargs=['original', 'decompressed'], kwargs={}" in assistant_turn['content'] # FidelityAttractor