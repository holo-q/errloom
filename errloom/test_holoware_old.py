import pytest
from errloom.holoware import ClassSpan, ContextResetSpan, EgoSpan, Holoware, ObjSpan, SamplerSpan, TextSpan

def holoware_to_text(holoware: Holoware) -> str:
    """Recursively extract text from a Holoware object and its nested bodies."""
    text_parts = []
    if not holoware or not holoware.spans:
        return ""
    for span in holoware.spans:
        if isinstance(span, TextSpan):
            text_parts.append(span.text)
        elif isinstance(span, ClassSpan) and span.body:
            text_parts.append(holoware_to_text(span.body))
    return "".join(text_parts)

def test_parse_simple_prompt():
    prompt_content = """<|+++|>
You are a helpful assistant.
<|o_o|>
What is the capital of France?
<|@_@|>
The capital of France is Paris."""

    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert isinstance(spans[0], ContextResetSpan)
    assert spans[0].train is True

    assert isinstance(spans[1], EgoSpan)
    assert spans[1].ego == 'system'

    assert isinstance(spans[2], TextSpan)
    assert "helpful assistant" in spans[2].text

    assert isinstance(spans[3], EgoSpan)
    assert spans[3].ego == 'user'

    assert isinstance(spans[4], TextSpan)
    assert "capital of France" in spans[4].text

    assert isinstance(spans[5], EgoSpan)
    assert spans[5].ego == 'assistant'

    assert isinstance(spans[6], TextSpan)
    assert "Paris" in spans[6].text

def test_parse_compressor_prompt():
    prompt_content = """<|+++|>
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
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    # First conversation
    assert isinstance(spans[0], ContextResetSpan)
    assert spans[0].train is True
    assert isinstance(spans[1], EgoSpan) and spans[1].ego == 'system'
    assert isinstance(spans[2], TextSpan)
    assert isinstance(spans[3], EgoSpan) and spans[3].ego == 'user'

    bingo_span = spans[4]
    assert isinstance(bingo_span, ClassSpan) and bingo_span.class_name == 'BingoAttractor'
    assert bingo_span.body is not None
    assert len(bingo_span.body.spans) > 0

    assert isinstance(spans[5], ObjSpan) and spans[5].var_ids == ['text', 'original', 'input', 'data']
    assert isinstance(spans[6], EgoSpan) and spans[6].ego == 'assistant'
    assert isinstance(spans[7], SamplerSpan) and spans[7].uuid == 'compressed' and spans[7].fence == 'compress'

    # Second conversation
    assert isinstance(spans[8], ContextResetSpan)
    assert spans[8].train is True
    assert isinstance(spans[9], EgoSpan) and spans[9].ego == 'system'
    assert isinstance(spans[10], TextSpan)
    assert isinstance(spans[11], EgoSpan) and spans[11].ego == 'user'
    assert isinstance(spans[12], TextSpan)
    assert isinstance(spans[13], ObjSpan) and spans[13].var_ids == ['compressed']
    assert isinstance(spans[14], EgoSpan) and spans[14].ego == 'assistant'
    assert isinstance(spans[15], SamplerSpan) and spans[15].uuid == 'decompressed' and spans[15].fence == 'decompress'

    # Third conversation
    assert isinstance(spans[16], ContextResetSpan)
    assert spans[16].train is False
    assert isinstance(spans[17], EgoSpan) and spans[17].ego == 'system'
    assert isinstance(spans[18], TextSpan)
    assert isinstance(spans[19], EgoSpan) and spans[19].ego == 'user'
    assert isinstance(spans[20], TextSpan)
    assert isinstance(spans[21], ObjSpan) and spans[21].var_ids == ['original']
    assert isinstance(spans[22], ObjSpan) and spans[22].var_ids == ['decompressed']
    assert isinstance(spans[23], ClassSpan) and spans[23].class_name == 'BingoAttractor'
    assert spans[23].body is not None
    body_text = holoware_to_text(spans[23].body)
    assert "Compare the two objects" in body_text
    assert isinstance(spans[24], TextSpan)
    assert isinstance(spans[25], ClassSpan) and spans[25].class_name == 'FidelityCritique'
    assert isinstance(spans[26], EgoSpan) and spans[26].ego == 'assistant'
    assert isinstance(spans[27], SamplerSpan) and spans[27].fence == 'think'
    assert isinstance(spans[28], SamplerSpan) and spans[28].fence == 'json'
    assert isinstance(spans[29], ClassSpan) and spans[29].class_name == 'FidelityAttractor'
    assert spans[29].var_args == ['original', 'decompressed']

def test_parse_empty_and_minimal():
    # Empty string
    holoware = Holoware.parse("")
    assert len(holoware.spans) == 0

    # Whitespace only
    holoware = Holoware.parse("  \n\t  ")
    assert len(holoware.spans) == 0

    # Only a context reset
    holoware = Holoware.parse("<|+++|>")
    assert len(holoware.spans) == 1
    assert isinstance(holoware.spans[0], ContextResetSpan)

def test_comment_filtering():
    prompt_content = """
# This is a full-line comment.
<|o_o|>
# This is another comment.
Hello user.
<|@_@|> # This is an inline comment, which should be parsed as text.
Hello assistant.
"""
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert len(spans) == 4
    assert isinstance(spans[0], EgoSpan) and spans[0].ego == 'user'
    assert isinstance(spans[1], TextSpan) and "Hello user" in spans[1].text
    assert isinstance(spans[2], EgoSpan) and spans[2].ego == 'assistant'
    assert isinstance(spans[3], TextSpan)
    assert "# This is an inline comment" in spans[3].text
    assert "Hello assistant" in spans[3].text

def test_class_span_indentation():
    prompt_content = """<|o_o|>
<|MyClass|>
    This is indented with spaces.
<|MyClass|>
	This is indented with a tab.
<|MyClass|>
This is not indented.
"""
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert len(spans) == 5
    assert isinstance(spans[0], EgoSpan)

    # First indented block
    assert isinstance(spans[1], ClassSpan) and spans[1].class_name == "MyClass"
    assert spans[1].body is not None
    body_spans_1 = spans[1].body.spans
    assert len(body_spans_1) == 2
    assert isinstance(body_spans_1[0], EgoSpan) and body_spans_1[0].ego == 'system'
    assert isinstance(body_spans_1[1], TextSpan)
    assert "indented with spaces" in body_spans_1[1].text

    # Second indented block
    assert isinstance(spans[2], ClassSpan) and spans[2].class_name == "MyClass"
    assert spans[2].body is not None
    body_spans_2 = spans[2].body.spans
    assert len(body_spans_2) == 2
    assert isinstance(body_spans_2[0], EgoSpan) and body_spans_2[0].ego == 'system'
    assert isinstance(body_spans_2[1], TextSpan)
    assert "indented with a tab" in body_spans_2[1].text

    # Not indented
    assert isinstance(spans[3], ClassSpan) and spans[3].class_name == "MyClass"
    assert spans[3].body is None

    # Text after non-indented class
    assert isinstance(spans[4], TextSpan)
    assert "This is not indented" in spans[4].text

def test_sampler_span_parsing():
    prompt_content = "<|@_@:sampler_id <>my_fence goal=do_stuff|>"
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert len(spans) == 2
    assert isinstance(spans[0], EgoSpan) and spans[0].ego == 'assistant'
    assert isinstance(spans[1], SamplerSpan)
    assert spans[1].uuid == 'sampler_id'
    assert spans[1].goal == 'do_stuff'
    assert spans[1].fence == 'my_fence'

def test_sampler_span_no_id_parsing():
    prompt_content = "<|@_@ <>my_fence|>"
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert len(spans) == 2
    assert isinstance(spans[0], EgoSpan) and spans[0].ego == 'assistant'
    assert isinstance(spans[1], SamplerSpan)
    assert spans[1].uuid == ''
    assert spans[1].fence == 'my_fence'

def test_error_conditions():
    # Unclosed tag
    with pytest.raises(ValueError, match="Unclosed tag"):
        Holoware.parse("<|o_o|")

    # Content before ego
    with pytest.raises(ValueError, match="Cannot have ClassSpan before a role"):
        Holoware.parse("<|MyClass|><|o_o|>")


def test_whitespace_handling():
    prompt_content = "  <|o_o|>  \n  some text  \n\n<|@_@|>  "
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert len(spans) == 3
    assert isinstance(spans[0], EgoSpan) and spans[0].ego == 'user'
    assert isinstance(spans[1], TextSpan) and spans[1].text == "some text  \n\n"
    assert isinstance(spans[2], EgoSpan) and spans[2].ego == 'assistant'

def test_implicit_system_role():
    prompt_content = """<|+++|>
This is a system prompt.
<|o_o|>
This is a user prompt."""
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert len(spans) == 5
    assert isinstance(spans[0], ContextResetSpan)
    assert isinstance(spans[1], EgoSpan) and spans[1].ego == 'system'
    assert isinstance(spans[2], TextSpan) and "system prompt" in spans[2].text
    assert isinstance(spans[3], EgoSpan) and spans[3].ego == 'user'
    assert isinstance(spans[4], TextSpan) and "user prompt" in spans[4].text

def test_no_duplicate_role_spans():
    prompt_content = "<|o_o|><|o_o|>Hello"
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert len(spans) == 2
    assert isinstance(spans[0], EgoSpan) and spans[0].ego == 'user'
    assert isinstance(spans[1], TextSpan) and spans[1].text == 'Hello'

def test_class_span_with_holoware_body():
    prompt_content = """<|o_o|>
<|MyContainerClass|>
    This is the body.
    <|@_@|>
    Inner assistant message.
This is outside the block."""
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    assert len(spans) == 3
    assert isinstance(spans[0], EgoSpan)
    assert spans[0].ego == 'user'

    container_span = spans[1]
    assert isinstance(container_span, ClassSpan)
    assert container_span.class_name == "MyContainerClass"
    assert hasattr(container_span, 'body')
    assert isinstance(container_span.body, Holoware)

    body_spans = container_span.body.spans

    # The body is "    This is the body.\n    <|@_@|>\n    Inner assistant message."
    # After dedenting and parsing, an implicit system ego should be created.
    assert len(body_spans) == 4
    assert isinstance(body_spans[0], EgoSpan) and body_spans[0].ego == 'system'
    assert isinstance(body_spans[1], TextSpan) and body_spans[1].text.strip() == "This is the body."
    assert isinstance(body_spans[2], EgoSpan) and body_spans[2].ego == 'assistant'
    assert isinstance(body_spans[3], TextSpan) and body_spans[3].text.strip() == "Inner assistant message."

    outer_text_span = spans[2]
    assert isinstance(outer_text_span, TextSpan)
    assert outer_text_span.text.strip() == "This is outside the block."

def test_deeply_nested_class_spans():
    prompt_content = """<|o_o|>
<|Foo|>
    Foo's body.
    <|Bar|>
        Bar's body.
        <|Baz|>
            Baz's body.
    <|Bar|>
        Another Bar body.
"""
    holoware = Holoware.parse(prompt_content)
    spans = holoware.spans

    # Top level: user ego, Foo class
    assert len(spans) == 2
    assert isinstance(spans[0], EgoSpan) and spans[0].ego == 'user'
    
    foo_span = spans[1]
    assert isinstance(foo_span, ClassSpan) and foo_span.class_name == 'Foo'
    assert foo_span.body is not None

    # Inside Foo's body
    foo_body_spans = foo_span.body.spans
    # system ego, text, Bar class, Bar class
    assert len(foo_body_spans) == 4
    assert isinstance(foo_body_spans[0], EgoSpan) and foo_body_spans[0].ego == 'system'
    assert isinstance(foo_body_spans[1], TextSpan) and "Foo's body" in foo_body_spans[1].text
    
    # First Bar span
    bar_span_1 = foo_body_spans[2]
    assert isinstance(bar_span_1, ClassSpan) and bar_span_1.class_name == 'Bar'
    assert bar_span_1.body is not None

    # Inside first Bar's body
    bar_body_1_spans = bar_span_1.body.spans
    # system ego, text, Baz class
    assert len(bar_body_1_spans) == 3
    assert isinstance(bar_body_1_spans[0], EgoSpan) and bar_body_1_spans[0].ego == 'system'
    assert isinstance(bar_body_1_spans[1], TextSpan) and "Bar's body" in bar_body_1_spans[1].text

    # Baz span
    baz_span = bar_body_1_spans[2]
    assert isinstance(baz_span, ClassSpan) and baz_span.class_name == 'Baz'
    assert baz_span.body is not None
    
    # Inside Baz's body
    baz_body_spans = baz_span.body.spans
    # system ego, text
    assert len(baz_body_spans) == 2
    assert isinstance(baz_body_spans[0], EgoSpan) and baz_body_spans[0].ego == 'system'
    assert isinstance(baz_body_spans[1], TextSpan) and "Baz's body" in baz_body_spans[1].text

    # Second Bar span
    bar_span_2 = foo_body_spans[3]
    assert isinstance(bar_span_2, ClassSpan) and bar_span_2.class_name == 'Bar'
    assert bar_span_2.body is not None
    
    # Inside second Bar's body
    bar_body_2_spans = bar_span_2.body.spans
    # system ego, text
    assert len(bar_body_2_spans) == 2
    assert isinstance(bar_body_2_spans[0], EgoSpan) and bar_body_2_spans[0].ego == 'system'
    assert isinstance(bar_body_2_spans[1], TextSpan) and "Another Bar body" in bar_body_2_spans[1].text