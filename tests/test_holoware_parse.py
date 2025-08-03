from testslide import TestCase
from rich.panel import Panel
import logging

from errloom.holoware.holoware import (
    ClassSpan,
    ContextResetSpan,
    EgoSpan,
    Holoware,
    ObjSpan,
    SampleSpan,
    TextSpan,
)
from errloom.holoware.holoware_parse import (_build_class, _build_context, _build_ego_or_sampler, _build_obj, filter_comments, HolowareParser, parse_span_tag)
from errloom.lib import log

logger = logging.getLogger(__name__)

def _load_and_print_holoware(code: str) -> Holoware:
    """Loads holoware, prints it in a box, and returns the parsed object."""
    holoware = HolowareParser(code).parse()
    logger.info(Panel(code, title="Holoware Code", expand=False, border_style="cyan"))
    logger.info(holoware.to_rich())
    return holoware

class FilterCommentsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        log.setup_logging("DEBUG", True, True)

    def test_filter_comments_empty(self):
        self.assertEqual(filter_comments(""), "")

    def test_filter_comments_no_comments(self):
        content = "line 1\nline 2"
        self.assertEqual(filter_comments(content), content)

    def test_filter_comments_only_comments(self):
        content = "# comment 1\n# comment 2"
        self.assertEqual(filter_comments(content), "\n")

    def test_filter_comments_mixed(self):
        content = "line 1\n# comment\nline 2"
        self.assertEqual(filter_comments(content), "line 1\n\nline 2")

    def test_filter_comments_with_whitespace(self):
        content = "  # comment with leading whitespace"
        self.assertEqual(filter_comments(content), "")

    def test_filter_comments_inline_content_not_filtered(self):
        content = "line_with_hash = '#value'"
        self.assertEqual(filter_comments(content), content)

class ParseSpanTagTest(TestCase):
    def test_parse_span_tag_simple(self):
        base, kargs, kwargs = parse_span_tag("MyClass")
        self.assertEqual(base, "MyClass")
        self.assertEqual(kargs, [])
        self.assertEqual(kwargs, {})

    def test_parse_span_tag_with_kargs(self):
        base, kargs, kwargs = parse_span_tag("MyClass arg1 arg2")
        self.assertEqual(base, "MyClass")
        self.assertEqual(kargs, ["arg1", "arg2"])
        self.assertEqual(kwargs, {})

    def test_parse_span_tag_with_kwargs(self):
        base, kargs, kwargs = parse_span_tag("MyClass key1=val1 key2=val2")
        self.assertEqual(base, "MyClass")
        self.assertEqual(kargs, [])
        self.assertEqual(kwargs, {"key1": "val1", "key2": "val2"})

    def test_parse_span_tag_mixed_args(self):
        base, kargs, kwargs = parse_span_tag("MyClass arg1 key1=val1 arg2")
        self.assertEqual(base, "MyClass")
        self.assertEqual(kargs, ["arg1", "arg2"])
        self.assertEqual(kwargs, {"key1": "val1"})

    def test_parse_span_tag_with_special_attr(self):
        base, kargs, kwargs = parse_span_tag("MyClass <>something")
        self.assertEqual(base, "MyClass")
        self.assertEqual(kargs, [])
        self.assertEqual(kwargs, {"<>": "something"})

    def test_parse_span_tag_with_empty_special_attr_raises_error(self):
        with self.assertRaises(ValueError) as e:
            parse_span_tag("MyClass <>")
        self.assertIn("Empty <> attribute", str(e.exception))

    def test_parse_span_tag_with_quoted_kwargs(self):
        base, kargs, kwargs = parse_span_tag('MyClass key="value with spaces" another_key=\'another value\'')
        self.assertEqual(base, "MyClass")
        self.assertEqual(kargs, [])
        self.assertEqual(kwargs, {"key": "value with spaces", "another_key": "another value"})

    def test_parse_span_tag_with_mismatched_quotes_raises_error(self):
        with self.assertRaises(ValueError) as e:
            parse_span_tag('MyClass key="value with spaces')
        self.assertIn("No closing quotation", str(e.exception))

class SpanBuildersTest(TestCase):
    def test_build_class_span(self):
        out = []
        _build_class(out, "MyClass", ["arg"], {"kw": "val"})
        self.assertEqual(len(out), 1)
        span = out[0]
        self.assertIsInstance(span, ClassSpan)
        self.assertEqual(span.class_name, "MyClass")
        self.assertEqual(span.kargs, ["arg"])
        self.assertEqual(span.kwargs, {"kw": "val"})

    def test_build_ego_or_sampler_span_ego_only(self):
        out = []
        _build_ego_or_sampler(out, "o_o", [], {})
        self.assertEqual(len(out), 1)
        span = out[0]
        self.assertIsInstance(span, EgoSpan)
        self.assertEqual(span.ego, "user")

    def test_build_ego_or_sampler_span_with_uuid(self):
        out = []
        _build_ego_or_sampler(out, "@_@:123", [], {})
        self.assertEqual(len(out), 1)
        span = out[0]
        self.assertIsInstance(span, EgoSpan)
        self.assertEqual(span.ego, "assistant")
        self.assertEqual(span.uuid, "123")

    def test_build_ego_or_sampler_span_as_sampler(self):
        out = []
        _build_ego_or_sampler(out, "o_o:id", ["karg"], {"goal": "test"})
        self.assertEqual(len(out), 2)
        ego_span, sampler_span = out
        self.assertIsInstance(ego_span, EgoSpan)
        self.assertEqual(ego_span.ego, "user")
        self.assertIsInstance(sampler_span, SampleSpan)
        self.assertEqual(sampler_span.uuid, "id")
        self.assertEqual(sampler_span.kargs, ["karg"])
        self.assertEqual(sampler_span.goal, "test")
        self.assertEqual(sampler_span.kwargs, {})

    def test_build_context_reset_span(self):
        out = []
        handler = _build_context(train=True)
        handler(out, "+++", [], {})
        self.assertEqual(len(out), 1)
        span = out[0]
        self.assertIsInstance(span, ContextResetSpan)
        self.assertTrue(span.train)

    def test_build_obj_span(self):
        out = []
        _build_obj(out, "var1|var2", ["karg"], {})
        self.assertEqual(len(out), 1)
        span = out[0]
        self.assertIsInstance(span, ObjSpan)
        self.assertEqual(span.var_ids, ["var1", "var2"])
        self.assertEqual(span.kargs, ["karg"])

class HolowareParserTest(TestCase):
    def test_ware_parser_basic(self):
        test_cases = [
            ("", []),
            ("   \n\t ", []),
            (
                "just text",
                [
                    {"type": EgoSpan, "attrs": {"ego": "system"}},
                    {"type": TextSpan, "attrs": {"text": "just text"}},
                ],
            ),
            ("<|o_o|>", [{"type": EgoSpan, "attrs": {"ego": "user"}}]),
            ("<|@_@|>", [{"type": EgoSpan, "attrs": {"ego": "assistant"}}]),
            (
                "<|o_o|>Hello",
                [
                    {"type": EgoSpan, "attrs": {"ego": "user"}},
                    {"type": TextSpan, "attrs": {"text": "Hello"}},
                ],
            ),
            (
                "Hello<|o_o|>",
                [
                    {"type": EgoSpan, "attrs": {"ego": "system"}},
                    {"type": TextSpan, "attrs": {"text": "Hello"}},
                    {"type": EgoSpan, "attrs": {"ego": "user"}},
                ],
            ),
            (
                "<|o_o|>Hello<|@_@|>World",
                [
                    {"type": EgoSpan, "attrs": {"ego": "user"}},
                    {"type": TextSpan, "attrs": {"text": "Hello"}},
                    {"type": EgoSpan, "attrs": {"ego": "assistant"}},
                    {"type": TextSpan, "attrs": {"text": "World"}},
                ],
            ),
            (
                "text <|o_o|> text",
                [
                    {"type": EgoSpan, "attrs": {"ego": "system"}},
                    {"type": TextSpan, "attrs": {"text": "text "}},
                    {"type": EgoSpan, "attrs": {"ego": "user"}},
                    {"type": TextSpan, "attrs": {"text": "text"}},
                ],
            ),
            # Should not have duplicate ego spans
            ("<|o_o|><|o_o|>", [{"type": EgoSpan, "attrs": {"ego": "user"}}]),
            ("<|+++|>", [{"type": ContextResetSpan, "attrs": {"train": True}}]),
            ("<|===|>", [{"type": ContextResetSpan, "attrs": {"train": False}}]),
            (
                "<|o_o|><|MyClass|>",
                [
                    {"type": EgoSpan, "attrs": {"ego": "user"}},
                    {"type": ClassSpan, "attrs": {"class_name": "MyClass"}},
                ],
            ),
            (
                "<|@_@|><|my_obj|>",
                [
                    {"type": EgoSpan, "attrs": {"ego": "assistant"}},
                    {"type": ObjSpan, "attrs": {"var_ids": ["my_obj"]}},
                ],
            ),
            (
                "<|@_@|><|obj1|obj2|>",
                [
                    {"type": EgoSpan, "attrs": {"ego": "assistant"}},
                    {"type": ObjSpan, "attrs": {"var_ids": ["obj1", "obj2"]}},
                ],
            ),
            (
                "<|@_@ goal=run|>",
                [
                    {"type": EgoSpan, "attrs": {"ego": "assistant"}},
                    {"type": SampleSpan, "attrs": {"goal": "run"}},
                ],
            ),
        ]
        for code, expected_details in test_cases:
            with self.subTest(code=code):
                spans = _load_and_print_holoware(code).spans
                self.assertEqual(len(spans), len(expected_details))
                for i, details in enumerate(expected_details):
                    span = spans[i]
                    self.assertIsInstance(span, details["type"])
                    if "attrs" in details:
                        for attr, value in details["attrs"].items():
                            self.assertEqual(getattr(span, attr), value)

    def test_parser_implicit_system_ego(self):
        ware = _load_and_print_holoware("some text")
        self.assertEqual(len(ware.spans), 2)
        self.assertIsInstance(ware.spans[0], EgoSpan)
        self.assertEqual(ware.spans[0].ego, "system")
        self.assertIsInstance(ware.spans[1], TextSpan)
        self.assertEqual(ware.spans[1].text, "some text")

    def test_parser_context_reset_clears_ego(self):
        code = "<|o_o|>hello<|+++|>world"
        ware = _load_and_print_holoware(code)
        self.assertEqual(len(ware.spans), 5)
        self.assertIsInstance(ware.spans[0], EgoSpan)
        self.assertEqual(ware.spans[0].ego, "user")
        self.assertIsInstance(ware.spans[2], ContextResetSpan)
        self.assertIsInstance(ware.spans[3], EgoSpan)
        self.assertEqual(ware.spans[3].ego, "system")
        self.assertIsInstance(ware.spans[4], TextSpan)
        self.assertEqual(ware.spans[4].text, "world")

    def test_parser_span_before_ego_raises_error(self):
        with self.assertRaises(ValueError):
            _load_and_print_holoware("<|MyClass|>")
        with self.assertRaises(ValueError):
            _load_and_print_holoware("<|my_obj|>")
        with self.assertRaises(ValueError):
            _load_and_print_holoware("<|goal=run|>")

    def test_parser_empty_tag(self):
        spans = _load_and_print_holoware("<||>").spans
        self.assertEqual(len(spans), 0)

    def test_parser_whitespace_tag(self):
        spans = _load_and_print_holoware("<| |>").spans
        self.assertEqual(len(spans), 0)

    def test_parser_handles_escaped_tag_in_text(self):
        code = "This is some text with a \\<|fake_tag|> in it."
        ware = _load_and_print_holoware(code)
        self.assertEqual(len(ware.spans), 2)
        self.assertIsInstance(ware.spans[0], EgoSpan)
        self.assertEqual(ware.spans[0].ego, "system")
        self.assertIsInstance(ware.spans[1], TextSpan)
        self.assertEqual(ware.spans[1].text, "This is some text with a <|fake_tag|> in it.")

    def test_parser_escaped_backslash_before_tag(self):
        code = "\\\\<|o_o|>"
        ware = _load_and_print_holoware(code)
        self.assertEqual(len(ware.spans), 3)
        self.assertIsInstance(ware.spans[1], TextSpan)
        self.assertEqual(ware.spans[1].text, "\\")
        self.assertIsInstance(ware.spans[2], EgoSpan)

    def test_parser_triple_backslash_escape(self):
        code = "\\\\\\<|o_o|>"
        ware = _load_and_print_holoware(code)
        self.assertEqual(len(ware.spans), 2)
        self.assertIsInstance(ware.spans[1], TextSpan)
        self.assertEqual(ware.spans[1].text, "\\<|o_o|>")

    def test_parser_unclosed_tag_raises_error(self):
        with self.assertRaises(ValueError) as e:
            _load_and_print_holoware("<|o_o")
        self.assertIn("Unclosed tag", str(e.exception))

    def test_parser_merges_consecutive_text_spans(self):
        code = "<|o_o|>one two three"
        ware = _load_and_print_holoware(code)
        self.assertEqual(len(ware.spans), 2)
        self.assertIsInstance(ware.spans[1], TextSpan)
        self.assertEqual(ware.spans[1].text, "one two three")

    def test_parser_handles_whitespace(self):
        code = "  <|o_o|>  \n  text  "
        ware = _load_and_print_holoware(code)
        self.assertEqual(len(ware.spans), 2)
        self.assertIsInstance(ware.spans[0], EgoSpan)
        self.assertIsInstance(ware.spans[1], TextSpan)
        self.assertEqual(ware.spans[1].text, "text  ")

    def test_parser_indented_block_simple(self):
        code = """<|o_o|>
<|MyClass|>
    Some indented text.
"""
        ware = _load_and_print_holoware(code)
        self.assertEqual(len(ware.spans), 2)
        class_span = ware.spans[1]
        self.assertIsInstance(class_span, ClassSpan)
        self.assertEqual(class_span.class_name, "MyClass")
        self.assertIsNotNone(class_span.body)

        body = class_span.body
        self.assertIsNotNone(body)
        self.assertEqual(len(body.spans), 2)
        self.assertIsInstance(body.spans[0], EgoSpan)
        self.assertEqual(body.spans[0].ego, "system")
        self.assertIsInstance(body.spans[1], TextSpan)
        self.assertEqual(body.spans[1].text.strip(), "Some indented text.")

    def test_parser_indented_block_complex(self):
        code = """<|o_o|>
<|Container|>
    <|@_@|>
    <|Item|>
        Nested item
"""
        ware = _load_and_print_holoware(code)
        container_span = ware.spans[1]
        self.assertIsInstance(container_span, ClassSpan)
        self.assertEqual(container_span.class_name, "Container")

        body = container_span.body
        self.assertIsNotNone(body)
        self.assertEqual(len(body.spans), 2)
        self.assertIsInstance(body.spans[0], EgoSpan)
        self.assertEqual(body.spans[0].ego, "assistant")

        item_span = body.spans[1]
        self.assertIsInstance(item_span, ClassSpan)
        self.assertEqual(item_span.class_name, "Item")

        nested_body = item_span.body
        self.assertIsNotNone(nested_body)
        self.assertEqual(len(nested_body.spans), 2)
        self.assertIsInstance(nested_body.spans[1], TextSpan)
        self.assertEqual(nested_body.spans[1].text.strip(), "Nested item")

    def test_parser_indented_block_no_block(self):
        code = "<|o_o|><|MyClass|>\nNot indented."
        ware = _load_and_print_holoware(code)
        self.assertEqual(len(ware.spans), 3)
        class_span = ware.spans[1]
        self.assertIsInstance(class_span, ClassSpan)
        self.assertIsNone(class_span.body)
        text_span = ware.spans[2]
        self.assertIsInstance(text_span, TextSpan)
        self.assertEqual(text_span.text, "Not indented.")

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

class CompressorHolowareTest(TestCase):
    def test_parser_compressor_holoware(self):
        ware = _load_and_print_holoware(COMPRESSOR_HOL)
        spans = ware.spans

        context_resets = [s for s in spans if isinstance(s, ContextResetSpan)]
        self.assertEqual(len(context_resets), 3)
        self.assertTrue(context_resets[0].train)
        self.assertTrue(context_resets[1].train)
        self.assertFalse(context_resets[2].train)

        class_spans = [s for s in spans if isinstance(s, ClassSpan)]
        self.assertEqual(len(class_spans), 4)
        class_names = [s.class_name for s in class_spans]
        self.assertEqual(class_names, ["BingoAttractor", "BingoAttractor", "FidelityCritique", "FidelityAttractor"])

        self.assertIsNone(class_spans[2].body)
        self.assertEqual(class_spans[3].kargs, ["original", "decompressed"])

        sample_spans = [s for s in spans if isinstance(s, SampleSpan)]
        self.assertEqual(len(sample_spans), 4)
        goals = {s.goal for s in sample_spans}
        self.assertEqual(goals, {"compress", "decompress", "think", "json"})

        obj_spans = [s for s in spans if isinstance(s, ObjSpan)]
        self.assertEqual(len(obj_spans), 4)
        var_id_list = [s.var_ids for s in obj_spans]
        self.assertEqual(var_id_list[0], ["text", "original", "input", "data"])
        self.assertEqual(var_id_list[1], ["compressed"])
        self.assertEqual(var_id_list[2], ["original"])
        self.assertEqual(var_id_list[3], ["decompressed"])