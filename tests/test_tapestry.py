import logging

from errloom.tapestry import (AutoMask, Context, FragmentType, Rollout, Tapestry)
from errloom.aliases import APIChat
from tests.base import ErrloomTest


class ContextToApiMessagesTest(ErrloomTest):
    def setUp(self) -> None:
        super().setUp()
        logging.basicConfig(level=logging.DEBUG)

    def test_role_normalization_and_aggregation(self):
        ctx = Context()
        # First None role becomes system
        ctx.add_fragment(role=None, content="sys1", type=FragmentType.FROZEN)
        # Unknown role normalizes to user (and breaks aggregation from system)
        ctx.add_fragment(role="unknown", content="u1", type=FragmentType.FROZEN)
        # Consecutive user fragments aggregate
        ctx.add_fragment(role="user", content="u2", type=FragmentType.REINFORCE)
        # Assistant message
        ctx.add_fragment(role="assistant", content="a1", type=FragmentType.REINFORCE)
        # Another unknown normalizes to user, new block
        ctx.add_fragment(role="weird", content="u3", type=FragmentType.FROZEN)

        msgs = ctx.to_api_messages()
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[0], {"role": "system", "content": "sys1"})
        self.assertEqual(msgs[1], {"role": "user", "content": "u1u2"})
        self.assertEqual(msgs[2], {"role": "assistant", "content": "a1"})
        self.assertEqual(msgs[3], {"role": "user", "content": "u3"})

    def test_to_api_string_formatting(self):
        ctx = Context()
        ctx.add_fragment(role=None, content="S", type=FragmentType.FROZEN)  # system
        ctx.add_fragment(role="user", content="U", type=FragmentType.FROZEN)
        ctx.add_fragment(role="assistant", content="A", type=FragmentType.REINFORCE)
        s = ctx.to_api_string()
        expected = "\n".join([
            "<|im_start|>system",
            "S",
            "<|im_end|>",
            "<|im_start|>user",
            "U",
            "<|im_end|>",
            "<|im_start|>assistant",
            "A",
            "<|im_end|>",
        ])
        self.assertEqual(s, expected)

    def test_from_text_parsing_and_roundtrip(self):
        text = "\n".join([
            "<|im_start|>system",
            "S",
            "<|im_end|>",
            "<|im_start|>user",
            "U",
            "<|im_end|>",
            "<|im_start|>assistant",
            "A",
            "<|im_end|>",
        ])
        ctx = Context.from_text(text, masking=AutoMask.FREEZE_ALL)
        msgs = ctx.to_api_messages()
        self.assertEqual(msgs, [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ])

    def test_from_api_chat_automask(self):
        api: APIChat = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
        # Freeze all
        c1 = Context.from_api_chat(api, masking=AutoMask.FREEZE_ALL)
        self.assertTrue(all(f.type == FragmentType.FROZEN for f in c1.fragments))
        # Reinforce all
        c2 = Context.from_api_chat(api, masking=AutoMask.REINFORCE_ALL)
        self.assertTrue(all(f.type == FragmentType.REINFORCE for f in c2.fragments))
        # Reinforce user only
        c3 = Context.from_api_chat(api, masking=AutoMask.REINFORCE_USER)
        types3 = [f.type for f in c3.fragments]
        self.assertEqual(types3, [FragmentType.FROZEN, FragmentType.REINFORCE, FragmentType.FROZEN])
        # Reinforce assistant only
        c4 = Context.from_api_chat(api, masking=AutoMask.REINFORCE_ASSISTANT)
        types4 = [f.type for f in c4.fragments]
        self.assertEqual(types4, [FragmentType.FROZEN, FragmentType.FROZEN, FragmentType.REINFORCE])

    def test_extract_fence_with_and_without_tag(self):
        ctx = Context()
        ctx.add_fragment(role=None, content="Intro", type=FragmentType.FROZEN)  # system
        ctx.add_fragment(role="assistant", content="<compress>XYZ</compress>", type=FragmentType.REINFORCE)
        ctx.add_fragment(role="user", content="Tail", type=FragmentType.FROZEN)

        # Without wrapper_tag returns last message for role
        self.assertEqual(ctx.extract_fence(None, role="assistant"), "<compress>XYZ</compress>")
        # With tag extracts inner content
        self.assertEqual(ctx.extract_fence("compress", role="assistant"), "XYZ")
        # Non-matching role returns None
        self.assertIsNone(ctx.extract_fence("compress", role="user"))
        # Unknown tag returns None
        self.assertIsNone(ctx.extract_fence("decompress", role="assistant"))

    def test_extract_mdjson_fenced_and_inline(self):
        # Fenced
        fenced_json = """```json
{"a": 1, "b": {"c": 2}}
```"""
        ctx1 = Context()
        ctx1.add_fragment(role="user", content="pre", type=FragmentType.FROZEN)
        ctx1.add_fragment(role="assistant", content=fenced_json, type=FragmentType.REINFORCE)
        out1 = ctx1.extract_mdjson(role="assistant")
        self.assertEqual(out1, '{"a": 1, "b": {"c": 2}}')

        # Inline
        inline = "Some text\n{" + '"x": 3, "y": {"z": 4}' + "}\nend"
        ctx2 = Context()
        ctx2.add_fragment(role="assistant", content=inline, type=FragmentType.REINFORCE)
        out2 = ctx2.extract_mdjson(role="assistant")
        self.assertEqual(out2, '{"x": 3, "y": {"z": 4}}')

        # None when no assistant content
        ctx3 = Context()
        ctx3.add_fragment(role="user", content="no json here", type=FragmentType.FROZEN)
        self.assertIsNone(ctx3.extract_mdjson(role="assistant"))


class RolloutWrapperTest(ErrloomTest):
    def test_rollout_context_management(self):
        r = Rollout(row={"id": 1})
        r.new_context()
        r.add_frozen(ego=None, content="S")
        r.add_reinforced(ego="assistant", content="A1")
        r.add_reinforced(ego="assistant", content="A2")
        msgs = r.to_api_chat()
        # System then aggregated assistant
        self.assertEqual(msgs, [
            {"role": "system", "content": "S"},
            {"role": "assistant", "content": "A1A2"},
        ])
        txt = r.to_text()
        expected = "\n".join([
            "<|im_start|>system",
            "S",
            "<|im_end|>",
            "<|im_start|>assistant",
            "A1A2",
            "<|im_end|>",
        ])
        self.assertEqual(txt, expected)

    def test_rollout_extract_wrappers(self):
        r = Rollout(row={"id": 2})
        r.new_context()
        r.add_reinforced(ego="assistant", content="<think>plan</think> prefix <json>{}</json>")
        # extract_fence last matching
        self.assertEqual(r.extract_fence("think", role="assistant"), "plan")
        # extract_json proxies to context.extract_mdjson
        self.assertEqual(r.extract_json(role="assistant"), "{}")


class TapestryDatasetTest(ErrloomTest):
    def test_extract_rollouts_to_dataset_prompt_and_completion(self):
        # Build two rollouts
        r1 = Rollout(row={"id": "r1"}, gravity=0.5)
        r1.new_context()
        r1.add_frozen(ego=None, content="sys")
        r1.add_frozen(ego="user", content="u1")
        r1.add_reinforced(ego="assistant", content="a1")
        r1.add_frozen(ego="user", content="u2")  # after assistant

        r2 = Rollout(row={"id": "r2"}, gravity=1.0)
        r2.new_context()
        r2.add_frozen(ego="user", content="hello")
        # no assistant message -> completion should be empty

        tap = Tapestry(rollouts=[r1, r2], task="default")

        data = tap.extract_rollouts_to_dataset()
        self.assertEqual(list(data.keys()), ["prompt", "completion", "answer", "gravity"])

        # For r1: prompt includes all non-assistant as "role: content" with blank lines
        self.assertEqual(data["prompt"][0], "system: sys\n\nuser: u1\n\nuser: u2")
        # completion/answer are last assistant content
        self.assertEqual(data["completion"][0], "a1")
        self.assertEqual(data["answer"][0], "a1")
        self.assertAlmostEqual(data["gravity"][0], 0.5)

        # For r2: only user prompt, no assistant -> empty completion/answer
        self.assertEqual(data["prompt"][1], "user: hello")
        self.assertEqual(data["completion"][1], "")
        self.assertEqual(data["answer"][1], "")
        self.assertAlmostEqual(data["gravity"][1], 1.0)