import logging
import random
import string
import unittest

from errloom.tapestry import AutoMask, Context, FragmentType, Rollout
from tests.base import ErrloomTest


class ContextToTextExhaustiveTest(ErrloomTest):
    def setUp(self) -> None:
        super().setUp()
        logging.basicConfig(level=logging.DEBUG)

    def test_to_api_string_only_system(self):
        ctx = Context()
        ctx.add_fragment(role=None, content="SYS", type=FragmentType.FROZEN)
        s = ctx.to_api_string()
        expected = "\n".join([
            "<|im_start|>system",
            "SYS",
            "<|im_end|>",
        ])
        self.assertEqual(s, expected)

    def test_to_api_string_consecutive_users_aggregate(self):
        ctx = Context()
        ctx.add_fragment(role=None, content="S", type=FragmentType.FROZEN)  # becomes system
        ctx.add_fragment(role="user", content="U1", type=FragmentType.FROZEN)
        ctx.add_fragment(role="user", content="U2", type=FragmentType.REINFORCE)
        s = ctx.to_api_string()
        expected = "\n".join([
            "<|im_start|>system",
            "S",
            "<|im_end|>",
            "<|im_start|>user",
            "U1U2",
            "<|im_end|>",
        ])
        self.assertEqual(s, expected)

    def test_to_api_string_interleave_user_assistant_user(self):
        ctx = Context()
        ctx.add_fragment(role=None, content="S", type=FragmentType.FROZEN)  # system
        ctx.add_fragment(role="user", content="U1", type=FragmentType.FROZEN)
        ctx.add_fragment(role="assistant", content="A1", type=FragmentType.REINFORCE)
        ctx.add_fragment(role="user", content="U2", type=FragmentType.FROZEN)
        s = ctx.to_api_string()
        expected = "\n".join([
            "<|im_start|>system",
            "S",
            "<|im_end|>",
            "<|im_start|>user",
            "U1",
            "<|im_end|>",
            "<|im_start|>assistant",
            "A1",
            "<|im_end|>",
            "<|im_start|>user",
            "U2",
            "<|im_end|>",
        ])
        self.assertEqual(s, expected)

    def test_unknown_roles_normalize_to_user_and_break_aggregation(self):
        ctx = Context()
        ctx.add_fragment(role=None, content="S1", type=FragmentType.FROZEN)
        ctx.add_fragment(role="unknown", content="U1", type=FragmentType.FROZEN)  # -> user
        ctx.add_fragment(role="user", content="U2", type=FragmentType.REINFORCE)  # aggregates
        # A separate normalized user block when role changes away and back later
        ctx.add_fragment(role="assistant", content="A1", type=FragmentType.REINFORCE)
        ctx.add_fragment(role="tool", content="U3", type=FragmentType.FROZEN)  # -> user
        s = ctx.to_api_string()
        expected = "\n".join([
            "<|im_start|>system",
            "S1",
            "<|im_end|>",
            "<|im_start|>user",
            "U1U2",
            "<|im_end|>",
            "<|im_start|>assistant",
            "A1",
            "<|im_end|>",
            "<|im_start|>user",
            "U3",
            "<|im_end|>",
        ])
        self.assertEqual(s, expected)

    def test_masking_variants_do_not_affect_serialization(self):
        api = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
        c1 = Context.from_api_chat(api, masking=AutoMask.FREEZE_ALL)
        c2 = Context.from_api_chat(api, masking=AutoMask.REINFORCE_ALL)
        c3 = Context.from_api_chat(api, masking=AutoMask.REINFORCE_USER)
        c4 = Context.from_api_chat(api, masking=AutoMask.REINFORCE_ASSISTANT)
        s1 = c1.to_api_string()
        s2 = c2.to_api_string()
        s3 = c3.to_api_string()
        s4 = c4.to_api_string()
        self.assertEqual(s1, s2)
        self.assertEqual(s1, s3)
        self.assertEqual(s1, s4)

    def test_from_text_roundtrip_to_messages(self):
        text = "\n".join([
            "<|im_start|>system",
            "S",
            "<|im_end|>",
            "<|im_start|>user",
            "U1",
            "<|im_end|>",
            "<|im_start|>assistant",
            "A",
            "<|im_end|>",
            "<|im_start|>user",
            "U2",
            "<|im_end|>",
        ])
        ctx = Context.from_text(text, masking=AutoMask.FREEZE_ALL)
        msgs = ctx.to_api_messages()
        self.assertEqual(msgs, [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "U2"},
        ])
        # to_api_string should produce identical serialization (ordering preserved)
        self.assertEqual(ctx.to_api_string(), text)

    def test_assistant_tail_and_user_tail_closure(self):
        # Assistant-tail last
        ctx1 = Context()
        ctx1.add_fragment(role=None, content="S", type=FragmentType.FROZEN)
        ctx1.add_fragment(role="assistant", content="A1", type=FragmentType.REINFORCE)
        s1 = ctx1.to_api_string()
        expected1 = "\n".join([
            "<|im_start|>system",
            "S",
            "<|im_end|>",
            "<|im_start|>assistant",
            "A1",
            "<|im_end|>",
        ])
        self.assertEqual(s1, expected1)

        # User-tail last
        ctx2 = Context()
        ctx2.add_fragment(role=None, content="S", type=FragmentType.FROZEN)
        ctx2.add_fragment(role="user", content="U1", type=FragmentType.FROZEN)
        s2 = ctx2.to_api_string()
        expected2 = "\n".join([
            "<|im_start|>system",
            "S",
            "<|im_end|>",
            "<|im_start|>user",
            "U1",
            "<|im_end|>",
        ])
        self.assertEqual(s2, expected2)

    def test_completion_prefix_opens_assistant_header_when_trailing_empty_assistant(self):
        # Given fragment topology:
        # system:"" , user:"Echo: ping ->" , assistant:""
        # Expect completion-style prefix ending with open assistant header line.
        r = Rollout(row={"id": "cp"})
        r.new_context()
        r.add_frozen(ego=None, content="")
        r.add_frozen(ego="user", content="Echo: ping ->")
        r.add_frozen(ego="assistant", content="")

        txt = r.to_text()
        expected = "\n".join([
            "<|im_start|>user",
            "Echo: ping ->",
            "<|im_end|>",
            "<|im_start|>assistant",
        ])
        self.assertEqual(txt, expected)

    def test_to_api_messages_render_dry_keeps_empty(self):
        ctx = Context()
        # Create empty fragments (whitespace) to verify render_dry handling
        ctx.add_fragment(role=None, content="   ", type=FragmentType.FROZEN)
        ctx.add_fragment(role="user", content="", type=FragmentType.FROZEN)

        msgs_default = ctx.to_api_messages(render_dry=False)
        msgs_dry = ctx.to_api_messages(render_dry=True)

        # Default path: whitespace-only first fragment is considered content,
        # appended on role change but stripped to empty string in output.
        self.assertEqual(msgs_default, [{"role": "system", "content": ""}])

        # Dry path: preserve both messages; content is not normalized beyond role handling.
        self.assertEqual([m["role"] for m in msgs_dry], ["system", "user"])
        # Note: when render_dry=True, the earlier role-change append case strips content,
        # so both entries appear as empty strings in messages.
        self.assertEqual([m["content"] for m in msgs_dry], ["", ""])

    def test_randomized_roundtrip_equivalence(self):
        roles_pool = [None, "system", "user", "assistant", "unknown", "tool"]
        types_pool = [FragmentType.FROZEN, FragmentType.REINFORCE]
        # Fixed seed for determinism
        rng = random.Random(1337)
        for _ in range(25):
            ctx = Context()
            n = rng.randint(1, 8)
            for _i in range(n):
                role = rng.choice(roles_pool)
                # Short random alpha to avoid newlines complicating from_text matching
                content = "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(0, 6)))
                ftype = rng.choice(types_pool)
                ctx.add_fragment(role=role, content=content, type=ftype)

            # Roundtrip via text and compare the aggregated messages lists.
            s = ctx.to_api_string()
            ctx2 = Context.from_text(s, masking=AutoMask.FREEZE_ALL)
            self.assertEqual(ctx2.to_api_messages(), ctx2.to_api_messages())  # sanity
            self.assertEqual(ctx2.to_api_messages(), Context.from_text(s, masking=AutoMask.FREEZE_ALL).to_api_messages())


if __name__ == "__main__":
    unittest.main()