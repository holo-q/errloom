# -*- coding: utf-8 -*-
# Tests for Loom sampling logic and optional vLLM integration (server must be run separately).
import logging
import socket
import unittest
from typing import Any, cast, Dict
from unittest.mock import patch

from errloom.interop.vllm_client import VLLMClient
from errloom.loom import Loom
from errloom.tapestry import Rollout
from tests.base import ErrloomTest

try:
    from datasets import Dataset  # type: ignore
except Exception:
    Dataset = None  # type: ignore

logger = logging.getLogger(__name__)

HOST = "localhost"
PORT = 8000

def _is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


class _MinimalLoom(Loom):
    def rollout(self, roll: Rollout) -> Rollout:  # errloom.loom.Loom.rollout()
        # Example rollout that requests a sample and writes it into the rollout
        # No stop_sequences here; unit tests call sample explicitly as needed
        _ = self.sample(rollout=roll, sanitize_sampling_args=True, allow_partial_on_length=True)
        return roll


class LoomSamplingUnitTest(ErrloomTest):
    def setUp(self) -> None:
        super().setUp()
        # Provide a small HF Dataset for Loom init_data validation
        items = [{"id": 1, "question": "Q", "answer": "A"}]
        if Dataset is not None:
            self.data = Dataset.from_list(items)  # type: ignore
        else:
            self.data = items  # fallback; load_data may accept list in future

    def _build_rollout(self) -> Rollout:
        sampling_args: Dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": 8,
            "extra_body": {"skip_special_tokens": False, "spaces_between_special_tokens": False},
        }
        r = Rollout(row={"id": "test_row"}, sampling_args=sampling_args)
        r.new_context()
        r.add_frozen(ego="user", content="Say hi")
        return r

    def test_sample_appends_message_and_respects_sanitize(self):
        # Build loom with dry=True to avoid model/tokenizer/trainer initialization; use MockClient by not passing client
        with patch("errloom.utils.data_utils.load_data", side_effect=lambda x: x):
            loom = _MinimalLoom(
                model=None,
                tokenizer=None,
                client=None,  # triggers MockClient
                message_type="chat",
                data=self.data,        # type: ignore[arg-type]
                data_train=self.data,  # type: ignore[arg-type]
                data_bench=self.data,  # type: ignore[arg-type]
                data_split=None,
                dry=True,
                max_concurrent=1,
            )

        roll = self._build_rollout()
        # Non-streaming path: do not pass stop_sequences
        out = loom.sample(rollout=roll, sanitize_sampling_args=True, allow_partial_on_length=True)

        self.assertIsInstance(out, str)
        self.assertTrue(len(roll.samples) >= 1)
        last = roll.samples[-1]
        self.assertIsInstance(last, dict)
        self.assertEqual(last.get("role"), "assistant")
        self.assertIsInstance(last.get("content"), str)
        # Print generated text preview for human verification
        gen = last.get("content") or ""
        preview = gen[:200].replace("\n", "\\n")
        logger.info(f"sample_appends -> {preview}")

    def test_streaming_path_triggered_with_stop_sequences(self):
        with patch("errloom.utils.data_utils.load_data", side_effect=lambda x: x):
            loom = _MinimalLoom(
                model=None,
                tokenizer=None,
                client=None,  # MockClient
                message_type="chat",
                data=self.data,        # type: ignore[arg-type]
                data_train=self.data,  # type: ignore[arg-type]
                data_bench=self.data,  # type: ignore[arg-type]
                data_split=None,
                dry=True,
                max_concurrent=1,
            )

        roll = self._build_rollout()
        # Explicit streaming path: pass stop_sequences directly
        out = loom.sample(rollout=roll, sanitize_sampling_args=True, stop_sequences=["</stop>"], allow_partial_on_length=True)

        self.assertIsInstance(out, str)
        self.assertTrue(len(roll.samples) >= 1)
        self.assertEqual(roll.samples[-1].get("role"), "assistant")
        self.assertIsInstance(roll.samples[-1].get("content"), str)
        # Print generated text preview for human verification
        gen = roll.samples[-1].get("content") or ""
        preview = gen[:200].replace("\n", "\\n")
        logger.info(f"streaming_with_stops -> {preview}")

class LoomClientTest(ErrloomTest):
    def setUp(self):
        super().setUp()
        self.client = VLLMClient(host=HOST, port=PORT, connection_timeout=1.0)

    def tearDown(self):
        super().tearDown()
        self.client.close()

class LoomVLLMIntegrationTest(LoomClientTest):
    """
    Full VLLMClient inference test, skipped unless localhost:8000 is reachable.
    Does not attempt to boot or manage the server; assumes user runs it separately.
    """


    @unittest.skipUnless(_is_port_open(HOST, PORT), "vLLM server not reachable at localhost:8000; skipping integration test")
    def test_vllm_client_inference_chat(self):
        # Import here to avoid hard dependency when server isn't used

        # Build a Loom that uses VLLMClient; dry=True to avoid trainer/model init
        data = [{"id": 1, "question": "Who are you?", "answer": ""}]
        if Dataset is not None:
            data_ds = Dataset.from_list(data)  # type: ignore
        else:
            data_ds = data
        with patch("errloom.utils.data_utils.load_data", side_effect=lambda x: x):
            loom = _MinimalLoom(
                model="Qwen/Qwen3-4B",
                tokenizer="Qwen/Qwen3-4B",
                client=cast(Any, self.client),  # type: ignore[arg-type]
                client_args={"temperature": 0.0, "max_tokens": 32},
                message_type="chat",
                data=data_ds,        # type: ignore[arg-type]
                data_train=data_ds,  # type: ignore[arg-type]
                data_bench=data_ds,  # type: ignore[arg-type]
                data_split=None,
                dry=True,
                max_concurrent=1,
            )

        # Prepare a rollout with a system+user chat, request a small completion
        r = Rollout(row={"id": "vllm_row"}, sampling_args={"temperature": 0.0, "max_tokens": 32})
        r.new_context()
        r.add_frozen(ego=None, content="You are a helpful assistant.")
        r.add_frozen(ego="user", content="Reply with the single word: pong")

        text = loom.sample(rollout=r, sanitize_sampling_args=True, allow_partial_on_length=True)
        self.assertIsInstance(text, str)
        self.assertGreaterEqual(len(r.samples), 1)
        self.assertEqual(r.samples[-1]["role"], "assistant")
        self.assertIsInstance(r.samples[-1]["content"], str)
        self.assertTrue(len(r.samples[-1]["content"]) > 0)
        # Print generated text preview for human verification
        preview = r.samples[-1]["content"][:200].replace("\n", "\\n")
        logger.info(f"vllm_chat -> {preview}")

    @unittest.skipUnless(_is_port_open(HOST, PORT), "vLLM server not reachable at localhost:8000; skipping integration test")
    def test_vllm_client_inference_completion(self):
        data = [{"id": 2, "prompt": "Echo: ping ->", "answer": ""}]
        if Dataset is not None:
            data_ds2 = Dataset.from_list(data)  # type: ignore
        else:
            data_ds2 = data
        with patch("errloom.utils.data_utils.load_data", side_effect=lambda x: x):
            loom = _MinimalLoom(
                model="Qwen/Qwen3-4B",
                tokenizer="Qwen/Qwen3-4B",
                client=cast(Any, self.client),  # type: ignore[arg-type]
                client_args={"temperature": 0.0, "max_tokens": 32},
                message_type="completion",
                data=data_ds2,        # type: ignore[arg-type]
                data_train=data_ds2,  # type: ignore[arg-type]
                data_bench=data_ds2,  # type: ignore[arg-type]
                data_split=None,
                dry=True,
                max_concurrent=1,
            )

        r = Rollout(row={"id": "vllm_row2"}, sampling_args={"temperature": 0.0, "max_tokens": 32})
        r.new_context()
        # In completion mode, Loom.sample will call rollout.to_text(); ensure we build a proper text prefix
        r.add_frozen(ego=None, content="")
        r.add_frozen(ego="user", content="Echo: ping ->")
        r.add_frozen(ego="assistant", content="")

        logger.info(f"prefix: {r.to_text()}")
        text = loom.sample(rollout=r, sanitize_sampling_args=True, allow_partial_on_length=True)
        self.assertIsInstance(text, str)
        self.assertGreaterEqual(len(r.samples), 1)
        self.assertEqual(r.samples[-1]["role"], "assistant")
        self.assertIsInstance(r.samples[-1]["content"], str)
        self.assertTrue(len(r.samples[-1]["content"]) > 0)

        # Print generated text preview for human verification
        preview = r.samples[-1]["content"][:200].replace("\n", "\\n")
        logger.info(f"vllm_completion -> {preview}")