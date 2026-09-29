from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
import token_permutation_demo as demo  # noqa: E402


PERMUTATION = [1, 2, 3, 0]
INVERSE = [3, 0, 1, 2]
SOURCE_FINGERPRINT = "00" * 32
NONCE = "11" * 32


def make_secret() -> dict[str, object]:
    secret: dict[str, object] = {
        "version": 1,
        "method": {"id": "aloepri-token", "version": "0.1"},
        "secret_id": "",
        "vocab_size": 4,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "binding_nonce": NONCE,
        "token_permutation": PERMUTATION.copy(),
        "inverse_token_permutation": INVERSE.copy(),
    }
    secret["secret_id"] = demo.commitment(
        secret, bytes.fromhex(SOURCE_FINGERPRINT), bytes.fromhex(NONCE)
    )
    return secret


class FixedModel:
    def __init__(self, logits: torch.Tensor, eos_token_id: int | None) -> None:
        self.logits = logits
        self.config = SimpleNamespace(eos_token_id=eos_token_id)

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False):
        del attention_mask, use_cache
        return SimpleNamespace(
            logits=self.logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        )


class TokenPermutationDemoTests(unittest.TestCase):
    def test_secret_rejects_duplicate_corrupt_and_vocab_mismatch_inputs(self) -> None:
        valid = make_secret()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secret.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(demo.load_secret(path)["secret_id"], valid["secret_id"])

            text = path.read_text(encoding="utf-8")
            duplicate = text.replace(
                '"vocab_size": 4', '"vocab_size": 999, "vocab_size": 4', 1
            ).replace(
                '"id": "aloepri-token"',
                '"id": "wrong", "id": "aloepri-token"',
                1,
            )
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaises(ValueError):
                demo.load_secret(path)

            corrupt = make_secret()
            corrupt["token_permutation"] = [0, 2, 3, 1]
            path.write_text(json.dumps(corrupt), encoding="utf-8")
            with self.assertRaises(ValueError):
                demo.load_secret(path)

            vocab_mismatch = make_secret()
            vocab_mismatch["vocab_size"] = 5
            path.write_text(json.dumps(vocab_mismatch), encoding="utf-8")
            with self.assertRaises(ValueError):
                demo.load_secret(path)

        with self.assertRaises(ValueError):
            demo.validate_artifact_binding(
                valid,
                {"vocab_size": 5},
                {"vocab_size": 5},
                {},
            )

    def test_tied_and_untied_tiny_llama_logits_and_greedy_are_equivalent(self) -> None:
        torch.set_num_threads(1)
        permutation = torch.tensor(PERMUTATION, dtype=torch.long)
        inverse = torch.tensor(INVERSE, dtype=torch.long)
        input_ids = torch.tensor([[0, 1]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        for tied in (True, False):
            with self.subTest(tied=tied):
                config = LlamaConfig(
                    vocab_size=4,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=16,
                    tie_word_embeddings=tied,
                    bos_token_id=0,
                    eos_token_id=3,
                    pad_token_id=0,
                )
                torch.manual_seed(1234)
                baseline = LlamaForCausalLM(config).eval()
                obfuscated = LlamaForCausalLM(config).eval()
                obfuscated.load_state_dict(baseline.state_dict())
                with torch.no_grad():
                    embedding = baseline.get_input_embeddings().weight.detach().clone()
                    obfuscated.get_input_embeddings().weight.copy_(
                        embedding.index_select(0, inverse)
                    )
                    if not tied:
                        head = baseline.get_output_embeddings().weight.detach().clone()
                        obfuscated.get_output_embeddings().weight.copy_(
                            head.index_select(0, inverse)
                        )
                if tied:
                    self.assertEqual(
                        obfuscated.get_input_embeddings().weight.data_ptr(),
                        obfuscated.get_output_embeddings().weight.data_ptr(),
                    )

                obfuscated_input = permutation.index_select(
                    0, input_ids.reshape(-1)
                ).reshape(input_ids.shape)
                with torch.inference_mode():
                    baseline_logits = demo.logits_for(
                        baseline, input_ids, attention_mask
                    )
                    obfuscated_logits = demo.logits_for(
                        obfuscated, obfuscated_input, attention_mask
                    )
                aligned_logits = obfuscated_logits.index_select(-1, permutation)
                torch.testing.assert_close(
                    baseline_logits, aligned_logits, rtol=1e-4, atol=1e-5
                )

                baseline_tokens = demo.greedy(
                    baseline,
                    input_ids,
                    attention_mask,
                    PERMUTATION,
                    2,
                    False,
                )
                obfuscated_tokens = demo.greedy(
                    obfuscated,
                    obfuscated_input,
                    attention_mask,
                    PERMUTATION,
                    2,
                    True,
                )
                self.assertTrue(torch.equal(baseline_tokens, obfuscated_tokens))

    def test_greedy_uses_original_axis_tie_break_and_eos(self) -> None:
        permutation = torch.tensor(PERMUTATION, dtype=torch.long)
        inverse = torch.tensor(INVERSE, dtype=torch.long)
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        tie_logits = torch.tensor([[[2.0, 2.0, 1.0, 0.0]]])
        baseline = FixedModel(tie_logits, None)
        obfuscated = FixedModel(tie_logits.index_select(-1, inverse), None)
        baseline_tokens = demo.greedy(
            baseline, input_ids, attention_mask, PERMUTATION, 1, False
        )
        obfuscated_input = permutation.index_select(0, input_ids.reshape(-1)).reshape(
            input_ids.shape
        )
        obfuscated_tokens = demo.greedy(
            obfuscated, obfuscated_input, attention_mask, PERMUTATION, 1, True
        )
        self.assertEqual(int(baseline_tokens[0, -1]), 0)
        self.assertTrue(torch.equal(baseline_tokens, obfuscated_tokens))

        eos_logits = torch.tensor([[[0.0, 1.0, 5.0, 0.0]]])
        eos_baseline = FixedModel(eos_logits, 2)
        eos_obfuscated = FixedModel(eos_logits.index_select(-1, inverse), 2)
        eos_baseline_tokens = demo.greedy(
            eos_baseline, input_ids, attention_mask, PERMUTATION, 5, False
        )
        eos_obfuscated_tokens = demo.greedy(
            eos_obfuscated,
            obfuscated_input,
            attention_mask,
            PERMUTATION,
            5,
            True,
        )
        self.assertEqual(eos_baseline_tokens.shape[1], 2)
        self.assertTrue(torch.equal(eos_baseline_tokens, eos_obfuscated_tokens))


if __name__ == "__main__":
    unittest.main()
