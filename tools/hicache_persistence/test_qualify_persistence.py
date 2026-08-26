import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("qualify_persistence.py")
SPEC = importlib.util.spec_from_file_location("qualify_persistence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PersistenceHarnessTest(unittest.TestCase):
    def test_prompt_cases_are_deterministic_and_contain_all_needles(self):
        config = {
            "served_model": "test-model",
            "prompt_identity": "representation-a",
            "prompt_target_chars": 10000,
            "concurrency": 2,
        }
        first = MODULE.build_cases(config)
        second = MODULE.build_cases(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertNotEqual(first[0].needles, first[1].needles)
        for case in first:
            prompt = case.payload["messages"][0]["content"]
            for needle in case.needles:
                self.assertIn(needle, prompt)
            self.assertTrue(case.payload["stream"])
            self.assertTrue(case.payload["stream_options"]["include_usage"])

    def test_metric_selection_requires_all_labels(self):
        text = "\n".join(
            [
                'sglang:prefill_effective_tokens_total{mode="storage_hit",tp_rank="0"} 64',
                'sglang:prefill_effective_tokens_total{mode="storage_hit",tp_rank="1"} 32',
                'sglang:prefill_effective_tokens_total{mode="host_hit",tp_rank="0"} 128',
            ]
        )
        values = MODULE.parse_metrics(text)
        result = MODULE.metric_value(
            values,
            {
                "name": "sglang:prefill_effective_tokens_total",
                "labels": {"mode": "storage_hit"},
            },
        )
        self.assertEqual(result, 96)

    def test_namespace_snapshot_detects_file_mutation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "bucket").mkdir()
            cache_file = root / "bucket" / "page"
            cache_file.write_bytes(b"one")
            before = MODULE.namespace_snapshot(root)
            cache_file.write_bytes(b"two-two")
            after = MODULE.namespace_snapshot(root)
            self.assertNotEqual(before["listing_sha256"], after["listing_sha256"])
            self.assertNotEqual(before["total_bytes"], after["total_bytes"])


if __name__ == "__main__":
    unittest.main()
