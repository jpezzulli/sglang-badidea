import threading
import time
import unittest
from queue import Queue

import torch

from sglang.srt.managers.cache_controller import StorageOperation
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiCacheBackupWorkerFailures(CustomTestCase):
    def test_hybrid_worker_preserves_progress_and_continues_after_exception(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.storage_stop_event = threading.Event()
        controller.backup_queue = Queue()
        controller.ack_backup_queue = Queue()
        calls = []

        first = StorageOperation(torch.tensor([0]), [1], hash_value=["page0"])
        second = StorageOperation(torch.tensor([1]), [2], hash_value=["page1"])

        def injected_backup(operation):
            calls.append(operation.id)
            operation.completed_tokens = 64
            if operation is first:
                raise RuntimeError("injected Mooncake failure after one batch")

        controller._page_backup = injected_backup
        worker = threading.Thread(target=controller.backup_thread_func, daemon=True)
        worker.start()
        controller.backup_queue.put(first)
        controller.backup_queue.put(second)

        deadline = time.monotonic() + 2
        while controller.ack_backup_queue.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        controller.storage_stop_event.set()
        worker.join(timeout=2)
        acknowledgements = [
            controller.ack_backup_queue.get_nowait(),
            controller.ack_backup_queue.get_nowait(),
        ]

        self.assertEqual(calls, [first.id, second.id])
        self.assertEqual(acknowledgements, [first, second])
        self.assertEqual(first.completed_tokens, 64)
        self.assertEqual(second.completed_tokens, 64)
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main(verbosity=3)
