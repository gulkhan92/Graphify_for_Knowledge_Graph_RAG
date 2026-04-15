import unittest

from graphify_rag.monitoring import MetricsRegistry


class MonitoringTests(unittest.TestCase):
    def test_metrics_snapshot_contains_aggregates(self) -> None:
        registry = MetricsRegistry()
        registry.record_request(0.1)
        registry.record_request(0.3)
        registry.record_ingest()
        registry.record_question()

        snapshot = registry.snapshot()

        self.assertEqual(snapshot["requests_total"], 2)
        self.assertEqual(snapshot["ingestions_total"], 1)
        self.assertEqual(snapshot["questions_total"], 1)
        self.assertAlmostEqual(snapshot["avg_request_latency_seconds"], 0.2, places=4)


if __name__ == "__main__":
    unittest.main()
