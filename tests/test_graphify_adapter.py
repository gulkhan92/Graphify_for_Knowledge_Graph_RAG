from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from graphify_rag.graphify_adapter import GraphifyAdapter


class GraphifyAdapterTests(unittest.TestCase):
    def test_parse_node_link_graph_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = Path(tmp_dir) / "data"
            artifacts_dir = Path(tmp_dir) / "artifacts"
            input_dir.mkdir()
            artifacts_dir.mkdir()
            graph_json = artifacts_dir / "graphify-out" / "graph.json"
            graph_json.parent.mkdir()
            payload = {
                "nodes": [
                    {
                        "id": "FinRL-X",
                        "name": "FinRL-X",
                        "label": "Concept",
                        "path": str(input_dir / "doc1.pdf"),
                        "summary": "FinRL-X unifies trading infrastructure."
                    },
                    {
                        "id": "Portfolio Allocation",
                        "name": "Portfolio Allocation",
                        "label": "Concept",
                        "path": str(input_dir / "doc1.pdf"),
                        "summary": "Portfolio allocation is part of the system."
                    }
                ],
                "links": [
                    {"source": "FinRL-X", "target": "Portfolio Allocation", "label": "integrates", "weight": 2}
                ]
            }
            graph_json.write_text(json.dumps(payload), encoding="utf-8")

            adapter = GraphifyAdapter(input_dir=input_dir, artifacts_dir=artifacts_dir)
            snapshot = adapter._parse_graph_json(graph_json)

            self.assertEqual(len(snapshot.documents), 1)
            self.assertEqual(len(snapshot.entities), 2)
            self.assertEqual(snapshot.relations[0].relation, "integrates")

    @patch("graphify_rag.graphify_adapter.shutil.which", return_value="/usr/bin/graphify")
    @patch("graphify_rag.graphify_adapter.subprocess.run")
    def test_build_snapshot_runs_graphify(self, mock_run, _mock_which) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = Path(tmp_dir) / "data"
            artifacts_dir = Path(tmp_dir) / "artifacts"
            input_dir.mkdir()
            graph_json = artifacts_dir / "graphify-out" / "graph.json"
            graph_json.parent.mkdir(parents=True)
            graph_json.write_text(
                json.dumps(
                    {
                        "nodes": [{"id": "A", "name": "A", "label": "Concept", "path": str(input_dir / "doc.pdf"), "summary": "A"}],
                        "links": []
                    }
                ),
                encoding="utf-8",
            )
            adapter = GraphifyAdapter(input_dir=input_dir, artifacts_dir=artifacts_dir)

            snapshot = adapter.build_snapshot()

            self.assertEqual(len(snapshot.entities), 1)
            mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
