from pathlib import Path
import tempfile
import unittest

from graphify_rag.graph_store import GraphStore
from graphify_rag.models import Chunk, Document, Entity, GraphSnapshot, Relation


class GraphStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = GraphStore(Path(tmp_dir))
            snapshot = GraphSnapshot(
                documents=[Document("doc", "Title", Path("sample.pdf"), "body")],
                chunks=[Chunk("c1", "doc", "chunk text", 0, 2)],
                entities=[Entity("e1", "FinRL-X", "CONCEPT", chunk_ids=["c1"])],
                relations=[Relation("e1", "e1", "related_to", 1.0, ["c1"])],
            )

            store.save(snapshot)
            loaded = store.load()

            self.assertEqual(loaded.documents[0].title, "Title")
            self.assertEqual(loaded.chunks[0].chunk_id, "c1")
            self.assertEqual(loaded.entities[0].name, "FinRL-X")


if __name__ == "__main__":
    unittest.main()
