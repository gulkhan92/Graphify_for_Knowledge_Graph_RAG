from pathlib import Path
import tempfile
import unittest

from graphify_rag.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(Path(tmp_dir))
            vectors = {"c1": [0.1, 0.2, 0.3], "c2": [0.5, 0.6, 0.7]}

            store.save(vectors)
            loaded = store.load()

            self.assertEqual(loaded, vectors)


if __name__ == "__main__":
    unittest.main()
