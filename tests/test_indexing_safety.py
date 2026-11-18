"""
tests/test_indexing_safety.py
İndeksleme güvenliği, hata görünürlüğü ve RAG hash fallback engeli testleri.
"""

import unittest
from unittest.mock import MagicMock, patch

from index_core import cleanup_stale_chunks
from index_ilkokul_books import get_existing_chunk_ids
from services.rag_motoru import GuvenliEmbeddingMotoru, RAGMotoru


class TestIndexingSafety(unittest.TestCase):
    def test_01_cleanup_stale_chunks_surfaces_db_errors(self):
        """cleanup_stale_chunks veritabanı hatalarını sessizce yutmak yerine görünür kılar (RuntimeError fırlatır)."""
        broken_store = MagicMock()
        # Mock Chroma collection that raises an OperationalError
        broken_store._collection.get.side_effect = RuntimeError("SQLite database locked or corrupted")

        with self.assertRaises(RuntimeError) as ctx:
            cleanup_stale_chunks(
                store=broken_store,
                filename="test_kitap.pdf",
                current_hash="abc123def456"
            )
        self.assertIn("ChromaDB eski sürüm temizliği", str(ctx.exception))

    def test_02_get_existing_chunk_ids_surfaces_db_errors(self):
        """get_existing_chunk_ids veritabanı hatalarını yutup set() dönmek yerine RuntimeError fırlatır."""
        broken_store = MagicMock()
        broken_store._collection.count.side_effect = RuntimeError("Disk I/O failure")

        with self.assertRaises(RuntimeError) as ctx:
            get_existing_chunk_ids(broken_store)
        self.assertIn("ChromaDB mevcut kayıtları taranırken veritabanı hatası", str(ctx.exception))

    def test_03_rag_motoru_hash_fallback_disabled_in_production(self):
        """Üretim modunda (allow_fallback=False) Ollama kapalıysa RAG motoru sessizce hash üretmek yerine RuntimeError fırlatır."""
        motor = GuvenliEmbeddingMotoru(allow_fallback=False)
        # Ollama istemcisini None yap ve kullanılamaz olarak işaretle (simüle çevrimdışı)
        motor._ollama_client = None
        motor._ollama_kullanilabilir = False

        with self.assertRaises(RuntimeError) as ctx:
            motor.embed_query("fen bilimleri elektrik santralleri")
        self.assertIn("hash fallback kapalıdır", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx_doc:
            motor.embed_documents(["doküman metni 1", "doküman metni 2"])
        self.assertIn("hash fallback kapalıdır", str(ctx_doc.exception))

    def test_04_rag_motoru_hash_fallback_explicit_opt_in(self):
        """allow_fallback=True açıkça belirtildiğinde (örneğin çevrimdışı testler) deterministik hash üretimine izin verir."""
        motor = GuvenliEmbeddingMotoru(allow_fallback=True)
        motor._ollama_client = None
        motor._ollama_kullanilabilir = False

        vec = motor.embed_query("elektrikli aletler")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), 768)

        vecs = motor.embed_documents(["örnek parça"])
        self.assertIsInstance(vecs, list)
        self.assertEqual(len(vecs[0]), 768)

    def test_05_cleanup_stale_chunks_deletes_only_different_hashes(self):
        """cleanup_stale_chunks aynı hash'li parçaları korur, yalnızca farklı hash'li parçaları siler."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        current_hash = "11112222333344445555"
        current_prefix = current_hash[:12]
        old_prefix = "999988887777"

        # 3 parça dön: 2 tanesi güncel hash, 1 tanesi eski hash
        mock_col.get.return_value = {
            "ids": ["c1", "c2", "c3"],
            "metadatas": [
                {"source": "kitap.pdf", "file_hash": current_prefix},
                {"source": "kitap.pdf", "file_hash": old_prefix},
                {"source": "kitap.pdf", "file_hash": current_prefix},
            ]
        }

        deleted_count = cleanup_stale_chunks(mock_store, "kitap.pdf", current_hash)
        self.assertEqual(deleted_count, 1)
        mock_col.delete.assert_called_once_with(ids=["c2"])

    def test_06_cleanup_stale_chunks_cleans_legacy_schema_with_same_hash(self):
        """cleanup_stale_chunks aynı hash'e sahip olsa dahi eski :pXXXX: şemalı ID'leri tespit edip temizler."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        current_hash = "421c34396672abcdef"
        current_prefix = current_hash[:12]

        # 1 adet yeni şema (:phys_), 1 adet eski şema (:p) - her ikisi de aynı güncel hash'e sahip
        new_schema_id = f"kitap.pdf:{current_prefix}:phys_0049:c000"
        legacy_schema_id = f"kitap.pdf:{current_prefix}:p0048:c000"

        mock_col.get.return_value = {
            "ids": [new_schema_id, legacy_schema_id],
            "metadatas": [
                {"source": "kitap.pdf", "file_hash": current_prefix},
                {"source": "kitap.pdf", "file_hash": current_prefix},
            ]
        }

        deleted_count = cleanup_stale_chunks(mock_store, "kitap.pdf", current_hash)
        self.assertEqual(deleted_count, 1)
        mock_col.delete.assert_called_once_with(ids=[legacy_schema_id])


if __name__ == "__main__":
    unittest.main()
