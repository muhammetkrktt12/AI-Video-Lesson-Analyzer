"""
tests/test_indexing_safety.py
İndeksleme güvenliği, hata görünürlüğü ve RAG hash fallback engeli testleri.
"""

import unittest
from unittest.mock import MagicMock, patch

from index_core import cleanup_stale_chunks
from index_ilkokul_books import (
    get_existing_chunk_ids,
    verify_book_chunks_in_db,
    check_book_legacy_records,
    can_cleanup_stale_chunks,
)
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

    def test_07_verify_book_chunks_in_db_exact_set_comparison(self):
        """verify_book_chunks_in_db yalnızca sayıları değil ID kümelerini doğrular; eksik ID varsa tespit eder."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        expected_ids = {"id_1", "id_2", "id_3"}

        # Durum 1: Tüm ID'ler mevcut
        mock_col.get.return_value = {"ids": ["id_1", "id_2", "id_3"]}
        is_ok, missing = verify_book_chunks_in_db(mock_store, expected_ids)
        self.assertTrue(is_ok)
        self.assertEqual(missing, set())

        # Durum 2: Bir ID eksik (2/3)
        mock_col.get.return_value = {"ids": ["id_1", "id_2"]}
        is_ok, missing = verify_book_chunks_in_db(mock_store, expected_ids)
        self.assertFalse(is_ok)
        self.assertEqual(missing, {"id_3"})

        # Durum 3: Sayı aynı (3 adet) fakat bir ID farklı (küme uyuşmazlığı!)
        mock_col.get.return_value = {"ids": ["id_1", "id_2", "id_wrong"]}
        is_ok, missing = verify_book_chunks_in_db(mock_store, expected_ids)
        self.assertFalse(is_ok)
        self.assertEqual(missing, {"id_3"})

    def test_08_can_cleanup_stale_chunks_blocks_on_missing_ids(self):
        """Beklenen ID'lerden biri dahi ChromaDB'de eksikse eski kayıtların silinmesi engellenir."""
        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=193,
            expected_new_count=193,
            missing_ids={"kitap.pdf:hash:phys_0049:c000"},
            page_errors=[]
        )
        self.assertFalse(can_clean, "Eksik ID varken temizliğe izin verilmemelidir.")

    def test_09_can_cleanup_stale_chunks_blocks_on_page_errors(self):
        """PDF metin çıkarma veya metadata hatası oluştuysa eksik içerik riski nedeniyle temizlik engellenir."""
        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=190,
            expected_new_count=190,
            missing_ids=set(),
            page_errors=[{"page": 12, "stage": "extraction", "error": "PyPDF extraction failed"}]
        )
        self.assertFalse(can_clean, "Sayfa metin çıkarma veya metadata hatası varken temizliğe izin verilmemelidir.")

    def test_10_can_cleanup_stale_chunks_blocks_on_batch_failure(self):
        """Batch veya tekil parça yazma hatası oluştuysa eski sürüm temizliği engellenir."""
        # book_has_failure = True durumu
        can_clean_1 = can_cleanup_stale_chunks(
            book_has_failure=True,
            book_success=192,
            expected_new_count=193,
            missing_ids=set(),
            page_errors=[]
        )
        self.assertFalse(can_clean_1)

        # book_success != expected_new_count durumu
        can_clean_2 = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=180,
            expected_new_count=193,
            missing_ids=set(),
            page_errors=[]
        )
        self.assertFalse(can_clean_2)

    def test_11_check_book_legacy_records_detects_legacy_schema(self):
        """check_book_legacy_records eski :pXXXX: şemalı ID'leri yeni :phys_ şemasından ayırarak listeler."""
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        mock_col.get.return_value = {
            "ids": [
                "kitap.pdf:421c:phys_0001:c000",
                "kitap.pdf:421c:phys_0002:c000",
                "kitap.pdf:421c:p0001:c000",  # Eski şema
                "kitap.pdf:421c:p0002:c000",  # Eski şema
            ]
        }

        legacy_ids = check_book_legacy_records(mock_store, "kitap.pdf")
        self.assertEqual(len(legacy_ids), 2)
        self.assertIn("kitap.pdf:421c:p0001:c000", legacy_ids)
        self.assertIn("kitap.pdf:421c:p0002:c000", legacy_ids)
        self.assertNotIn("kitap.pdf:421c:phys_0001:c000", legacy_ids)

    def test_12_can_cleanup_stale_chunks_allows_only_when_all_conditions_satisfied(self):
        """Tüm koşullar (sıfır hata, tam batch, tam ID kümesi, sıfır sayfa hatası) eksiksiz sağlandığında temizliğe izin verir."""
        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=193,
            expected_new_count=193,
            missing_ids=set(),
            page_errors=[]
        )
        self.assertTrue(can_clean, "Tüm koşullar sağlandığında temizliğe izin verilmelidir.")


if __name__ == "__main__":
    unittest.main()
