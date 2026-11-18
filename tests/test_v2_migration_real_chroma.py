"""
EduFrame AI - Gerçek ChromaDB Üzerinde İzole v2 Migrasyon Testleri
------------------------------------------------------------------
Bu test modülü mock kullanmaksızın doğrudan yerel diskte geçici bir
`chromadb.PersistentClient` ve LangChain `Chroma` örneği oluşturarak
tüm v2 migrasyon akışını ve güvenlik bariyerlerini doğrular:

1. v1'den v2'ye başarılı migrasyon ve eski kayıtların temizlenmesi
2. Kitaplar arası tam izolasyon (Book A temizlenirken Book B korunur)
3. Bozuk / sahte hash durumunda temizliğin engellenmesi ve v1'in korunması
4. Eksik v2 chunk durumunda (set difference) temizliğin engellenmesi
5. Yarıda kesilen migrasyonun kaldığı yerden devam etmesi (idempotent resumption)
6. Tamamlanmış migrasyonun ikinci kez çalıştırılmasında mükerrer kayıt oluşmaması
7. Veritabanındaki metnin tahrif edilmesi durumunda hash doğrulamasıyla yakalanması

NOT: Testler geçici dizin kullanır, gerçek canlı data/chroma_db'ye DOKUNMAZ.
Test sonunda tüm kaynaklar serbest bırakılarak dizinler temizlenir.
"""

import gc
import os
import shutil
import hashlib
import tempfile
import unittest
from typing import List, Dict, Any, Set, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from index_core import (
    generate_chunk_id,
    cleanup_stale_chunks,
)
from index_ilkokul_books import (
    CURRENT_VERSION,
    get_existing_chunk_ids,
    check_book_legacy_records,
    can_cleanup_stale_chunks,
    verify_book_chunks_in_db,
    verify_indexed_book_integrity,
)


class Dummy768Embeddings:
    """768 boyutlu deterministik sentetik embedding üretici (Ollama gerektirmez)."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i, text in enumerate(texts):
            # Metin uzunluğu ve indekse dayalı deterministik 768-D vektör
            val = round(((len(text) % 50) + i) * 0.01, 4)
            results.append([val] * 768)
        return results

    def embed_query(self, text: str) -> List[float]:
        val = round((len(text) % 50) * 0.01, 4)
        return [val] * 768


class TestRealChromaDBMigration(unittest.TestCase):
    """Gerçek ChromaDB persistent client üzerinde izole migrasyon testleri."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="eduframe_chroma_test_")
        self.embeddings = Dummy768Embeddings()
        self.collection_name = "test_eduframe_meb_768"
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.tmpdir,
        )

        self.book_a = "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf"
        self.hash_a = "a1b2c3d4e5f611112222"
        self.book_b = "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"
        self.hash_b = "f6e5d4c3b2a199998888"

    def tearDown(self):
        # Windows dosya kilitlemelerini önlemek için ChromaDB sistemini durdur
        if hasattr(self, "store") and hasattr(self.store, "_client"):
            try:
                self.store._client._system.stop()
            except Exception:
                pass
        del self.store
        gc.collect()

        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_v1_doc(self, filename: str, file_hash: str, page: int, chunk_idx: int, text: str) -> Tuple[Document, str]:
        """Eski v1 şemalı (versiyonsuz) parça oluşturur."""
        cid = f"{filename}:{file_hash[:12]}:phys_{page:04d}:c{chunk_idx:03d}"
        meta = {
            "source": filename,
            "kitap_adi": "Fen Bilimleri",
            "egitim_seviyesi": "İlkokul",
            "ders_adi": "Fen Bilimleri",
            "sinif": "3. Sınıf",
            "physical_page": page,
            "printed_page": page,
            "page": page,
            "sayfa_no": page,
            "has_printed_page": True,
            "unite": "1. Ünite",
            "konu": "Gezegenimizi Tanıyalım",
            "alt_konu": "Dünya'nın Şekli",
            "referans": f"Fen Bilimleri - 1. Ünite [Sayfa {page}]",
            "chunk_id": cid,
            "file_hash": file_hash[:12],
            "char_count": len(text),
            "token_count": 25,
            "version": "v1",
        }
        return Document(page_content=text, metadata=meta), cid

    def _create_v2_doc(
        self,
        filename: str,
        file_hash: str,
        page: int,
        chunk_idx: int,
        text: str,
        corrupt_hash: bool = False,
    ) -> Tuple[Document, str]:
        """Yeni v2 şemalı (version='v2' ve SHA-256 content_hash) parça oluşturur."""
        cid = generate_chunk_id(filename, file_hash, page, chunk_idx, version="v2")
        real_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        c_hash = "corrupted000" if corrupt_hash else real_hash

        meta = {
            "source": filename,
            "kitap_adi": "Fen Bilimleri",
            "egitim_seviyesi": "İlkokul",
            "ders_adi": "Fen Bilimleri",
            "sinif": "3. Sınıf",
            "physical_page": page,
            "printed_page": page,
            "page": page,
            "sayfa_no": page,
            "has_printed_page": True,
            "unite": "1. Ünite",
            "konu": "Gezegenimizi Tanıyalım",
            "alt_konu": "Dünya'nın Şekli",
            "referans": f"Fen Bilimleri - 1. Ünite [Sayfa {page}]",
            "chunk_id": cid,
            "file_hash": file_hash[:12],
            "char_count": len(text),
            "token_count": 25,
            "version": "v2",
            "content_hash": c_hash,
        }
        return Document(page_content=text, metadata=meta), cid

    # -------------------------------------------------------------------------
    # TEST 1: Gerçek ChromaDB'de v1 -> v2 Temiz Migrasyon ve Kitap İzolasyonu
    # -------------------------------------------------------------------------
    def test_01_real_chroma_v1_to_v2_clean_migration_and_isolation(self):
        """Gerçek ChromaDB'de v1 kayıtları v2 başarıyla yazıldıktan sonra silinir; Book B korunur."""
        # 1. Book A için 3 adet v1 kaydı ekle
        v1_docs_a, v1_ids_a = [], []
        for i in range(1, 4):
            doc, cid = self._create_v1_doc(
                self.book_a, self.hash_a, page=i, chunk_idx=0,
                text=f"Eski v1 ders içeriği sayfa {i} Kayaçlar ve mineraller konusu."
            )
            v1_docs_a.append(doc)
            v1_ids_a.append(cid)
        self.store.add_documents(v1_docs_a, ids=v1_ids_a)

        # 2. Book B için 2 adet v1 kaydı ekle (henüz migrasyona girmemiş kitap)
        v1_docs_b, v1_ids_b = [], []
        for i in range(1, 3):
            doc, cid = self._create_v1_doc(
                self.book_b, self.hash_b, page=i, chunk_idx=0,
                text=f"Book B v1 ders içeriği sayfa {i} Elektrikli araç gereçler konusu."
            )
            v1_docs_b.append(doc)
            v1_ids_b.append(cid)
        self.store.add_documents(v1_docs_b, ids=v1_ids_b)

        self.assertEqual(self.store._collection.count(), 5)

        # 3. Book A için yeni v2 kayıtları ekle
        v2_docs_a, v2_ids_a = [], []
        for i in range(1, 4):
            doc, cid = self._create_v2_doc(
                self.book_a, self.hash_a, page=i, chunk_idx=0,
                text=f"Temizlenmiş pedagojik v2 ders içeriği sayfa {i} Kayaçlar konusu."
            )
            v2_docs_a.append(doc)
            v2_ids_a.append(cid)
        self.store.add_documents(v2_docs_a, ids=v2_ids_a)

        # Şu an veritabanında toplam 8 kayıt olmalı (3 v1 A + 2 v1 B + 3 v2 A)
        self.assertEqual(self.store._collection.count(), 8)

        # 4. Book A için doğrulama yap
        expected_ids = set(v2_ids_a)
        is_verified, missing = verify_book_chunks_in_db(self.store, expected_ids)
        self.assertTrue(is_verified)
        self.assertEqual(len(missing), 0)

        integrity_ok, issues = verify_indexed_book_integrity(
            self.store, self.book_a, expected_ids, target_version="v2"
        )
        self.assertTrue(integrity_ok)
        self.assertEqual(len(issues), 0)

        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=len(v2_docs_a),
            expected_new_count=len(v2_docs_a),
            missing_ids=missing,
            page_errors=[],
            integrity_ok=integrity_ok,
        )
        self.assertTrue(can_clean)

        # 5. Book A için eski sürüm temizliğini çalıştır
        cleaned = cleanup_stale_chunks(
            self.store, self.book_a, self.hash_a, target_version="v2"
        )
        self.assertEqual(cleaned, 3)

        # 6. Veritabanını kontrol et
        self.assertEqual(self.store._collection.count(), 5)

        # Book A'nın v1 parçaları silinmiş olmalı
        res_a_v1 = self.store._collection.get(ids=v1_ids_a, include=[])
        self.assertEqual(len(res_a_v1["ids"]), 0)

        # Book A'nın v2 parçaları duruyor olmalı
        res_a_v2 = self.store._collection.get(ids=v2_ids_a, include=[])
        self.assertEqual(len(res_a_v2["ids"]), 3)

        # Book B'nin v1 parçaları HİÇ DOKUNULMADAN korunmuş olmalı
        res_b_v1 = self.store._collection.get(ids=v1_ids_b, include=[])
        self.assertEqual(len(res_b_v1["ids"]), 2)

    # -------------------------------------------------------------------------
    # TEST 2: Bozuk Hash Durumunda Temizliğin Engellenmesi
    # -------------------------------------------------------------------------
    def test_02_real_chroma_corrupted_hash_blocks_cleanup(self):
        """Gerçek ChromaDB'de hash uyuşmazlığı olduğunda verify_indexed_book_integrity reddeder, v1 silinmez."""
        # 1. Book A v1 parçası ekle
        v1_doc, v1_id = self._create_v1_doc(
            self.book_a, self.hash_a, 1, 0, "Eski v1 içerik metni fen bilgisi."
        )
        self.store.add_documents([v1_doc], ids=[v1_id])

        # 2. Book A v2 parçası ekle fakat corrupt_hash=True
        v2_doc, v2_id = self._create_v2_doc(
            self.book_a, self.hash_a, 1, 0,
            "Temiz v2 içerik metni fakat hash bozuk.", corrupt_hash=True
        )
        self.store.add_documents([v2_doc], ids=[v2_id])

        # 3. Bütünlük denetimi
        integrity_ok, issues = verify_indexed_book_integrity(
            self.store, self.book_a, {v2_id}, target_version="v2"
        )
        self.assertFalse(integrity_ok)
        self.assertTrue(any("content_hash uyuşmazlığı" in s for s in issues))

        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=1,
            expected_new_count=1,
            missing_ids=set(),
            page_errors=[],
            integrity_ok=integrity_ok,
        )
        self.assertFalse(can_clean)

        # 4. Temizlik yapılmadığından v1 kaydı korunur
        res_v1 = self.store._collection.get(ids=[v1_id], include=[])
        self.assertEqual(len(res_v1["ids"]), 1)

    # -------------------------------------------------------------------------
    # TEST 3: Eksik v2 Parça Durumunda Temizliğin Engellenmesi
    # -------------------------------------------------------------------------
    def test_03_real_chroma_missing_v2_chunk_blocks_cleanup(self):
        """ChromaDB'ye yazılmayan veya eksik kalan v2 parçası varsa eski kayıtlar silinmez."""
        # 1. 2 adet v1 kaydı ekle
        v1_docs, v1_ids = [], []
        for i in range(1, 3):
            doc, cid = self._create_v1_doc(self.book_a, self.hash_a, i, 0, f"v1 metin {i}")
            v1_docs.append(doc)
            v1_ids.append(cid)
        self.store.add_documents(v1_docs, ids=v1_ids)

        # 2. Beklenen 2 adet v2 var fakat yalnızca 1 tanesini ChromaDB'ye ekle
        v2_doc_1, v2_id_1 = self._create_v2_doc(
            self.book_a, self.hash_a, 1, 0, "Temiz pedagojik v2 metni 1."
        )
        _, v2_id_2 = self._create_v2_doc(
            self.book_a, self.hash_a, 2, 0, "Temiz pedagojik v2 metni 2."
        )
        self.store.add_documents([v2_doc_1], ids=[v2_id_1])

        expected_ids = {v2_id_1, v2_id_2}

        # 3. ID doğrulama eksik ID'yi yakalar
        is_verified, missing = verify_book_chunks_in_db(self.store, expected_ids)
        self.assertFalse(is_verified)
        self.assertEqual(missing, {v2_id_2})

        can_clean = can_cleanup_stale_chunks(
            book_has_failure=False,
            book_success=1,
            expected_new_count=2,
            missing_ids=missing,
            page_errors=[],
            integrity_ok=True,
        )
        self.assertFalse(can_clean)

        # v1 kayıtları korunur
        res_v1 = self.store._collection.get(ids=v1_ids, include=[])
        self.assertEqual(len(res_v1["ids"]), 2)

    # -------------------------------------------------------------------------
    # TEST 4: Yarıda Kesilen Migrasyonun Doğru Şekilde Devam Etmesi (Resumption)
    # -------------------------------------------------------------------------
    def test_04_real_chroma_interrupted_migration_resumption(self):
        """Kesilen migrasyon yeniden başlatıldığında mevcut v2 parçaları atlanır, eksikler tamamlanır."""
        # v1 kayıtları
        v1_docs, v1_ids = [], []
        for i in range(1, 4):
            doc, cid = self._create_v1_doc(self.book_a, self.hash_a, i, 0, f"v1 metin {i}")
            v1_docs.append(doc)
            v1_ids.append(cid)
        self.store.add_documents(v1_docs, ids=v1_ids)

        # Hedeflenen 3 v2 parçası
        all_v2 = [
            self._create_v2_doc(self.book_a, self.hash_a, i, 0, f"Pedagojik v2 içerik {i}")
            for i in range(1, 4)
        ]

        # 1. Aşama: İlk çalıştırmada sadece 1. parça eklendi ve işlem kesildi
        self.store.add_documents([all_v2[0][0]], ids=[all_v2[0][1]])
        self.assertEqual(self.store._collection.count(), 4)  # 3 v1 + 1 v2

        # 2. Aşama: Resumption — İndeksleyici baştan başlatılıyor
        existing_ids = get_existing_chunk_ids(self.store)
        self.assertIn(all_v2[0][1], existing_ids)

        queued_docs = []
        queued_ids = []
        skipped_count = 0
        for doc, cid in all_v2:
            if cid in existing_ids:
                skipped_count += 1
                continue
            queued_docs.append(doc)
            queued_ids.append(cid)

        # İlk parça atlanmalı, kalan 2 parça eklenmeli
        self.assertEqual(skipped_count, 1)
        self.assertEqual(len(queued_docs), 2)

        self.store.add_documents(queued_docs, ids=queued_ids)

        # 3. Artık tüm v2 parçaları tamamlandı, doğrulama ve temizlik yapılır
        expected_ids = {cid for _, cid in all_v2}
        is_verified, missing = verify_book_chunks_in_db(self.store, expected_ids)
        self.assertTrue(is_verified)
        self.assertEqual(len(missing), 0)

        integrity_ok, issues = verify_indexed_book_integrity(
            self.store, self.book_a, expected_ids, target_version="v2"
        )
        self.assertTrue(integrity_ok)

        cleaned = cleanup_stale_chunks(self.store, self.book_a, self.hash_a, target_version="v2")
        self.assertEqual(cleaned, 3)

        # Sonuç: Tam olarak 3 v2 kaydı kalmalı
        self.assertEqual(self.store._collection.count(), 3)
        res = self.store._collection.get(ids=list(expected_ids), include=[])
        self.assertEqual(len(res["ids"]), 3)

    # -------------------------------------------------------------------------
    # TEST 5: İkinci Kez Çalıştırmada Mükerrer Kayıt Oluşmaması (Idempotency)
    # -------------------------------------------------------------------------
    def test_05_real_chroma_idempotency_second_run_no_duplicates(self):
        """Tamamlanmış migrasyon ikinci kez çalıştırıldığında DB'ye hiçbir yeni kayıt eklenmez."""
        all_v2 = [
            self._create_v2_doc(self.book_a, self.hash_a, i, 0, f"Pedagojik v2 metin sayfa {i}")
            for i in range(1, 4)
        ]
        self.store.add_documents([d for d, _ in all_v2], ids=[c for _, c in all_v2])
        self.assertEqual(self.store._collection.count(), 3)

        # 2. Kez çalıştırma simülasyonu
        existing_ids = get_existing_chunk_ids(self.store)
        queued_docs = []
        for doc, cid in all_v2:
            if cid in existing_ids:
                continue
            queued_docs.append(doc)

        self.assertEqual(len(queued_docs), 0)
        # book_docs boş olduğunda devreye giren mekanizma:
        expected_ids = {c for _, c in all_v2}
        is_verified, missing = verify_book_chunks_in_db(self.store, expected_ids)
        self.assertTrue(is_verified)

        integrity_ok, issues = verify_indexed_book_integrity(
            self.store, self.book_a, expected_ids, target_version="v2"
        )
        self.assertTrue(integrity_ok)

        legacy = check_book_legacy_records(self.store, self.book_a, target_version="v2")
        self.assertEqual(legacy, [])

        # DB sayısı değişmemiştir
        self.assertEqual(self.store._collection.count(), 3)

    # -------------------------------------------------------------------------
    # TEST 6: Gerçek DB'de Metin Tahrif Edildiğinde Hash Doğrulamasının Yakalaması
    # -------------------------------------------------------------------------
    def test_06_real_chroma_text_modified_in_db_fails_integrity(self):
        """ChromaDB'deki dokümanın metni değiştirilirse hash kontrolü uyuşmazlığı tespit eder."""
        doc, cid = self._create_v2_doc(
            self.book_a, self.hash_a, 1, 0, "Orijinal pedagojik ders metni içeriği."
        )
        self.store.add_documents([doc], ids=[cid])

        # Başlangıçta bütünlük tam olmalı
        ok, issues = verify_indexed_book_integrity(self.store, self.book_a, {cid}, target_version="v2")
        self.assertTrue(ok)

        # ChromaDB'deki metin doğrudan güncellenip değiştirilsin (tahrifat)
        self.store._collection.update(
            ids=[cid],
            documents=["Tahrif edilmiş sahte metin içeriği."],
            embeddings=[[0.01] * 768]
        )

        # Artık hash doğrulaması yakalamalıdır
        ok_tampered, issues_tampered = verify_indexed_book_integrity(
            self.store, self.book_a, {cid}, target_version="v2"
        )
        self.assertFalse(ok_tampered)
        self.assertTrue(any("content_hash uyuşmazlığı" in s for s in issues_tampered))
        self.assertTrue(any(cid in s for s in issues_tampered))


if __name__ == "__main__":
    unittest.main()
