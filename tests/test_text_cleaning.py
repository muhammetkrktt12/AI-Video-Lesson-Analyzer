"""
tests/test_text_cleaning.py
RAG Metin Temizleme (Normalizasyon), Gürültü Cezası, 4. Sınıf Manifesto Doğrulaması ve Versiyonlu Chunk ID Testleri.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from index_core import (
    clean_pedagogical_text,
    generate_chunk_id,
    cleanup_stale_chunks,
    get_book_metadata_and_unit,
    PRIMARY_BOOK_MANIFESTS,
)
from utils.lexical_tools import (
    compute_noise_penalty,
    clean_snippet_preview,
    rerank_candidates,
)


class TestTextCleaningAndQuality(unittest.TestCase):
    def test_01_clean_pedagogical_text_removes_pure_dot_lines(self):
        """clean_pedagogical_text tamamen noktalardan oluşan öğrenci cevap satırlarını temizler."""
        sample = (
            "4. Kayaçlarla ilgili ödev hazırlayan öğrenciler sıralama yapacaktır.\n"
            ".......................................................................................\n"
            ".......................................................................................\n"
            "5. Fosiller ile ilgili bilgileri yazınız."
        )
        cleaned = clean_pedagogical_text(sample)
        self.assertNotIn("................................", cleaned)
        self.assertIn("Kayaçlarla ilgili ödev", cleaned)
        self.assertIn("Fosiller ile ilgili bilgileri yazınız.", cleaned)

    def test_02_clean_pedagogical_text_simplifies_in_line_dots_and_preserves_questions(self):
        """Satır içindeki 4+ ardışık nokta dizilerini ' ... ' yapar; soru, Roma rakamları ve şıkları korur."""
        sample = (
            "I. Milyonlarca yıl öncesine ait canlıların kalıntılarıdır. ........................................\n"
            "II. Sadece bitkilerin fosilleri bulunabilir. ....................................................\n"
            "III. Bulundukları bölgelerin iklimini gösterir."
        )
        cleaned = clean_pedagogical_text(sample)
        self.assertIn("I. Milyonlarca", cleaned)
        self.assertIn("II. Sadece", cleaned)
        self.assertIn("III. Bulundukları", cleaned)
        self.assertNotIn("................", cleaned)
        self.assertIn("...", cleaned)

    def test_03_clean_pedagogical_text_preserves_scientific_sentences(self):
        """Pedagojik bilimsel anlatım metinlerinin tek bir kelimesini dahi kaybetmeden korur."""
        text = (
            "Ekonomik değeri olan kayaçlara maden denir. "
            "Kayaçların yapısında mineraller bulunur. "
            "Magma soğuyarak katılaşır ve yer kabuğunu oluşturur."
        )
        cleaned = clean_pedagogical_text(text)
        self.assertEqual(cleaned, text)

    def test_04_clean_pedagogical_text_handles_underscores_and_dashes(self):
        """Öğrenci cevap çizgilerini (___ ve ---) sadeleştirir."""
        sample = "Elektrik enerjisi ________________ santrallerinde üretilir. Sonuç: --------"
        cleaned = clean_pedagogical_text(sample)
        self.assertIn("Elektrik enerjisi ___ santrallerinde üretilir.", cleaned)
        self.assertIn("Sonuç: ---", cleaned)

    def test_05_4th_grade_manifest_covers_pages_20_and_21(self):
        """4. Sınıf 1. Ünite başlangıç sayfasının 20 olduğunu ve Sayfa 20-21'in 1. Ünite'ye eşlendiğini doğrular."""
        fname = "ilkokul_fenbilimleri_4.sinif.pdf"
        
        # Sayfa 20: 1. Ünite Hazırlık ve Kazanım Sayfası
        meta_20, u_20, k_20, a_20 = get_book_metadata_and_unit(
            filename=fname, phys_page_no=21, printed_page_no=20, raw_text="test"
        )
        self.assertIn("1. Ünite: Yer Kabuğu ve Hareketler", u_20)
        self.assertEqual(k_20, "Yer Kabuğunun Yapısı")

        # Sayfa 21: Ünite Alanı
        meta_21, u_21, k_21, a_21 = get_book_metadata_and_unit(
            filename=fname, phys_page_no=22, printed_page_no=21, raw_text="test"
        )
        self.assertIn("1. Ünite: Yer Kabuğu ve Hareketler", u_21)

        # Sayfa 22: Ana Metin
        meta_22, u_22, k_22, a_22 = get_book_metadata_and_unit(
            filename=fname, phys_page_no=23, printed_page_no=22, raw_text="test"
        )
        self.assertIn("1. Ünite: Yer Kabuğu ve Hareketler", u_22)

    def test_06_compute_noise_penalty_clean_text_gets_no_penalty(self):
        """Nokta oranı eşik (%15) altındaki temiz anlatım metinlerine ceza uygulanmaz (çarpan 1.0)."""
        clean_text = "Dünya'nın katmanlarından biri yer kabuğudur. Magma soğuyarak kayaçları oluşturur."
        penalty = compute_noise_penalty(clean_text, dot_threshold=0.15)
        self.assertEqual(penalty, 1.0)

    def test_07_compute_noise_penalty_noisy_text_gets_penalized(self):
        """Nokta oranı yüksek (%50+) metinlerin puanı kademeli olarak sert biçimde düşürülür."""
        noisy_text = "Kayaçlar ............................................................................"
        penalty = compute_noise_penalty(noisy_text, dot_threshold=0.15)
        self.assertLess(penalty, 0.50)
        self.assertGreaterEqual(penalty, 0.15)

    def test_08_rerank_candidates_demotes_noisy_chunk_below_clean_chunk(self):
        """rerank_candidates gürültülü parçayı temiz parçanın altına düşürür."""
        def make_doc(content, name):
            return SimpleNamespace(page_content=content, metadata={"source": name, "chunk_id": name})

        noisy_chunk = make_doc(
            "yer kabuğunun yapısı kayaçlar ve madenler " + ("." * 400),
            "noisy_doc"
        )
        clean_chunk = make_doc(
            "yer kabuğunun yapısı kayaçlar ve madenler minerallerden oluşur.",
            "clean_doc"
        )

        # Gürültülü parça ilk sırada gelse bile reranking temiz olanı 1. sıraya taşımalıdır
        candidates = [noisy_chunk, clean_chunk]
        reranked = rerank_candidates("yer kabuğunun yapısı kayaçlar ve madenler", candidates, top_n=2)
        top1_doc, top1_score = reranked[0]
        self.assertEqual(top1_doc.metadata["source"], "clean_doc")

    def test_09_clean_snippet_preview_trims_leading_dots(self):
        """clean_snippet_preview baş taraftaki anlamsız nokta ve boşluk dizilerini budar."""
        raw_snippet = "........................................ III. Canlıların yapısı hakkında bilgi verir."
        clean_snip = clean_snippet_preview(raw_snippet, max_chars=80)
        self.assertTrue(clean_snip.startswith("III. Canlıların"))
        self.assertNotIn("................", clean_snip)

    def test_10_versioned_chunk_id_and_cleanup_compatibility(self):
        """Sürüm etiketli Chunk ID üretimini ve cleanup_stale_chunks'ın hedef sürüme göre eski parçaları tanımasını doğrular."""
        fname = "kitap.pdf"
        f_hash = "abc123456789"

        cid_default = generate_chunk_id(fname, f_hash, 10, 0)
        self.assertEqual(cid_default, "kitap.pdf:abc123456789:phys_0010:c000")

        cid_v2 = generate_chunk_id(fname, f_hash, 10, 0, version="v2")
        self.assertEqual(cid_v2, "kitap.pdf:abc123456789:v2:phys_0010:c000")

        # Mock ChromaDB cleanup with target_version="v2"
        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_store._collection = mock_col

        mock_col.get.return_value = {
            "ids": [
                "kitap.pdf:abc123456789:v2:phys_0010:c000",   # Hedef sürüm
                "kitap.pdf:abc123456789:phys_0010:c000",      # Eski v1 sürüm
            ],
            "metadatas": [
                {"source": "kitap.pdf", "file_hash": "abc123456789"},
                {"source": "kitap.pdf", "file_hash": "abc123456789"},
            ]
        }

        deleted = cleanup_stale_chunks(mock_store, fname, f_hash, target_version="v2")
        self.assertEqual(deleted, 1)
        mock_col.delete.assert_called_once_with(ids=["kitap.pdf:abc123456789:phys_0010:c000"])


if __name__ == "__main__":
    unittest.main()
