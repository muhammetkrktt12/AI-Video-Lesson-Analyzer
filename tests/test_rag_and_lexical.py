"""
EduFrame AI - Lexical Benzerlik, Reranking ve RAG Entegrasyon Testleri (tests/test_rag_and_lexical.py)
-----------------------------------------------------------------------------------------------------
Bu test modülü:
1. Türkçe I/İ harf dönüşümü ve kelime ayrıştırma doğruluğunu
2. 3-gram trigram küme üretimini
3. Lexical similarity oranlarını ve matematiksel hassasiyeti
4. Aday doküman sıralama (rerank_candidates) algoritmasını
5. [K1], [K2] atıf doğrulama (validate_citations) mekanizmasını
6. Hit@k başarı ölçümünü
7. services.rag_motoru ile utils.lexical_tools arasındaki d5_05'ten arındırılmış bağımsızlığı
8. quality_utils.py kök sarmalayıcısının geriye dönük uyumluluğunu
doğrular.
"""

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

# Proje kökünü sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.lexical_tools import (
    kelimelere_ayir,
    ucgramlar,
    lexical_similarity,
    tokens,
    lexical_score,
    rerank_candidates,
    validate_citations,
    hit_at_k,
)
import quality_utils


def make_doc(content: str, source: str = "kitap.pdf", page: int = 1):
    """Testler için hafif LangChain Document benzeri nesne üretir."""
    return SimpleNamespace(
        page_content=content,
        metadata={"source": source, "page": page, "printed_page": page, "sayfa_no": page}
    )


class TestLexicalTools(unittest.TestCase):
    """Türkçe 3-gram ve metin işleme testleri."""

    def test_turkish_case_folding(self):
        """Türkçe I/İ harflerinin doğru küçültüldüğünü ve noktalama temizliğini test eder."""
        metin = "İSTANBUL IŞIK Şemsiye Çığ Ördek 123!"
        sonuc = kelimelere_ayir(metin)
        self.assertEqual(sonuc, ["istanbul", "ışık", "şemsiye", "çığ", "ördek", "123"])

    def test_ucgram_generation(self):
        """Karakter 3-gramlarının ^ ve $ sınır işaretçileriyle doğru oluştuğunu test eder."""
        metin = "arı"
        gramlar = ucgramlar(metin)
        # ^arı$ -> ['^ar', 'arı', 'rı$']
        beklenen = {"^ar", "arı", "rı$"}
        self.assertEqual(gramlar, beklenen)

    def test_lexical_similarity_extremes(self):
        """Birebir aynı metinde 1.0, boş metinlerde 0.0 dönmelidir."""
        self.assertEqual(lexical_similarity("elektrik", "elektrik"), 1.0)
        self.assertEqual(lexical_similarity("", "elektrik"), 0.0)
        self.assertEqual(lexical_similarity("elektrik", ""), 0.0)

    def test_lexical_similarity_partial_match(self):
        """Kısmi eşleşmelerde 0.0 ile 1.0 arasında deterministik oran döner."""
        q = "elektrik"
        metin = "elektrikli aletler"
        skor = lexical_similarity(q, metin)
        # 'elektrik' için '^elektrik$' (8 trigram). 'elektrikli' içinde 'ik$' yerine 'ikl' olduğu için 7/8 = 0.875 eşleşir
        self.assertAlmostEqual(skor, 0.875, places=3)
        self.assertGreater(skor, 0.8)

        # Farklı bir kelimede
        q2 = "elektrik enerjisi"
        metin2 = "sadece rüzgar gücü"
        skor2 = lexical_similarity(q2, metin2)
        self.assertLess(skor2, 0.2)


class TestRerankCandidates(unittest.TestCase):
    """Aday dokümanları hibrit skorla yeniden sıralama testleri."""

    def test_rerank_ordering(self):
        """Hem anlamsal sırayı hem de lexical benzerliği birleştiren sıralamayı test eder."""
        docs = [
            make_doc("Fosiller ve kayaçların oluşumu yer kabuğunda gerçekleşir.", page=20),
            make_doc("Elektrikli araç gereçler şehir elektriği veya pil ile çalışır.", page=44),
            make_doc("Besinler vücudumuzun büyümesi için gereklidir.", page=60),
        ]
        sorgu = "elektrikli araç gereçler"
        sonuclar = rerank_candidates(sorgu, docs, lexical_agirlik=0.4, top_n=3)

        self.assertEqual(len(sonuclar), 3)
        # 44. sayfadaki elektrik dokümanı 1. sıraya yükselmeli
        top_doc, top_score = sonuclar[0]
        self.assertEqual(top_doc.metadata["page"], 44)
        self.assertGreater(top_score, 0.5)

    def test_empty_candidates(self):
        """Boş aday listesi verildiğinde boş liste dönmelidir."""
        self.assertEqual(rerank_candidates("test", []), [])

    def test_invalid_weight(self):
        """Geçersiz ağırlık verildiğinde ValueError fırlatmalıdır."""
        docs = [make_doc("test")]
        with self.assertRaises(ValueError):
            rerank_candidates("test", docs, lexical_agirlik=1.5)
        with self.assertRaises(ValueError):
            rerank_candidates("test", docs, lexical_agirlik=-0.1)


class TestCitationAndHitAtK(unittest.TestCase):
    """Atıf doğrulama ve Hit@k metrik testleri."""

    def test_validate_citations_valid(self):
        cevap = "Öğrendiğimize göre [K1] ve [K2] temel kaynaklardır."
        res = validate_citations(cevap, ["K1", "K2", "K3"])
        self.assertTrue(res["valid"])
        self.assertEqual(res["found"], ["K1", "K2"])
        self.assertEqual(res["invalid"], [])

    def test_validate_citations_invalid(self):
        cevap = "Metinde [K1] ve bilinmeyen [K9] atıfları var."
        res = validate_citations(cevap, ["K1", "K2"])
        self.assertFalse(res["valid"])
        self.assertIn("K9", res["invalid"])

    def test_validate_citations_empty(self):
        cevap = "Cevapta hiç atıf yok."
        res = validate_citations(cevap, ["K1"])
        self.assertFalse(res["valid"])

    def test_hit_at_k(self):
        docs = [
            make_doc("sayfa 10", source="fen1.pdf", page=10),
            make_doc("sayfa 20", source="fen2.pdf", page=20),
        ]
        self.assertEqual(hit_at_k(docs, 10), 1)
        self.assertEqual(hit_at_k(docs, 20), 1)
        self.assertEqual(hit_at_k(docs, 99), 0)
        self.assertEqual(hit_at_k(docs, [99, 10]), 1)
        self.assertEqual(hit_at_k(docs, "fen1.pdf"), 1)
        self.assertEqual(hit_at_k(docs, "olmayan.pdf"), 0)


class TestRAGMotoruDecoupling(unittest.TestCase):
    """RAG motorunun utils.lexical_tools ile bağımsız çalıştığını doğrular."""

    def test_rag_motoru_imports_cleanly_without_d5(self):
        """services.rag_motoru yüklendiğinde d5_05_search'e ihtiyaç duymamalıdır."""
        from services.rag_motoru import RAGMotoru
        motor = RAGMotoru()
        durum = motor.mevcut_veritabani_durumu()
        self.assertIsInstance(durum, dict)
        self.assertIn("koleksiyon", durum)

    def test_backward_compatibility_wrapper(self):
        """Kök dizindeki quality_utils.py'nin aynı API'yi hatasız sağladığını test eder."""
        self.assertTrue(hasattr(quality_utils, "rerank_candidates"))
        self.assertTrue(hasattr(quality_utils, "lexical_similarity"))
        self.assertTrue(hasattr(quality_utils, "validate_citations"))
        self.assertTrue(hasattr(quality_utils, "hit_at_k"))
        
        # Fonksiyon çağrısı testi
        res = quality_utils.validate_citations("Doğru [K1]", ["K1"])
        self.assertTrue(res["valid"])


if __name__ == "__main__":
    unittest.main()
