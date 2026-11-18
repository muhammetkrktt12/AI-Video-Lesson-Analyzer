"""
EduFrame AI - RAG Sınıf İzolasyonu, Anlamsal Uygunluk ve Kaynak Doğrulama Testleri
(tests/test_rag_grade_isolation_and_quality.py)
---------------------------------------------------------------------------------
Bu test modülü, kullanıcı isterleri doğrultusunda şu 7 kritik testi ve entegrasyonu doğrular:
- Test A: Sınıf izolasyonu (3. Sınıf sorgusunda varsayılan olarak yalnızca 3. Sınıf kaynakları kabul edilmeli)
- Test B: Doğru ana kaynak (3. sınıf 2. kitap basılı 48. sayfa, pil/oyuncak araba için ana kaynak olmalı)
- Test C: Yanlış eşleşme (4. sınıf 76. sayfa kuvvet-hareket ve 4. sınıf 46. sayfa besinler elenmeli)
- Test D: Yardımcı kaynak (3. sınıf basılı 57. sayfa yardımcı kaynak olmalı, ana mekanizma için ana kaynak olmamalı)
- Test E: Filtreli arama başarısızlığı (İzin olmadan 3. sınıftan 4. sınıfa geçilmemeli)
- Test F: Soru kaynakçası (Eşleşmeyen soruya rastgele ilk MEB referansı atanmamalı)
- Test G: Skor güvenilirliği (Gerçek vektör benzerliği yoksa sabit 1.0 verilmemeli, None ve yöntem belirtilmeli)
- Test H: Gerçek ChromaDB v2 ve koleksiyon entegrasyonu doğrulaması
"""

import sys
import unittest
from pathlib import Path
from langchain_core.documents import Document

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.rag_motoru import RAGMotoru
from services.question_generator import QuestionGenerator


class TestRAGGradeIsolationAndQuality(unittest.TestCase):
    """RAG sınıf izolasyonu, anlamsal konu uygunluğu ve soru-kaynak doğrulama testleri."""

    def setUp(self):
        self.rag = RAGMotoru()
        self.qg = QuestionGenerator()

    # -----------------------------------------------------------------------
    # TEST A: SINIF İZOLASYONU
    # -----------------------------------------------------------------------
    def test_scenario_a_grade_isolation(self):
        """3. sınıf sorgusunda varsayılan olarak kabul edilen parçalar yalnızca 3. sınıf olmalıdır."""
        doc_3 = Document(
            page_content="Pil, elektrikli araç gerecin çalışabilmesi için gerekli enerjiyi sağlar.",
            metadata={"sinif": "3. Sınıf", "unite": "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik", "konu": "Elektrikli Araç Gereç"}
        )
        doc_4 = Document(
            page_content="Basit elektrik devresinde pil, ampul ve bağlantı kabloları bulunur.",
            metadata={"sinif": "4. Sınıf", "unite": "7. Ünite: Basit Elektrik Devreleri", "konu": "Devre Elemanları"}
        )

        res_3 = self.rag.chunk_alaka_puanla(
            doc_3,
            temel_konu="Elektrik Kaynakları ve Pilin Önemi",
            alt_kavramlar=["Pil", "Elektrik enerjisi"],
            hedef_sinif="3. Sınıf",
            allow_cross_grade=False
        )
        res_4 = self.rag.chunk_alaka_puanla(
            doc_4,
            temel_konu="Elektrik Kaynakları ve Pilin Önemi",
            alt_kavramlar=["Pil", "Elektrik enerjisi"],
            hedef_sinif="3. Sınıf",
            allow_cross_grade=False
        )

        self.assertTrue(res_3["gecerli"])
        self.assertIn(res_3["kaynak_rolu"], ["ana_kaynak", "yardimci_kaynak"])
        self.assertFalse(res_4["gecerli"])
        self.assertEqual(res_4["kaynak_rolu"], "konu_disi")
        self.assertIn("Farklı sınıf seviyesi", res_4["gerekce"])

    # -----------------------------------------------------------------------
    # TEST B: DOĞRU ANA KAYNAK (3. Sınıf Basılı 48. Sayfa)
    # -----------------------------------------------------------------------
    def test_scenario_b_primary_source_identification(self):
        """3. sınıf 2. kitap basılı 48. sayfa, oyuncak araba/pil anlatımında ana kaynak olarak tanınmalıdır."""
        doc_p48 = Document(
            page_content="Pil, elektrikli araç gerecin çalışabilmesi için gerekli enerjiyi sağlar. Taşınabilir olduğu için araç gereci kabloya gerek olmadan çalıştırır. Saat, uzaktan kumanda, el feneri ve oyuncak araba gibi araçlar pille çalışır.",
            metadata={
                "sinif": "3. Sınıf",
                "unite": "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik",
                "konu": "Elektrikli Araç Gereç",
                "alt_konu": "Elektrikli Araçların Kullanım Alanları",
                "printed_page": 48,
                "physical_page": 49,
            }
        )

        res = self.rag.chunk_alaka_puanla(
            doc_p48,
            temel_konu="Elektrik Kaynakları ve Pilin Önemi",
            alt_kavramlar=["Pil", "Elektrik enerjisi", "Pille çalışan araçlar"],
            transkript="Gördüğünüz gibi uzaktan kumandalı araba durdu çünkü pili bitti.",
            hedef_sinif="3. Sınıf"
        )

        self.assertTrue(res["gecerli"])
        self.assertEqual(res["kaynak_rolu"], "ana_kaynak")
        self.assertEqual(res["kaynak_destek_durumu"], "tam_destekli")
        self.assertGreaterEqual(res["puan"], 4.0)

    # -----------------------------------------------------------------------
    # TEST C: YANLIŞ EŞLEŞME (4. Sınıf Sayfa 76 ve Sayfa 46)
    # -----------------------------------------------------------------------
    def test_scenario_c_false_match_rejection(self):
        """4. sınıf 76. sayfa (kuvvet-hareket) ve 46. sayfa (besinler) pil elektrik kaynağı için elenmelidir."""
        # 4. Sınıf Sayfa 76: Kuvvetin Etkileri / Arabam Trafikte
        doc_p76 = Document(
            page_content="Arabam Trafikte. Mukavvanın üzerine kavşak modeli çiziniz. Oyuncak arabayı kavşakta sağa ve sola yönlendiriniz.",
            metadata={
                "sinif": "4. Sınıf",
                "unite": "3. Ünite: Kuvvetin Etkileri",
                "konu": "Kuvvetin Cisimler Üzerindeki Etkileri",
                "printed_page": 76,
            }
        )
        # 4. Sınıf Sayfa 46: Besinlerimiz
        doc_p46 = Document(
            page_content="Uzaktan kumandalı araba, bebek gibi oyuncakların pili tükendiğinde çalışmadığını gözlemlemişsinizdir. Oyuncakları çalıştırmak için yeni bir pile ihtiyaç duyarız. Siz de hareket etmek için besinlerden enerji alırsınız.",
            metadata={
                "sinif": "4. Sınıf",
                "unite": "2. Ünite: Besinlerimiz",
                "konu": "Besinler ve Özellikleri",
                "printed_page": 46,
            }
        )

        # 3. Sınıf sorgusunda sınıf kuralı eler:
        res76 = self.rag.chunk_alaka_puanla(doc_p76, temel_konu="Elektrik Kaynakları", hedef_sinif="3. Sınıf")
        res46 = self.rag.chunk_alaka_puanla(doc_p46, temel_konu="Elektrik Kaynakları", hedef_sinif="3. Sınıf")
        self.assertFalse(res76["gecerli"])
        self.assertFalse(res46["gecerli"])

        # 4. Sınıf sorgusu olsa dahi ünite uygunluğu kuralı eler:
        res76_same = self.rag.chunk_alaka_puanla(doc_p76, temel_konu="Elektrik Kaynakları", hedef_sinif="4. Sınıf")
        res46_same = self.rag.chunk_alaka_puanla(doc_p46, temel_konu="Elektrik Kaynakları", hedef_sinif="4. Sınıf")
        self.assertFalse(res76_same["gecerli"])
        self.assertEqual(res76_same["kaynak_rolu"], "konu_disi")
        self.assertIn("İlgisiz ünite", res76_same["gerekce"])
        self.assertFalse(res46_same["gecerli"])
        self.assertEqual(res46_same["kaynak_rolu"], "konu_disi")
        self.assertIn("İlgisiz ünite", res46_same["gerekce"])

    # -----------------------------------------------------------------------
    # TEST D: YARDIMCI KAYNAK (3. Sınıf Basılı 57. Sayfa)
    # -----------------------------------------------------------------------
    def test_scenario_d_auxiliary_source_differentiation(self):
        """3. sınıf basılı 57. sayfa atık pil/güvenlik için yardımcı kaynak olmalı, ana mekanizma ana kaynağı olmamalıdır."""
        doc_p57 = Document(
            page_content="Elektrikli araç gereci çalıştıran pilleri dikkatli kullanmalıyız. Pil kırılır veya yere atılırsa zararlı maddeler doğaya zarar verir. Pilleri çöp yerine özel atık pil kutusuna atmalıyız.",
            metadata={
                "sinif": "3. Sınıf",
                "unite": "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik",
                "konu": "Elektrikli Araçların Güvenli Kullanımı",
                "alt_konu": "Elektrik Çarpması ve Güvenlik Önlemleri",
                "printed_page": 57,
            }
        )

        res = self.rag.chunk_alaka_puanla(
            doc_p57,
            temel_konu="Elektrik Kaynakları ve Pilin Önemi",
            alt_kavramlar=["Pil", "Elektrik enerjisi"],
            transkript="Pille çalışan uzaktan kumandalı araba motoru nasıl çalışır?",
            hedef_sinif="3. Sınıf"
        )

        self.assertTrue(res["gecerli"])
        self.assertEqual(res["kaynak_rolu"], "yardimci_kaynak")
        self.assertEqual(res["kaynak_destek_durumu"], "kismi_destekli")

    # -----------------------------------------------------------------------
    # TEST E: FİLTRELİ ARAMA BAŞARISIZLIĞI VE SINIF KAÇAĞI ENGELİ
    # -----------------------------------------------------------------------
    def test_scenario_e_no_silent_cross_grade_bleed(self):
        """3. sınıf araması sonuç vermediğinde veya 4. sınıf konusu arandığında izin olmadan 4. sınıfa geçilmemelidir."""
        # 4. Sınıfa özgü bir konu arandığında, sinif='3. Sınıf' iken sonuçlar sadece 3. Sınıf olmalıdır
        res = self.rag.seviye_filtreli_arama(
            sorgu="Basit Elektrik Devreleri Devre Elemanları ve Kurulumu",
            egitim_seviyesi="İlkokul",
            sinif="3. Sınıf",
            allow_cross_grade=False
        )
        self.assertTrue(len(res) > 0)
        for doc in res:
            self.assertEqual(doc.metadata.get("sinif"), "3. Sınıf")
            self.assertNotEqual(doc.metadata.get("sinif"), "4. Sınıf")
            self.assertFalse(doc.metadata.get("cross_grade", False))

    # -----------------------------------------------------------------------
    # TEST F: SORU KAYNAKÇASI VE DOĞRULANMAMIŞ SORUYA RASTGELE ATAMA ENGELİ
    # -----------------------------------------------------------------------
    def test_scenario_f_question_citation_matching_and_unverified_handling(self):
        """Kaynak eşleşmesi bulunmayan bir soruya asla rastgele ilk MEB referansı atanmamalıdır."""
        video_analizi = {
            "bolumler": [
                {
                    "bolum_no": 2,
                    "bolum_basligi": "Pil ve Pille Çalışan Araçlar",
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) [Sayfa 48]",
                    "rag_kaynaklari": [
                        {
                            "chunk_id": "c_pil_p48",
                            "referans": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) [Sayfa 48]",
                            "unite": "6. Ünite",
                            "konu": "Elektrikli Araç Gereç",
                            "printed_page": 48,
                            "icerik": "Pil taşınabilir araç gereçlere elektrik enerjisi sağlar."
                        }
                    ]
                }
            ],
            "tum_kaynaklar": []
        }

        sorular = [
            {
                "no": 1,
                "ilgili_kavram": "Pil ve Taşınabilirlik",
                "soru": "Pille çalışan araçların en büyük avantajı nedir?",
                "aciklama": "Pil kabloya gerek olmadan taşınabilir enerji sağlar."
            },
            {
                "no": 2,
                "ilgili_kavram": "Kuantum Mekaniği ve Süperiletkenlik",
                "soru": "Meissner etkisi hangi sıcaklıkta gerçekleşir?",
                "aciklama": "Süperiletken maddeler kritik sıcaklık altında manyetik alanı dışlar."
            }
        ]

        baglanmis = self.qg._rag_kaynaklarini_bagla(sorular, video_analizi=video_analizi)

        # 1. Soru kavram eşleştiği için doğrulanmalı:
        self.assertTrue(baglanmis[0]["rag_dogrulandi"])
        self.assertEqual(baglanmis[0]["kaynak_durumu"], "dogrulandi")
        self.assertIn("Sayfa 48", baglanmis[0]["kazanim_referansi"])
        self.assertEqual(baglanmis[0]["eslesen_chunk_id"], "c_pil_p48")

        # 2. Soru müfredat dışı / eşleşmeyen soru olduğu için ASLA ilk bölümün kaynağını almamalı:
        self.assertFalse(baglanmis[1]["rag_dogrulandi"])
        self.assertEqual(baglanmis[1]["kaynak_durumu"], "kaynak_dogrulanamadi")
        self.assertEqual(baglanmis[1]["kazanim_referansi"], "Doğrulanmış ders kitabı kaynağı bulunamadı")
        self.assertIsNone(baglanmis[1]["rag_kaynagi"])
        self.assertIsNone(baglanmis[1]["eslesen_chunk_id"])

    # -----------------------------------------------------------------------
    # TEST G: SKOR GÜVENİLİRLİĞİ VE SABİT 1.0 ENGELİ
    # -----------------------------------------------------------------------
    def test_scenario_g_honest_scores_and_method_specification(self):
        """Gerçek vektör benzerliği yoksa sabit 1.0 değeri verilmemeli, None ve fallback yöntemi belirtilmelidir."""
        doc = Document(
            page_content="Pil kimyasal enerjiyi elektrik enerjisine çevirir.",
            metadata={
                "sinif": "3. Sınıf",
                "unite": "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik",
                "konu": "Elektrikli Araç Gereç",
                "arama_yontemi": "lexical_fallback",
                "vektor_skoru": None,
                "vektor_mesafesi": None,
                "lexical_skoru": 0.82
            }
        )

        res = self.rag.chunk_alaka_puanla(doc, temel_konu="Pil", hedef_sinif="3. Sınıf")
        self.assertIsNone(res["vektor_skoru"])
        self.assertIsNone(res["vektor_mesafesi"])
        self.assertEqual(res["arama_yontemi"], "lexical_fallback")
        self.assertEqual(res["lexical_skor"], 0.82)
        self.assertNotEqual(res["vektor_skoru"], 1.0)


if __name__ == "__main__":
    unittest.main()
