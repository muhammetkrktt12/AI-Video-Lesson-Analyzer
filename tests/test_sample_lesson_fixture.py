"""
tests/test_sample_lesson_fixture.py
Fixture doğrulama ve MEB PDF Ground Truth regresyon testi.
"""

import json
import os
import unittest
from pypdf import PdfReader


class TestSampleLessonFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_path = os.path.join(
            os.path.dirname(__file__), "fixtures", "sample_lesson_fixture.json"
        )
        with open(cls.fixture_path, "r", encoding="utf-8") as f:
            cls.fixture = json.load(f)

        cls.pdf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "reference_books",
            cls.fixture["metadata"]["referans_kitap"],
        )
        if os.path.exists(cls.pdf_path):
            cls.reader = PdfReader(cls.pdf_path)
        else:
            cls.reader = None

    def test_01_fixture_metadata(self):
        """Metadata alanlarının eksiksizliğini doğrular."""
        meta = self.fixture.get("metadata", {})
        self.assertEqual(meta.get("video_id"), "J9_yr257Jog")
        self.assertEqual(meta.get("sinif_duzeyi"), "3. Sınıf")
        self.assertEqual(meta.get("ders_adi"), "Fen Bilimleri")
        self.assertEqual(
            meta.get("referans_kitap"), "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"
        )

    def test_02_bolumler_yapisi(self):
        """Bölümler dizisinin 10 bölüm içerdiğini ve sıralı olduğunu doğrular."""
        bolumler = self.fixture.get("bolumler", [])
        self.assertEqual(len(bolumler), 10, "Bölüm sayısı tam 10 olmalıdır.")

        for i, b in enumerate(bolumler, start=1):
            self.assertEqual(b["bolum_no"], i)
            self.assertTrue(len(b["transkript"].strip()) > 20)
            self.assertIn("zaman_araligi", b)
            self.assertIn("kare_dosyasi", b)

    def test_03_ground_truth_kumesi_genel(self):
        """12 adet değerlendirme iddiasının schema uyumunu test eder."""
        gt_list = self.fixture.get("ground_truth_degerlendirme_kumesi", [])
        self.assertEqual(
            len(gt_list), 12, "Ground truth değerlendirme kümesi 12 iddia olmalıdır."
        )

        for gt in gt_list:
            self.assertIn(
                gt["dogrulama_durumu"],
                ["DESTEKLENIYOR", "CELISIYOR", "YETERSIZ_KANIT"],
            )
            self.assertTrue(1 <= gt["bolum_no"] <= 10)
            self.assertTrue(len(gt["transkript_iddiasi"]) > 10)

    def test_04_desteklenen_iddialarin_pdf_dogrulamasi(self):
        """DESTEKLENİYOR statüsündeki iddiaların gerçek PDF sayfalarında yer aldığını doğrular."""
        if self.reader is None:
            self.skipTest("Referans PDF dosyası bulunamadı.")

        gt_list = self.fixture.get("ground_truth_degerlendirme_kumesi", [])
        desteklenenler = [
            gt for gt in gt_list if gt["dogrulama_durumu"] == "DESTEKLENIYOR"
        ]
        self.assertEqual(len(desteklenenler), 10)

        for gt in desteklenenler:
            fiz_sayfa = gt["beklenen_fiziksel_sayfa"]
            self.assertIsNotNone(fiz_sayfa, f"{gt['id']} fiziksel sayfa null olamaz.")

            # pypdf 0-indexed sayfa alır: fiz_sayfa - 1
            page_text = self.reader.pages[fiz_sayfa - 1].extract_text()
            page_text_clean = " ".join(page_text.split()).lower()

            # Gerçek alıntıdan anahtar kelimeleri doğrula
            alinti = gt["gercek_kitap_alintisi"]
            # En azından alıntıdaki ilk 3 kelimenin sayfada geçtiğini kontrol et
            ilk_kelimeler = [
                k.lower() for k in alinti.replace(":", " ").replace(".", " ").split()[:4]
            ]
            bulundu = any(k in page_text_clean for k in ilk_kelimeler)
            self.assertTrue(
                bulundu,
                f"{gt['id']} için alıntı PDF sayfa {fiz_sayfa} içinde bulunamadı!",
            )

    def test_05_yetersiz_kanit_iddialari(self):
        """Halüsinasyon engelleyici (YETERSIZ_KANIT) iddiaların sayfa içermediğini doğrular."""
        gt_list = self.fixture.get("ground_truth_degerlendirme_kumesi", [])
        yetersizler = [
            gt for gt in gt_list if gt["dogrulama_durumu"] == "YETERSIZ_KANIT"
        ]
        self.assertEqual(
            len(yetersizler), 2, "2 adet YETERSIZ_KANIT kontrol iddiası bulunmalıdır."
        )

        for gt in yetersizler:
            self.assertIsNone(
                gt["beklenen_basili_sayfa"],
                f"{gt['id']} basılı sayfa referansı içermemelidir.",
            )
            self.assertIsNone(
                gt["beklenen_fiziksel_sayfa"],
                f"{gt['id']} fiziksel sayfa referansı içermemelidir.",
            )


if __name__ == "__main__":
    unittest.main()
