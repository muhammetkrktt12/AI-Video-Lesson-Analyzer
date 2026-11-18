"""
tests/test_sample_lesson_fixture.py
Mevcut sample_lesson_fixture.json dosyasına özel doğrulama ve MEB PDF Ground Truth regresyon testi.
NOT: Bu testler yalnızca 'sample_lesson_fixture.json' sabit veri kümesine özeldir;
gerçek kullanıcı videolarının bölüm sayısı, süresi veya iddia sayısı için sabit bir kısıt teşkil etmez.
"""

import json
import os
import re
import unittest
from pypdf import PdfReader


def normalize_text_for_comparison(text: str) -> str:
    """Türkçe karakter ve noktalama işaretlerini eşleştirme için normalize eder."""
    t = text.replace("İ", "i").replace("I", "ı").lower()
    return " ".join(re.findall(r"[a-zçğıöşü0-9]+", t))


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

    def test_01_fixture_metadata_and_provenance(self):
        """Bu fixture'ın kökenini ve gerekli metadata alanlarını doğrular."""
        meta = self.fixture.get("metadata", {})
        self.assertEqual(meta.get("video_id"), "J9_yr257Jog")
        self.assertEqual(meta.get("sinif_duzeyi"), "3. Sınıf")
        self.assertEqual(meta.get("ders_adi"), "Fen Bilimleri")
        self.assertEqual(
            meta.get("referans_kitap"), "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"
        )
        # Provenance: Canlı analiz kaydı değil simülasyon verisi olarak açıkça etiketlenmeli
        self.assertEqual(meta.get("veri_kaynagi_turu"), "pedagojik_simulasyon_verisi")

    def test_02_bolumler_ve_kare_tutarliligi(self):
        """Mevcut fixture bölümlerinde kare_zamani ile kare_dosyasi dosya adının tam eşleştiğini doğrular."""
        bolumler = self.fixture.get("bolumler", [])
        self.assertTrue(len(bolumler) > 0, "Fixture bölümler listesi boş olamaz.")

        for b in bolumler:
            kare_yolu = b.get("kare_dosyasi", "")
            z_info = b.get("zaman_araligi", {})
            kare_zamani = z_info.get("kare_zamani")
            self.assertIsNotNone(kare_zamani, f"Bölüm {b['bolum_no']} kare_zamani eksik.")

            # Dosya adındaki saniye değeri ile kare_zamani tam örtüşmelidir: bolum_XX_YYYYs.jpg
            beklenen_dosya_adi = f"bolum_{b['bolum_no']:02d}_{int(kare_zamani):04d}s.jpg"
            self.assertTrue(
                kare_yolu.endswith(beklenen_dosya_adi),
                f"Bölüm {b['bolum_no']}: kare_dosyasi ({kare_yolu}) ile kare_zamani ({kare_zamani}s) uyuşmuyor! Beklenen: {beklenen_dosya_adi}"
            )

    def test_03_transkript_iddialarinin_metin_uyumu(self):
        """Transkript doğrulama kümesindeki iddiaların transkriptte birebir geçtiğini doğrular."""
        bolumler = {b["bolum_no"]: b["transkript"] for b in self.fixture.get("bolumler", [])}
        tr_kumesi = self.fixture.get("transkript_dogrulama_kumesi", [])
        self.assertTrue(len(tr_kumesi) > 0, "Transkript doğrulama kümesi boş olamaz.")

        for gt in tr_kumesi:
            b_no = gt["bolum_no"]
            self.assertIn(b_no, bolumler, f"{gt['id']} için bölüm {b_no} bulunamadı.")
            tr = bolumler[b_no]

            # transkript_cumlesi transkript metninde birebir alt dize (substring) olmalıdır
            tr_cumlesi = gt.get("transkript_cumlesi", "")
            self.assertTrue(
                len(tr_cumlesi) > 5,
                f"{gt['id']} transkript_cumlesi boş veya çok kısa olamaz."
            )
            self.assertIn(
                tr_cumlesi,
                tr,
                f"{gt['id']} için belirtilen transkript_cumlesi transkript içinde bulunamadı!"
            )

    def test_04_desteklenen_iddialarin_guclu_pdf_dogrulamasi(self):
        """DESTEKLENİYOR statüsündeki iddiaların gerçek PDF sayfalarında yer aldığını GÜÇLÜ metin eşleştirmesiyle doğrular."""
        if self.reader is None:
            self.skipTest("Referans PDF dosyası bulunamadı.")

        # Hem transkript hem kitap kümesindeki desteklenen iddiaları topla
        tum_desteklenenler = [
            gt for gt in self.fixture.get("transkript_dogrulama_kumesi", [])
            if gt["dogrulama_durumu"] == "DESTEKLENIYOR"
        ] + [
            bk for bk in self.fixture.get("kitap_kaynak_bulma_kumesi", [])
            if bk["dogrulama_durumu"] == "DESTEKLENIYOR"
        ]

        self.assertTrue(len(tum_desteklenenler) >= 10)

        for item in tum_desteklenenler:
            fiz_sayfa = item["beklenen_fiziksel_sayfa"]
            basili_sayfa = item["beklenen_basili_sayfa"]
            self.assertIsNotNone(fiz_sayfa, f"{item['id']} fiziksel sayfa null olamaz.")
            self.assertIsNotNone(basili_sayfa, f"{item['id']} basılı sayfa null olamaz.")

            # pypdf 0-indexed sayfa alır: fiz_sayfa - 1
            raw_page_text = self.reader.pages[fiz_sayfa - 1].extract_text() or ""
            page_text_clean = normalize_text_for_comparison(raw_page_text)

            # 1. Basılı sayfa numarasının o sayfada geçtiğini doğrula
            self.assertIn(
                str(basili_sayfa),
                raw_page_text,
                f"{item['id']}: Basılı sayfa numarası {basili_sayfa}, fiziksel sayfa {fiz_sayfa} içinde bulunamadı!"
            )

            # 2. Güçlü Metin Doğrulaması: Alıntıdan en az 5 ardışık kelimelik bir öbeğin sayfada birebir bulunması
            alinti_clean = normalize_text_for_comparison(item["gercek_kitap_alintisi"])
            alinti_words = alinti_clean.split()
            self.assertTrue(
                len(alinti_words) >= 5,
                f"{item['id']} alıntısı en az 5 kelimeden oluşmalıdır."
            )

            # 5'li ardışık pencere (5-gram) ile sayfada tam eşleşme ara
            bulunan_5gram = False
            for start in range(len(alinti_words) - 4):
                pencere = " ".join(alinti_words[start : start + 5])
                if pencere in page_text_clean:
                    bulunan_5gram = True
                    break

            self.assertTrue(
                bulunan_5gram,
                f"{item['id']} için 5 ardışık kelimelik tam metin eşleşmesi PDF sayfa {fiz_sayfa} içinde bulunamadı! (Alıntı: '{item['gercek_kitap_alintisi'][:60]}...')"
            )

            # 3. Alıntıdaki tüm kelimelerin en az %80'inin sayfada geçtiğini teyit et
            kelime_eslesme_orani = sum(1 for w in alinti_words if w in page_text_clean) / len(alinti_words)
            self.assertGreaterEqual(
                kelime_eslesme_orani,
                0.80,
                f"{item['id']} kelime kapsama oranı yetersiz: {kelime_eslesme_orani:.2f}"
            )

    def test_05_yetersiz_kanit_ve_sinir_kontrolleri(self):
        """Halüsinasyon engelleyici (YETERSIZ_KANIT) iddiaların sayfa içermediğini doğrular."""
        tr_kumesi = self.fixture.get("transkript_dogrulama_kumesi", [])
        yetersizler = [
            gt for gt in tr_kumesi if gt["dogrulama_durumu"] == "YETERSIZ_KANIT"
        ]
        self.assertEqual(
            len(yetersizler), 2, "Fixture içinde 2 adet YETERSIZ_KANIT sınır iddiası bulunmalıdır."
        )

        for gt in yetersizler:
            self.assertIsNone(
                gt["beklenen_basili_sayfa"],
                f"{gt['id']} basılı sayfa referansı içermemelidir."
            )
            self.assertIsNone(
                gt["beklenen_fiziksel_sayfa"],
                f"{gt['id']} fiziksel sayfa referansı içermemelidir."
            )


if __name__ == "__main__":
    unittest.main()
