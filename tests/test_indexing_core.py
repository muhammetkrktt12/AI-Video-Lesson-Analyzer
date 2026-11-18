"""
tests/test_indexing_core.py
Ortak indeksleme çekirdeği (index_core.py) birim ve bütünlük testleri.
ChromaDB'ye yazma işlemi yapmadan; dosya hash'i, chunk ID üretimi, sayfa ofseti,
manifesto eşleştirmesi ve pedagojik parçalama mantığını bağımsız test eder.
"""

import os
import unittest
import pypdf
from pathlib import Path

from config import REFERENCE_BOOKS_DIR
from index_core import (
    calculate_file_hash,
    generate_chunk_id,
    detect_page_offset,
    get_book_metadata_and_unit,
    create_pedagogical_splitter,
    count_tokens,
    PRIMARY_BOOK_MANIFESTS,
)


class TestIndexingCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book_2_path = REFERENCE_BOOKS_DIR / "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"

    def test_01_file_hash_and_chunk_id_determinism(self):
        """SHA-256 hash ve Chunk ID üretiminin deterministik olduğunu doğrular."""
        if not self.book_2_path.exists():
            self.skipTest("Referans kitap bulunamadı.")

        h1 = calculate_file_hash(self.book_2_path)
        h2 = calculate_file_hash(self.book_2_path)
        self.assertEqual(h1, h2, "Dosya hash'i deterministik olmalıdır.")
        self.assertEqual(len(h1), 64, "SHA-256 hash 64 karakter olmalıdır.")

        cid = generate_chunk_id(self.book_2_path.name, h1, 48, 2)
        expected_cid = f"{self.book_2_path.name}:{h1[:12]}:p0048:c002"
        self.assertEqual(cid, expected_cid)

    def test_02_detect_page_offset(self):
        """3. Sınıf 2. Kitap için sayfa ofsetinin +1 olduğunu doğrular."""
        if not self.book_2_path.exists():
            self.skipTest("Referans kitap bulunamadı.")

        reader = pypdf.PdfReader(str(self.book_2_path))
        offset = detect_page_offset(reader, max_scan_pages=25)
        self.assertEqual(offset, 1, "3. Sınıf 2. Kitap sayfa ofseti +1 olmalıdır.")

    def test_03_primary_book_manifest_metadata(self):
        """3 seviyeli kitap manifestosunun ünite, konu ve alt konu eşleştirmesini doğrular."""
        fname = "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"

        # Sayfa 48: 6. Ünite, Elektrikli Araç Gereç
        meta, unite, konu, alt_konu = get_book_metadata_and_unit(
            filename=fname, phys_page_no=49, printed_page_no=48, raw_text="test"
        )
        self.assertEqual(meta["sinif"], "3. Sınıf")
        self.assertEqual(meta["ders_adi"], "Fen Bilimleri")
        self.assertEqual(meta["egitim_seviyesi"], "İlkokul")
        self.assertIn("Yaşamımızı Kolaylaştıran Elektrik", unite)
        self.assertEqual(konu, "Elektrikli Araç Gereç")
        self.assertEqual(alt_konu, "Elektrikli Araçların Kullanım Alanları")

        # Sayfa 55: Elektrikli Araçların Güvenli Kullanımı
        meta, unite, konu, alt_konu = get_book_metadata_and_unit(
            filename=fname, phys_page_no=56, printed_page_no=55, raw_text="test"
        )
        self.assertEqual(konu, "Elektrikli Araçların Güvenli Kullanımı")

        # Sayfa 65: Elektriğin Tasarruflu Kullanımı
        meta, unite, konu, alt_konu = get_book_metadata_and_unit(
            filename=fname, phys_page_no=66, printed_page_no=65, raw_text="test"
        )
        self.assertEqual(konu, "Elektriğin Tasarruflu Kullanımı")

    def test_04_unknown_book_raises_error(self):
        """Manifestoda tanımlı olmayan bilinmeyen kitaplarda sessizce varsayılan atamak yerine hata verir."""
        with self.assertRaises(ValueError):
            get_book_metadata_and_unit(
                filename="bilinmeyen_kitap_99.pdf",
                phys_page_no=10,
                printed_page_no=10,
                raw_text="test",
            )

    def test_05_pedagogical_splitter(self):
        """Pedagojik metin parçalayıcının cümle bütünlüğünü koruduğunu doğrular."""
        splitter = create_pedagogical_splitter(chunk_size=300, chunk_overlap=50)
        sample_text = (
            "Elektrikli araçlar hayatımızı kolaylaştırır. "
            "Buzdolabı besinleri korur. Çamaşır makinesi kıyafetleri yıkar. "
            "Ütü elbiselerin kırışıklıklarını açar. "
            "Elektrik enerjisi elektrik santrallerinde üretilir. "
            "Gereksiz yanan lambaları söndürerek enerji tasarrufu yapmalıyız."
        )
        chunks = splitter.split_text(sample_text)
        self.assertTrue(len(chunks) >= 1)
        for c in chunks:
            self.assertTrue(len(c) <= 350)
            self.assertTrue(len(c.strip()) > 10)

    def test_06_token_counter(self):
        """Token sayıcının pozitif tamsayı ürettiğini doğrular."""
        text = "Elektrikli araç gereçlerin kullanım alanları ve enerji tasarrufu."
        tokens = count_tokens(text)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)


if __name__ == "__main__":
    unittest.main()
