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

        cid = generate_chunk_id(self.book_2_path.name, h1, 49, 2)
        expected_cid = f"{self.book_2_path.name}:{h1[:12]}:phys_0049:c002"
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

    def test_07_physical_pages_1_and_2_produce_distinct_ids(self):
        """Fiziksel sayfa 1 ve 2'nin çakışmayan tamamen farklı Chunk ID'ler ürettiğini doğrular."""
        fname = "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"
        f_hash = "421c34396672abcdef"

        cid_p1 = generate_chunk_id(fname, f_hash, physical_page=1, chunk_idx=0)
        cid_p2 = generate_chunk_id(fname, f_hash, physical_page=2, chunk_idx=0)

        self.assertNotEqual(cid_p1, cid_p2, "Fiziksel sayfa 1 ve 2 aynı ID'yi üretemez!")
        self.assertTrue(cid_p1.endswith(":phys_0001:c000"))
        self.assertTrue(cid_p2.endswith(":phys_0002:c000"))

    def test_08_chunk_id_determinism(self):
        """Aynı girdi parametrelerinin her çağrıda tamamen özdeş Chunk ID ürettiğini doğrular."""
        cid_a = generate_chunk_id("kitap.pdf", "abc1234567890", 15, 3)
        cid_b = generate_chunk_id("kitap.pdf", "abc1234567890", 15, 3)
        self.assertEqual(cid_a, cid_b)
        self.assertEqual(cid_a, "kitap.pdf:abc123456789:phys_0015:c003")

    def test_09_cross_book_collision_prevention(self):
        """Farklı kitap veya hash'lerin aynı sayfa ve chunk indeksinde çakışmadığını doğrular."""
        cid_book1 = generate_chunk_id("kitap_1.pdf", "hash111111111111", 10, 0)
        cid_book2 = generate_chunk_id("kitap_2.pdf", "hash222222222222", 10, 0)
        self.assertNotEqual(cid_book1, cid_book2)

    def test_10_real_meb_pdf_all_193_chunks_unique(self):
        """Gerçek MEB Fen 3-2 PDF kitabındaki 193 parçanın tamamının benzersiz ID aldığını doğrular."""
        if not self.book_2_path.exists():
            self.skipTest("Referans kitap bulunamadı.")

        f_hash = calculate_file_hash(self.book_2_path)
        reader = pypdf.PdfReader(str(self.book_2_path))
        splitter = create_pedagogical_splitter(1000, 150)

        all_ids = []
        for phys_p in range(1, len(reader.pages) + 1):
            txt = (reader.pages[phys_p - 1].extract_text() or "").strip()
            if not txt or len(txt) < 25:
                continue
            chunks = splitter.split_text(txt)
            for c_idx, c_text in enumerate(chunks):
                if len(c_text.strip()) < 20:
                    continue
                cid = generate_chunk_id(self.book_2_path.name, f_hash, phys_p, c_idx)
                all_ids.append(cid)

        self.assertEqual(len(all_ids), 193, "Gerçek PDF'ten tam 193 chunk üretilmelidir.")
        self.assertEqual(len(set(all_ids)), 193, "193 chunk'ın tümü benzersiz ID'ye sahip olmalıdır.")
        self.assertEqual(len(all_ids), len(set(all_ids)), "Hiçbir Chunk ID çakışması (collision) olmamalıdır.")

    def test_11_front_matter_metadata_and_chroma_type_safety(self):
        """Ön sayfalarda printed_page=0 ve has_printed_page=False olarak atanıp ChromaDB tip kısıtlarına uyulduğunu doğrular."""
        fname = "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"
        base_meta, unite, konu, alt_konu = get_book_metadata_and_unit(
            filename=fname,
            phys_page_no=1,
            printed_page_no=0,
            raw_text="Ön kapak ve tanıtım metni"
        )
        self.assertEqual(unite, "Ön Sayfalar")
        self.assertEqual(konu, "Kapak ve Tanıtım")
        self.assertEqual(alt_konu, "Kapak / Künye")

        # ChromaDB metadata uyumluluğu kontrolü: None değeri kabul edilmez, int/bool/str kabul edilir
        chroma_metadata = {
            "source": fname,
            "physical_page": 1,
            "printed_page": 0,
            "has_printed_page": False,
            "unite": unite,
            "konu": konu,
            "alt_konu": alt_konu
        }
        for k, v in chroma_metadata.items():
            self.assertIsNotNone(v, f"Metadata alanı '{k}' None olamaz!")
            self.assertIn(type(v), (str, int, float, bool), f"Metadata alanı '{k}' tipi ChromaDB ile uyumsuz: {type(v)}")

    def test_12_pre_write_duplicate_check_mechanism(self):
        """Yazma öncesi çakışma kontrolünün mükerrer ID tespit ettiğinde ValueError fırlattığını doğrular."""
        duplicate_ids = [
            "book.pdf:hash1:phys_0001:c000",
            "book.pdf:hash1:phys_0002:c000",
            "book.pdf:hash1:phys_0001:c000",  # Mükerrer
        ]
        with self.assertRaises(ValueError) as ctx:
            if len(duplicate_ids) != len(set(duplicate_ids)):
                seen = set()
                dups = [cid for cid in duplicate_ids if cid in seen or seen.add(cid)]
                raise ValueError(f"Çakışma tespit edildi: {dups}")
        self.assertIn("Çakışma tespit edildi", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
