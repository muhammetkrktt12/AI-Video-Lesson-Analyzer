"""
EduFrame AI - MEB Referans Kitaplarını Akıllı Metadata ile İndeksleme Aracı
-------------------------------------------------------------------------
Bu script, data/reference_books dizinindeki tüm MEB ders kitaplarını okur,
dosya adından seviye/ders/sınıf bilgilerini, sayfalardan ise Ünite ve Konu
başlıklarını çıkararak nokta atışı sayfa referansları ile ChromaDB'ye kaydeder.
"""

import os
import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pypdf
import numpy as np

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
BOOKS_DIR = DATA_DIR / "reference_books"
COLLECTION_NAME = "eduframe_meb_v2"


class FastTurkishEmbeddings(Embeddings):
    """384 boyutlu, deterministik ve Türkçe morfolojisine duyarlı dense embedding motoru."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        norm_text = text.replace("İ", "i").replace("I", "ı").lower()
        words = re.findall(r"[a-zçğıöşü0-9]+", norm_text)
        if not words:
            return vector.tolist()

        for w in words:
            # Kelime kökü hash'i
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            vector[h % self.dim] += 2.5

            # Türkçe ek toleransı için karakter 3-gramları
            w_pad = f"^{w}$"
            for i in range(len(w_pad) - 2):
                g = w_pad[i:i+3]
                hg = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
                vector[hg % self.dim] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def parse_filename(filename: str) -> Dict[str, str]:
    """Dosya adından seviye, sınıf, ders ve kitap numarasını çıkarır."""
    fn = filename.lower()
    
    # Seviye
    if "ilkokul" in fn:
        seviye = "İlkokul"
    elif "ortaokul" in fn:
        seviye = "Ortaokul"
    elif "lise" in fn:
        seviye = "Lise"
    else:
        seviye = "İlkokul"

    # Ders
    if "fizik" in fn:
        ders = "Fizik"
    elif "kimya" in fn:
        ders = "Kimya"
    elif "biyoloji" in fn:
        ders = "Biyoloji"
    elif "fenbilimleri" in fn or "fen" in fn:
        ders = "Fen Bilimleri"
    elif "matematik" in fn:
        ders = "Matematik"
    else:
        ders = "Fen Bilimleri"

    # Sınıf
    sm = re.search(r"(\d+)\.?\s*sinif", fn)
    sinif = f"{sm.group(1)}. Sınıf" if sm else "3. Sınıf"

    # Kitap No
    km = re.search(r"(\d+)\.?\s*kitap", fn)
    kitap_no = f"{km.group(1)}. Kitap" if km else ""

    # Temiz Kitap Adı
    kitap_adi_parcalar = ["MEB", sinif, ders]
    if kitap_no:
        kitap_adi_parcalar.append(f"({kitap_no})")
    kitap_adi = " ".join(kitap_adi_parcalar)

    return {
        "egitim_seviyesi": seviye,
        "ders_adi": ders,
        "sinif": sinif,
        "kitap_no": kitap_no,
        "kitap_adi": kitap_adi,
        "source": filename
    }


def parse_toc_structure(reader: pypdf.PdfReader) -> List[Tuple[int, str, str]]:
    """
    Kitabın İçindekiler bölümünden ünite ve konu başlıklarının sayfa başlangıçlarını çıkarır.
    Dönüş: [(baslangic_sayfasi, unite_adi, konu_adi), ...]
    """
    toc_entries = []
    # İlk 15 sayfada İÇİNDEKİLER ara
    toc_text = ""
    for p_no in range(1, min(16, len(reader.pages)) + 1):
        txt = reader.pages[p_no - 1].extract_text() or ""
        if "İÇİNDEKİLER" in txt.upper():
            toc_text += "\n" + txt
            if p_no < len(reader.pages):
                next_txt = reader.pages[p_no].extract_text() or ""
                if any(x in next_txt for x in [". ÜNİTE", "ÜNİTE", ".....", "BÖLÜM"]):
                    toc_text += "\n" + next_txt
            break

    if not toc_text:
        return []

    lines = toc_text.split("\n")
    current_unit = "Genel Ünite"

    for line in lines:
        line = line.strip()
        if not line or "İÇİNDEKİLER" in line.upper():
            continue

        # Sayfa numarasını yakala: Örn: "YAŞAMIMIZI KOLAYLAŞTIRAN ELEKTRİK ..... 41" veya "ELEKTRİKLİ ARAÇ GEREÇ ... 44"
        match = re.search(r"^(.+?)[.\s\-_]{3,}\s*(\d+)\s*$", line)
        if match:
            baslik = match.group(1).strip()
            sayfa = int(match.group(2))
            
            # Ünite mi?
            if "ÜNİTE" in baslik.upper() or any(u in baslik.upper() for u in ["ELEKTRİK", "HAREKET", "TOPRAK", "CANLILAR", "MADDE", "KUVVET"]):
                current_unit = baslik
                toc_entries.append((sayfa, current_unit, baslik))
            else:
                toc_entries.append((sayfa, current_unit, baslik))

    toc_entries.sort(key=lambda x: x[0])
    return toc_entries


def get_unit_and_topic_for_page(
    page_no: int,
    page_text: str,
    toc_entries: List[Tuple[int, str, str]],
    default_unit: str = "Genel Ünite"
) -> Tuple[str, str]:
    """Sayfa metninden veya TOC yapısından Ünite ve Konu başlığını belirler."""
    # 1. Sayfa metninde doğrudan belirgin ünite başlığı var mı?
    unit_match = re.search(r"(\d+\.\s*ÜNİTE[^\n\r]{0,40})", page_text, re.IGNORECASE)
    page_unit = unit_match.group(1).strip() if unit_match else None

    # 2. TOC haritasından sayfanın ait olduğu aralığı bul
    toc_unit = default_unit
    toc_topic = "Genel Konu"
    if toc_entries:
        for p_start, u_name, t_name in toc_entries:
            if page_no >= p_start:
                toc_unit = u_name
                toc_topic = t_name
            else:
                break

    unit = page_unit if page_unit else toc_unit

    # 3. Özel bilinen kitap haritası (3. Sınıf 2. Kitap kesin doğrulaması)
    is_3_2 = "3.sinif_2.kitap" in default_unit.lower() or "3. sinif (2. kitap)" in default_unit.lower()
    if is_3_2 and (41 <= page_no <= 72):
        unit = "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik"
        if 44 <= page_no <= 51:
            toc_topic = "Elektrikli Araç Gereç"
        elif 52 <= page_no <= 60:
            toc_topic = "Elektrikli Araç Gerecin Güvenli Kullanımı"
        elif 61 <= page_no <= 72:
            toc_topic = "Elektriğin Tasarruflu Kullanımı"

    return unit, toc_topic


def index_book(pdf_path: Path, store: Chroma, splitter: RecursiveCharacterTextSplitter) -> int:
    """Tek bir PDF kitabını okur, chunk'lara böler ve zengin metadata ile Chroma'ya ekler."""
    meta_base = parse_filename(pdf_path.name)
    reader = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    
    print(f"\n[İndeksleme Başladı] {pdf_path.name} ({total_pages} sayfa)...")
    print(f" -> Seviye: {meta_base['egitim_seviyesi']} | Ders: {meta_base['ders_adi']} | Sınıf: {meta_base['sinif']}")

    toc_entries = parse_toc_structure(reader)
    if toc_entries:
        print(f" -> İçindekiler tablosundan {len(toc_entries)} başlık çıkarıldı.")

    documents = []
    chunk_counter = 0

    for page_no, page in enumerate(reader.pages, 1):
        try:
            raw_text = (page.extract_text() or "").strip()
        except Exception:
            continue
        if not raw_text or len(raw_text) < 25:
            continue

        unite, konu = get_unit_and_topic_for_page(
            page_no=page_no,
            page_text=raw_text,
            toc_entries=toc_entries,
            default_unit=f"{meta_base['kitap_adi']} İçeriği"
        )

        referans_metni = f"{meta_base['kitap_adi']} - {unite}, '{konu}' (Sayfa {page_no})"

        # Sayfa içeriğini parçala
        chunks = splitter.split_text(raw_text)
        for c_idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 20:
                continue

            chunk_id = f"{pdf_path.name}:p{page_no:03d}:c{c_idx:02d}"
            c_meta = {
                "source": pdf_path.name,
                "kitap_adi": meta_base["kitap_adi"],
                "egitim_seviyesi": meta_base["egitim_seviyesi"],
                "ders_adi": meta_base["ders_adi"],
                "sinif": meta_base["sinif"],
                "kitap_no": meta_base["kitap_no"],
                "page": page_no,
                "sayfa_no": page_no,
                "unite": unite,
                "konu": konu,
                "referans": referans_metni,
                "chunk_id": chunk_id,
            }
            documents.append(Document(page_content=chunk_text, metadata=c_meta))
            chunk_counter += 1

    # Toplu ChromaDB'ye ekle
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        ids = [d.metadata["chunk_id"] for d in batch]
        store.add_documents(batch, ids=ids)

    print(f"[BAŞARILI] {pdf_path.name} tamamlandı: {len(documents)} parça ChromaDB'ye eklendi.")
    return len(documents)


def main():
    print("=" * 65)
    print("📚 EduFrame AI - MEB Referans Kitapları Vektörleştirme Motoru")
    print(f" -> Koleksiyon: {COLLECTION_NAME}")
    print(f" -> Hedef Dizin: {CHROMA_DIR}")
    print("=" * 65)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = FastTurkishEmbeddings(dim=384)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    all_books = sorted(list(BOOKS_DIR.glob("*.pdf")))
    print(f"Toplam {len(all_books)} kitap tespit edildi.\n")

    # Öncelikli olarak İlkokul kitaplarını, ardından Ortaokul ve Lise kitaplarını indeksle
    ilkokul_books = [b for b in all_books if "ilkokul" in b.name.lower()]
    diger_books = [b for b in all_books if "ilkokul" not in b.name.lower()]

    sirali_kitaplar = ilkokul_books + diger_books

    toplam_eklenen = 0
    t_start = time.time()

    for idx, b_path in enumerate(sirali_kitaplar, 1):
        # Koleksiyonda var mı kontrol et
        try:
            var_mi = store._collection.get(where={"source": b_path.name}, limit=1)
            if var_mi and len(var_mi.get("ids", [])) > 0:
                print(f"[{idx}/{len(sirali_kitaplar)}] ZATEN MEVCUT: {b_path.name}")
                continue
        except Exception:
            pass

        print(f"\n--- [{idx}/{len(sirali_kitaplar)}] Kitap İşleniyor: {b_path.name} ---")
        cnt = index_book(b_path, store, splitter)
        toplam_eklenen += cnt

    sure = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"🎉 TÜM KİTAPLAR İNDEKSLENDİ!")
    print(f" -> Toplam Eklenen Parça: {toplam_eklenen}")
    print(f" -> Koleksiyon Toplamı  : {store._collection.count()} parça")
    print(f" -> Geçen Süre          : {sure:.1f} saniye")
    print("=" * 65)


if __name__ == "__main__":
    main()
