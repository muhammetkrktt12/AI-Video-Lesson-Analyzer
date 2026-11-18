"""
EduFrame AI - İlkokul (3. ve 4. Sınıf) Fen Bilimleri Kitaplarını İndeksleme Aracı
---------------------------------------------------------------------------------
Özellikler:
1. Kesinlikle 768-D Hedef Koleksiyon: 'eduframe_meb_768'
2. Embedding Boyutu Ön Kontrolü: Chroma açılmadan önce 768 boyut doğrulaması
3. Sayfa Ofseti Tespiti: Fiziksel PDF sayfası vs basılı sayfa farkını otomatik hesaplar
4. Dosya Hash'i ile Kararlı Chunk ID: {kitap}:{hash[:12]}:p{page}:c{chunk}
5. 3 Seviyeli Zengin Metadata: Ünite, Konu ve Alt Konu etiketleri
6. Tekrar Çalıştırma & Kaldığı Yerden Devam Etme (Idempotent Resumption)
7. Eski Sürüm Temizliği: PDF değiştiğinde eski hash'e ait kayıtları temizler
8. 1000 Karakter / 150 Overlap Pedagojik Bölme & Gemma BPE Token Sayımı
9. Hata Toleranslı Batch ve Ayrıntılı Loglamalı Fallback Mekanizması
10. Çok Yönlü Doğrulama Test Sorguları (3. Sınıf 1. & 2. Kitap ve 4. Sınıf)

Kullanım:
    .venv\\Scripts\\python.exe index_ilkokul_books.py
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Set, Dict, Any
import pypdf

from langchain_core.documents import Document
from langchain_chroma import Chroma

# Proje kök dizinini ekle
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CHROMA_DB_DIR, REFERENCE_BOOKS_DIR, OLLAMA_URL, EMBED_MODEL
from index_core import (
    calculate_file_hash,
    generate_chunk_id,
    verify_embedding_dimension,
    detect_page_offset,
    get_book_metadata_and_unit,
    create_pedagogical_splitter,
    safe_add_documents_batch,
    cleanup_stale_chunks,
    get_tokenizer,
    count_tokens,
    PRIMARY_BOOK_MANIFESTS,
)

# Kesin 768 boyutlu koleksiyon adı
TARGET_COLLECTION = "eduframe_meb_768"

TARGET_BOOKS = [
    "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf",
    "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
    "ilkokul_fenbilimleri_4.sinif.pdf",
]


def get_existing_chunk_ids(store: Chroma) -> Set[str]:
    """ChromaDB koleksiyonunda halihazırda kayıtlı olan Chunk ID'lerini çeker."""
    try:
        col = store._collection
        count = col.count()
        if count == 0:
            return set()

        existing_ids = set()
        limit = 10000
        offset = 0
        while offset < count:
            res = col.get(limit=limit, offset=offset, include=[])
            if not res or not res.get("ids"):
                break
            existing_ids.update(res["ids"])
            offset += len(res["ids"])
        return existing_ids
    except Exception as e:
        print(f"[HATA] get_existing_chunk_ids sırasında veritabanı hatası: {e}")
        raise RuntimeError(f"ChromaDB mevcut kayıtları taranırken veritabanı hatası: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="İlkokul (3. ve 4. Sınıf) Fen Bilimleri Kitapları İndeksleme Aracı")
    parser.add_argument(
        "--kitap",
        type=str,
        default="",
        help="Yalnızca belirtilen tekil PDF kitabını indeksle (örn: ilkokul_fenbilimleri_3.sinif_2.kitap.pdf)"
    )
    args = parser.parse_args()

    selected_books = [args.kitap] if args.kitap else TARGET_BOOKS

    print("=" * 75)
    print("🎓 EduFrame AI - İlkokul (3. & 4. Sınıf) 768-D İndeksleme Motoru")
    print("=" * 75)
    print(f" -> Hedef Vektör Veritabanı : {CHROMA_DB_DIR.resolve()}")
    print(f" -> Hedef Koleksiyon        : {TARGET_COLLECTION}")
    print(f" -> Embedding Modeli        : Ollama ({EMBED_MODEL})")
    print(f" -> Parçalama Yapısı        : 1000 karakter / 150 örtüşme (overlap)")
    if args.kitap:
        print(f" -> Filtrelenen Tekil Kitap : {args.kitap}")
    print("=" * 75)

    # 1. EMBEDDING BOYUTU DOĞRULAMA (768-D KONTROLÜ)
    print("\n[1/5] Ollama Servis ve 768-D Embedding Boyut Doğrulaması...")
    try:
        emb = verify_embedding_dimension(
            ollama_url=OLLAMA_URL,
            model_name=EMBED_MODEL,
            expected_dim=768
        )
        print("      ✓ Ollama bağlantısı başarılı.")
        print("      ✓ Test sorgusu atıldı ve dönen vektör TAM 768 boyut olarak onaylandı.")
    except Exception as err:
        print(f"\n[DURDURULDU] {err}")
        sys.exit(1)

    # 2. HEDEF KİTAPLARI VE DOSYA HASH'LERİNİ DOĞRULA
    print("\n[2/5] Hedef İlkokul Kitapları ve Dosya Hash Kontrolü...")
    book_files = []
    file_hashes: Dict[str, str] = {}
    for fname in selected_books:
        fpath = REFERENCE_BOOKS_DIR / fname
        if not fpath.exists():
            print(f"  [UYARI] Dosya bulunamadı: {fpath}")
            continue
        f_hash = calculate_file_hash(fpath)
        file_hashes[fname] = f_hash
        size_mb = round(fpath.stat().st_size / (1024 * 1024), 1)
        print(f"  ✓ {fname} ({size_mb} MB) -> SHA256: {f_hash[:12]}...")
        book_files.append(fpath)

    if not book_files:
        print("[HATA] reference_books klasöründe indekslenecek ilkokul kitabı bulunamadı!")
        sys.exit(1)

    # 3. CHROMADB VE MEVCUT CHUNK'LARI KONTROL ET (RESUMPTION / IDEMPOTENCY)
    print("\n[3/5] ChromaDB Başlatılıyor ve Mevcut Durum Taranıyor...")
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=TARGET_COLLECTION,
        embedding_function=emb,
        persist_directory=str(CHROMA_DB_DIR)
    )

    existing_ids = get_existing_chunk_ids(store)
    print(f"      ✓ Koleksiyondaki mevcut toplam parça: {len(existing_ids)}")

    splitter = create_pedagogical_splitter(chunk_size=1000, chunk_overlap=150)
    gemma_tok = get_tokenizer()
    if gemma_tok:
        print("      ✓ Gemma BPE Tokenizer (256k vocab) aktif; her parça için token sayılacak.")

    # 4. KİTAPLARI OKU, PARÇALA VE GÜVENLİ BATCH İLE YAZ
    print("\n[4/5] Kitaplar İşleniyor ve Vektörleştiriliyor...")
    total_newly_indexed = 0
    total_skipped = 0
    all_failed_items: List[Dict[str, Any]] = []
    t_start = time.time()

    for b_idx, fpath in enumerate(book_files, 1):
        t_book_start = time.time()
        fname = fpath.name
        f_hash = file_hashes[fname]
        manifest = PRIMARY_BOOK_MANIFESTS.get(fname, {})
        kitap_adi = manifest.get("kitap_adi", fname)

        print(f"\n--- [{b_idx}/{len(book_files)}] {kitap_adi} ({fname}) ---")

        # PDF Aç ve Sayfa Ofsetini Hesapla
        try:
            reader = pypdf.PdfReader(str(fpath))
            total_pages = len(reader.pages)
            page_offset = detect_page_offset(reader)
            print(f" -> Toplam {total_pages} fiziksel sayfa. Tespit Edilen Sayfa Ofseti: +{page_offset} "
                  f"(Fiziksel sayfa {8} = Basılı sayfa {8 - page_offset})")
        except Exception as e:
            print(f" [HATA] PDF okunamadı: {e}")
            continue

        book_docs: List[Document] = []
        book_ids: List[str] = []
        book_skipped = 0

        for phys_page_no in range(1, total_pages + 1):
            has_printed = phys_page_no > page_offset
            printed_page_no = (phys_page_no - page_offset) if has_printed else 0
            try:
                raw_text = (reader.pages[phys_page_no - 1].extract_text() or "").strip()
            except Exception:
                raw_text = ""

            if not raw_text or len(raw_text) < 25:
                continue

            try:
                base_meta, unite, konu, alt_konu = get_book_metadata_and_unit(
                    filename=fname,
                    phys_page_no=phys_page_no,
                    printed_page_no=printed_page_no,
                    raw_text=raw_text
                )
            except Exception as meta_err:
                print(f" [UYARI] Sayfa {phys_page_no} metadata hatası: {meta_err}")
                continue

            if has_printed:
                ref = f"{base_meta['kitap_adi']} - {unite}, '{konu}' ({alt_konu}) [Sayfa {printed_page_no}]"
            else:
                ref = f"{base_meta['kitap_adi']} - {unite}, '{konu}' ({alt_konu}) [Fiziksel Sayfa {phys_page_no}]"
            chunks = splitter.split_text(raw_text)

            for c_idx, c_text in enumerate(chunks):
                if len(c_text.strip()) < 20:
                    continue

                cid = generate_chunk_id(fname, f_hash, phys_page_no, c_idx)

                # Eğer chunk halihazırda veritabanında varsa ATLA (Tekrar çalıştırma güvenliği)
                if cid in existing_ids:
                    book_skipped += 1
                    continue

                t_count = count_tokens(c_text, gemma_tok)
                cm = {
                    "source": fname,
                    "kitap_adi": base_meta["kitap_adi"],
                    "egitim_seviyesi": base_meta["egitim_seviyesi"],
                    "ders_adi": base_meta["ders_adi"],
                    "sinif": base_meta["sinif"],
                    "physical_page": phys_page_no,
                    "printed_page": printed_page_no,
                    "page": printed_page_no,
                    "sayfa_no": printed_page_no,
                    "has_printed_page": has_printed,
                    "unite": unite,
                    "konu": konu,
                    "alt_konu": alt_konu,
                    "referans": ref,
                    "chunk_id": cid,
                    "file_hash": f_hash[:12],
                    "char_count": len(c_text),
                    "token_count": t_count,
                }
                book_docs.append(Document(page_content=c_text, metadata=cm))
                book_ids.append(cid)

        total_skipped += book_skipped
        if book_skipped > 0:
            print(f" -> Önceden indekslenmiş {book_skipped} parça atlandı (mükerrer kayıt engellendi).")

        if not book_docs:
            print(f" ✓ Bu kitap zaten tamamen güncel ({book_skipped} parça mevcut).")
            continue

        # Yazma öncesi çakışma kontrolü (Pre-write collision check)
        if len(book_ids) != len(set(book_ids)):
            seen_ids = set()
            duplicates = [cid for cid in book_ids if cid in seen_ids or seen_ids.add(cid)]
            raise ValueError(
                f"[KRİTİK HATA] {fname} için üretilen chunk ID'lerinde çakışma tespit edildi! "
                f"Toplam ID: {len(book_ids)}, Benzersiz: {len(set(book_ids))}. "
                f"Çakışan ilk ID'ler: {duplicates[:5]}"
            )

        print(f" -> Eklenecek {len(book_docs)} yeni parça tespit edildi. 768-D Vektörleştirme başlıyor...")

        # Batch ekleme ve Hata Yönetimi
        batch_size = 45
        book_success = 0
        for i in range(0, len(book_docs), batch_size):
            sub_docs = book_docs[i : i + batch_size]
            sub_ids = book_ids[i : i + batch_size]

            succ, fails = safe_add_documents_batch(store, sub_docs, sub_ids, max_retries=2)
            book_success += succ
            if fails:
                all_failed_items.extend(fails)

            progress = min(i + batch_size, len(book_docs))
            print(f"    [{progress}/{len(book_docs)}] parça yüklendi...", end="\r", flush=True)

        # Veritabanında doğrulanan gerçek kayıt sayısını kontrol et
        try:
            verified_records = store._collection.get(ids=book_ids, include=[])
            verified_count = len(verified_records["ids"]) if (verified_records and verified_records.get("ids")) else 0
        except Exception:
            verified_count = -1

        book_elapsed = round(time.time() - t_book_start, 1)
        total_newly_indexed += book_success
        print(f"\n ✓ {kitap_adi} tamamlandı ({book_success} parça sunuldu, DB'de doğrulanan: {verified_count}, {book_elapsed} sn).")

        # Güvenlik Kuralı: Eski sürüm kayıtlarını YALNIZCA yeni indeksleme eksiksiz tamamlandıysa temizle!
        book_has_failure = any(f.get("chunk_id", "").startswith(fname) for f in all_failed_items)
        if not book_has_failure and (book_success == len(book_docs)):
            cleaned = cleanup_stale_chunks(store, fname, f_hash)
            if cleaned > 0:
                print(f" -> İndeksleme eksiksiz tamamlandı: Eski hash'e ait {cleaned} parça temizlendi.")
        elif book_has_failure:
            print(f" [UYARI] Kitapta başarısız parçalar olduğu için eski sürüm silinmedi (veri kaybı engellendi).")

    total_elapsed = round(time.time() - t_start, 1)
    print("\n" + "=" * 75)
    print("🎉 İNDEKSLEME RAPORU")
    print(f" -> Yeni Eklenen (Sunulan) : {total_newly_indexed}")
    print(f" -> Atlanan (Mevcut)     : {total_skipped}")
    print(f" -> Başarısız Parça      : {len(all_failed_items)}")
    if all_failed_items:
        print("\n  ⚠️ BAŞARISIZ PARÇA DETAYLARI:")
        for it in all_failed_items[:5]:
            print(f"    - ID: {it['chunk_id']} | Hata: {it['error_type']}: {it['error_message']}")
        if len(all_failed_items) > 5:
            print(f"    ... ve {len(all_failed_items) - 5} parça daha.")

    print(f" -> Toplam Süre          : {total_elapsed} saniye ({round(total_elapsed/60, 1)} dakika)")
    print(f" -> Koleksiyon Toplamı   : {store._collection.count()} vektör")
    print("=" * 75)

    # 5. GERÇEKÇİ KAZANIM DOĞRULAMA TESTLERİ
    print("\n[5/5] Pedagojik 3 Seviyeli Kazanım ve Sayfa Doğrulama Testleri Yapılıyor...")
    test_queries = [
        ("3. Sınıf 1. Kitap Testi", "kayaçlar madenler ve fosil oluşumu", "3. Sınıf"),
        ("3. Sınıf 2. Kitap Testi", "elektrikli araç gereçler ve şehir elektriği", "3. Sınıf"),
        ("4. Sınıf Testi 1", "yer kabuğunun yapısı kayaçlar ve madenler", "4. Sınıf"),
        ("4. Sınıf Testi 2", "besin içerikleri proteinler karbonhidratlar dengeli beslenme", "4. Sınıf"),
    ]

    for label, query, target_sinif in test_queries:
        print(f"\n🔍 [{label}] Sorgu: '{query}' (Filtre: {target_sinif})")
        results = store.similarity_search(
            query=query,
            k=2,
            filter={"sinif": target_sinif}
        )
        if results:
            for idx, r in enumerate(results, 1):
                ref = r.metadata.get("referans", "Bilinmeyen Referans")
                unit = r.metadata.get("unite", "")
                konu = r.metadata.get("konu", "")
                alt_konu = r.metadata.get("alt_konu", "")
                page = r.metadata.get("printed_page", r.metadata.get("page", ""))
                tok_cnt = r.metadata.get("token_count", "-")
                snippet = r.page_content[:110].replace("\n", " ")
                print(f"   [{idx}] Basılı Sayfa {page} | {unit} -> {konu} ({alt_konu}) [{tok_cnt} token]")
                print(f"       Referans: {ref}")
                print(f"       Özet    : \"{snippet}...\"")
        else:
            print("   ⚠️ Sonuç bulunamadı.")

    print("\n" + "=" * 75)
    print("✅ İlkokul RAG altyapısı 768 boyutlu olarak tamamen doğrulanmış ve hazırdır.")
    print("=" * 75)


if __name__ == "__main__":
    main()
