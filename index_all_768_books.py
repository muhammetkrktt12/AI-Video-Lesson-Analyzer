"""
EduFrame AI - Tüm MEB Referans Kitaplarını 768 Boyutlu Ollama Embedding ile İndeksleme
-------------------------------------------------------------------------------------
data/reference_books klasöründeki tüm MEB ders kitaplarını tarar.
Daha önce eklenmiş olan kitapları atlar; kalan tüm kitapları 768-D
embeddinggemma modeliyle 'eduframe_meb_768' koleksiyonuna ekler.
"""

import sys
import time
from pathlib import Path
import pypdf
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from index_all_reference_books import (
    parse_filename,
    parse_toc_structure,
    get_unit_and_topic_for_page,
)
from config import CHROMA_DB_DIR, REFERENCE_BOOKS_DIR

TARGET_COLLECTION = "eduframe_meb_768"

def main():
    print("=" * 70, flush=True)
    print("📚 EduFrame AI - Tüm Ders Kitapları 768-D İndeksleme Motoru", flush=True)
    print("=" * 70, flush=True)

    # 1. Mevcut koleksiyonu ve önceden eklenen kitapları kontrol et
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    col = client.get_or_create_collection(name=TARGET_COLLECTION)
    existing_count = col.count()
    print(f"[*] '{TARGET_COLLECTION}' koleksiyonundaki mevcut parça sayısı: {existing_count}", flush=True)

    # Önceden eklenmiş dosyaları tespit et
    already_indexed_sources = set()
    if existing_count > 0:
        res = col.get(limit=100000, include=["metadatas"])
        for m in res["metadatas"]:
            if m and "source" in m:
                already_indexed_sources.add(m["source"])

    print(f"[*] Halihazırda indeksli kitap sayısı: {len(already_indexed_sources)}", flush=True)
    for s in sorted(already_indexed_sources):
        print(f"    ✓ {s}", flush=True)

    # 2. Embedding motorunu ve ChromaStore'u bağla
    emb = OllamaEmbeddings(model="embeddinggemma", base_url="http://localhost:11434")
    store = Chroma(
        collection_name=TARGET_COLLECTION,
        embedding_function=emb,
        persist_directory=str(CHROMA_DB_DIR)
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "]
    )

    # 3. Klasördeki tüm PDF dosyalarını listele
    all_pdfs = sorted(list(REFERENCE_BOOKS_DIR.glob("*.pdf")))
    print(f"\n[*] Klasördeki toplam PDF sayısı: {len(all_pdfs)}", flush=True)

    to_index = [p for p in all_pdfs if p.name not in already_indexed_sources]
    print(f"[*] İndekslenecek YENİ kitap sayısı: {len(to_index)}", flush=True)

    if not to_index:
        print("\n🎉 Tüm kitaplar zaten 768 boyutlu olarak indekslenmiş durumda!", flush=True)
        return

    print("\n🚀 İndeksleme işlemi başlatılıyor...", flush=True)
    total_added = 0
    t_start = time.time()

    for idx, fpath in enumerate(to_index, 1):
        t_book_start = time.time()
        print(f"\n>>> [{idx}/{len(to_index)}] Kitap Başlatılıyor: {fpath.name}", flush=True)
        meta_base = parse_filename(fpath.name)
        try:
            reader = pypdf.PdfReader(str(fpath))
            toc_entries = parse_toc_structure(reader)
        except Exception as e:
            print(f" [HATA] PDF okunamadı: {e}", flush=True)
            continue

        docs = []
        ids = []
        for page_no, page in enumerate(reader.pages, 1):
            try:
                raw_text = (page.extract_text() or "").strip()
            except Exception:
                raw_text = ""
            if not raw_text or len(raw_text) < 25:
                continue

            unite, konu = get_unit_and_topic_for_page(
                page_no, raw_text, toc_entries, f"{meta_base['kitap_adi']} İçeriği"
            )
            ref = f"{meta_base['kitap_adi']} - {unite}, '{konu}' (Sayfa {page_no})"
            chunks = splitter.split_text(raw_text)

            for c_idx, c_text in enumerate(chunks):
                if len(c_text.strip()) < 20:
                    continue
                cid = f"{fpath.name}:p{page_no:03d}:c{c_idx:02d}"
                cm = {
                    "source": fpath.name,
                    "kitap_adi": meta_base["kitap_adi"],
                    "egitim_seviyesi": meta_base["egitim_seviyesi"],
                    "ders_adi": meta_base["ders_adi"],
                    "sinif": meta_base["sinif"],
                    "kitap_no": meta_base["kitap_no"],
                    "page": page_no,
                    "sayfa_no": page_no,
                    "unite": unite,
                    "konu": konu,
                    "referans": ref,
                    "chunk_id": cid,
                }
                docs.append(Document(page_content=c_text, metadata=cm))
                ids.append(cid)

        if not docs:
            print("  ⚠️ Metin çıkarılamadı veya boş sayfalardan oluşuyor.", flush=True)
            continue

        print(f"  -> Toplam {len(docs)} parça çıkarıldı. 768-D vektörleştirme başlıyor...", flush=True)
        batch_size = 50
        for b_idx in range(0, len(docs), batch_size):
            sub_docs = docs[b_idx : b_idx + batch_size]
            sub_ids = ids[b_idx : b_idx + batch_size]
            try:
                store.add_documents(documents=sub_docs, ids=sub_ids)
            except Exception as e:
                print(f"\n  [UYARI] Batch ({b_idx}) yazılırken hata: {e}", flush=True)
                time.sleep(2)
                try:
                    store.add_documents(documents=sub_docs, ids=sub_ids)
                except Exception as e2:
                    print(f"  [HATA] İkinci deneme de başarısız: {e2}", flush=True)
            
            ilerleme = min(b_idx + batch_size, len(docs))
            print(f"     [{ilerleme}/{len(docs)}] parça eklendi...", flush=True)

        total_added += len(docs)
        book_elapsed = round(time.time() - t_book_start, 1)
        print(f"  ✓ {fpath.name} tamamlandı ({len(docs)} parça, {book_elapsed} sn). Toplam: {total_added}", flush=True)

    elapsed_total = round(time.time() - t_start, 1)
    print("\n" + "=" * 70, flush=True)
    print(f"🎉 TÜM İŞLEM BAŞARIYLA BİTTİ!", flush=True)
    print(f"   -> Eklenen Yeni Parça: {total_added}", flush=True)
    print(f"   -> Toplam Geçen Süre: {elapsed_total} saniye ({round(elapsed_total/60, 1)} dakika)", flush=True)
    print(f"   -> '{TARGET_COLLECTION}' Toplam Vektör Sayısı: {store._collection.count()}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
