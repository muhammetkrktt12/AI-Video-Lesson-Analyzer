"""
EduFrame AI - MEB 3. Sınıf Fen Kitaplarını 768 Boyutlu Ollama Embedding ile İndeksleme
"""
import time
from pathlib import Path
import pypdf
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from index_all_reference_books import parse_filename, parse_toc_structure, get_unit_and_topic_for_page
from config import CHROMA_DB_DIR, REFERENCE_BOOKS_DIR

target_collection = "eduframe_meb_768"
emb = OllamaEmbeddings(model="embeddinggemma", base_url="http://localhost:11434")
store = Chroma(collection_name=target_collection, embedding_function=emb, persist_directory=str(CHROMA_DB_DIR))

splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "])

target_files = [
    "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
    "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf"
]

total_indexed = 0
t0 = time.time()

for fname in target_files:
    fpath = REFERENCE_BOOKS_DIR / fname
    if not fpath.exists():
        print(f"Dosya bulunamadı: {fname}")
        continue
    meta_base = parse_filename(fpath.name)
    reader = pypdf.PdfReader(str(fpath))
    toc_entries = parse_toc_structure(reader)
    
    docs = []
    ids = []
    for page_no, page in enumerate(reader.pages, 1):
        raw_text = (page.extract_text() or "").strip()
        if not raw_text or len(raw_text) < 25:
            continue
        unite, konu = get_unit_and_topic_for_page(page_no, raw_text, toc_entries, f"{meta_base['kitap_adi']} İçeriği")
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
                "chunk_id": cid
            }
            docs.append(Document(page_content=c_text, metadata=cm))
            ids.append(cid)
    
    print(f"Adding {len(docs)} chunks for {fname}...")
    batch_size = 25
    for i in range(0, len(docs), batch_size):
        sub_docs = docs[i:i+batch_size]
        sub_ids = ids[i:i+batch_size]
        store.add_documents(documents=sub_docs, ids=sub_ids)
        print(f" -> [{i+1}-{min(i+batch_size, len(docs))} / {len(docs)}] indexed")
    total_indexed += len(docs)

print(f"ALL DONE! Indexed {total_indexed} chunks in {round(time.time() - t0, 1)}s. Total in store: {store._collection.count()}")
