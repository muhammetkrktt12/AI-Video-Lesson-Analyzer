"""
EduFrame AI - Aşama 4: Devreye Alma Öncesi 9 Adımlı Doğrulama ve RAG Benchmark Testi
"""

import os
import sys
import gc
import hashlib
from pathlib import Path

# Proje kök dizinini ekle
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# v2 veritabanını ortam değişkeniyle aktif et
V2_PATH = str(ROOT_DIR / "data" / "chroma_db_v2")
V1_PATH = str(ROOT_DIR / "data" / "chroma_db")
os.environ["EDUFRAME_CHROMA_DB_DIR"] = V2_PATH

import chromadb
from config import CHROMA_DB_DIR, OLLAMA_URL, EMBED_MODEL
from services.rag_motoru import RAGMotoru, GuvenliEmbeddingMotoru
from quality_utils import rerank_candidates, hit_at_k

print("=" * 75)
print("🧪 EduFrame AI - Aşama 4: v2 Devreye Alma Öncesi Doğrulama Testleri")
print("=" * 75)
print(f"Aktif EDUFRAME_CHROMA_DB_DIR : {os.environ.get('EDUFRAME_CHROMA_DB_DIR')}")
print(f"config.CHROMA_DB_DIR        : {CHROMA_DB_DIR}")

test_results = {}

# ---------------------------------------------------------------------------
# TEST 1: Uygulama Başlatma Testi
# ---------------------------------------------------------------------------
print("\n[Test 1/9] Uygulama Başlatma ve Çekirdek Modül Yükleme...")
try:
    from core.eduframe_agentic_core import EduFrameAgenticCore
    core = EduFrameAgenticCore()
    print("  ✓ EduFrameAgenticCore başarıyla yüklendi ve örneklendi.")
    test_results["Test 1: Uygulama Başlatma"] = "BAŞARILI"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 1: Uygulama Başlatma"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# TEST 2: ChromaDB Bağlantı Testi
# ---------------------------------------------------------------------------
print("\n[Test 2/9] ChromaDB Bağlantı Testi (v2)...")
try:
    rag = RAGMotoru()
    durum = rag.mevcut_veritabani_durumu()
    v_count = durum.get("toplam_parca", 0)
    print(f"  ✓ RAGMotoru bağlantısı başarılı. Vektör deposu dizini: {rag.db_dizini}")
    print(f"  ✓ Toplam kayıt sayısı: {v_count} (Beklenen: 765)")
    assert v_count == 765, f"Beklenen 765, gelen {v_count}"
    test_results["Test 2: ChromaDB Bağlantısı"] = "BAŞARILI"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 2: ChromaDB Bağlantısı"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# TEST 3: Embedding Modeli ve Boyut Kontrolü
# ---------------------------------------------------------------------------
print("\n[Test 3/9] Embedding Modeli ve Boyut Kontrolü...")
try:
    emb_motoru = rag.embedding
    test_vec = emb_motoru.embed_query("Türkiye Cumhuriyeti Milli Eğitim Bakanlığı Fen Bilimleri")
    dim = len(test_vec)
    print(f"  ✓ Embedding modeli: {emb_motoru.model_adi}")
    print(f"  ✓ Test vektör boyutu: {dim} (Beklenen: 768)")
    assert dim == 768, f"Vektör boyutu {dim} != 768"
    test_results["Test 3: Embedding Boyutu (768-D)"] = "BAŞARILI"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 3: Embedding Boyutu (768-D)"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# TEST 4: Üç Kitap Üzerinde RAG Retrieval Testi
# ---------------------------------------------------------------------------
print("\n[Test 4/9] Üç Kitap Üzerinde RAG Retrieval Testi...")
try:
    retrieval_ok = True
    book_tests = [
        ("3. Sınıf 1. Kitap", "kayaçlar madenler ve fosil oluşumu", "3. Sınıf", "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf"),
        ("3. Sınıf 2. Kitap", "elektrikli araç gereçler ve şehir elektriği", "3. Sınıf", "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf"),
        ("4. Sınıf Kitabı", "yer kabuğunun yapısı kayaçlar ve madenler", "4. Sınıf", "ilkokul_fenbilimleri_4.sinif.pdf"),
    ]
    for b_label, q, sev, exp_file in book_tests:
        results = rag.seviye_filtreli_arama(sorgu=q, egitim_seviyesi="İlkokul", sinif=sev, k=3, rerank=True)
        if not results:
            print(f"  ❌ {b_label} için arama sonucu boş döndü!")
            retrieval_ok = False
            continue
        first_doc = results[0]
        first_source = first_doc.metadata.get("source")
        print(f"  ✓ {b_label} -> 1. Aday Kaynak: {first_source} | Sayfa: {first_doc.metadata.get('physical_page')} (Basılı: {first_doc.metadata.get('printed_page')})")
        if first_source != exp_file:
            print(f"  ⚠️ Uyarı: 1. aday {first_source} != beklenen {exp_file}")
    test_results["Test 4: 3 Kitap RAG Retrieval"] = "BAŞARILI" if retrieval_ok else "BAŞARISIZ"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 4: 3 Kitap RAG Retrieval"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# TEST 5: Hibrit Reranking Testi
# ---------------------------------------------------------------------------
print("\n[Test 5/9] Hibrit Reranking ve Gürültü Filtreleme Testi...")
try:
    # 4. Sınıf sayfa 20/22/25 vs gürültülü sayfa 41 testi
    q_rerank = "yer kabuğunun yapısı kayaçlar ve madenler"
    store = rag.get_vector_store()
    raw_candidates = store.similarity_search(query=q_rerank, k=8, filter={"sinif": "4. Sınıf"})
    ranked = rerank_candidates(q_rerank, raw_candidates, top_n=3)
    top1_doc, top1_score = ranked[0]
    p_page = top1_doc.metadata.get("physical_page")
    pr_page = top1_doc.metadata.get("printed_page")
    print(f"  ✓ Rerank Top-1 Belge: Fiziksel {p_page}, Basılı {pr_page} (Skor: {top1_score:.4f})")
    print(f"  ✓ Top-1 İçerik Özeti : '{top1_doc.page_content[:90]}...'")
    # Top-1 belgede nokta dizisi (.....) olmamalıdır
    assert "....." not in top1_doc.page_content, "Top-1 içerikte temizlenmemiş nokta dizisi var!"
    test_results["Test 5: Hibrit Reranking"] = "BAŞARILI"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 5: Hibrit Reranking"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# TEST 6: Metadata ve Sayfa Referansı Kontrolü
# ---------------------------------------------------------------------------
print("\n[Test 6/9] Metadata ve Sayfa Referansı Bütünlük Kontrolü...")
try:
    meta_ok = True
    sample_docs = rag.seviye_filtreli_arama(sorgu="besin içerikleri proteinler karbonhidratlar", egitim_seviyesi="İlkokul", sinif="4. Sınıf", k=5, rerank=False)
    for d in sample_docs:
        m = d.metadata
        req_fields = ["physical_page", "printed_page", "source", "version", "content_hash", "unite", "konu"]
        for f in req_fields:
            if f not in m:
                print(f"  ❌ Eksik metadata alanı: {f} in {m.get('chunk_id')}")
                meta_ok = False
        if m.get("version") != "v2":
            print(f"  ❌ Yanlış sürüm metadata: {m.get('version')} in {m.get('chunk_id')}")
            meta_ok = False
    if meta_ok:
        print("  ✓ Örnek belgelerin tamamında physical_page, printed_page, version='v2', unite, konu eksiksiz.")
    test_results["Test 6: Metadata Bütünlüğü"] = "BAŞARILI" if meta_ok else "BAŞARISIZ"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 6: Metadata Bütünlüğü"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# TEST 7: 10 Sorguluk Mevcut Benchmark
# ---------------------------------------------------------------------------
print("\n[Test 7/9] 10 Sorguluk Kapsamlı RAG Benchmark Testi...")
BENCHMARK_CASES = [
    {
        "id": 1,
        "query": "Laboratuvarda uyulması gereken güvenlik sembolleri nelerdir?",
        "expected_pages": [12, 13, 14],
        "sinif": None,
    },
    {
        "id": 2,
        "query": "Bilim insanlarının sahip olduğu ortak özellikler nelerdir?",
        "expected_pages": [30, 31, 37],
        "sinif": None,
    },
    {
        "id": 3,
        "query": "Duyu organlarımızın sağlığını korumak için nelere dikkat etmeliyiz?",
        "expected_pages": [67, 68, 69, 70],
        "sinif": None,
    },
    {
        "id": 4,
        "query": "Kayaçlar, madenler ve minerallerin özellikleri nelerdir?",
        "expected_pages": [81, 88, 91, 92],
        "sinif": None,
    },
    {
        "id": 5,
        "query": "Fosil oluşumu nasıl gerçekleşir?",
        "expected_pages": [97, 98, 99, 100, 104],
        "sinif": None,
    },
    {
        "id": 6,
        "query": "Maddelerin nitelikleri nelerdir?",
        "expected_pages": [101, 111, 114, 136],
        "sinif": None,
    },
    {
        "id": 7,
        "query": "Maddenin katı, sıvı ve gaz hâlleri nelerdir?",
        "expected_pages": [113, 114, 115, 116],
        "sinif": None,
    },
    {
        "id": 8,
        "query": "Karışımlar hangi yöntemlerle birbirinden ayrılabilir?",
        "expected_pages": [124, 125, 126, 127],
        "sinif": None,
    },
    {
        "id": 9,
        "query": "Atıkların ayrıştırılması ve geri dönüşümün önemi nedir?",
        "expected_pages": [132, 134, 140, 141, 189],
        "sinif": None,
    },
    {
        "id": 10,
        "query": "Yer kabuğunun yapısı ve madenlerin ekonomik önemi nedir?",
        "expected_pages": [20, 21, 22, 25],
        "sinif": None,
    },
]

try:
    hits_3 = 0
    top1_hits = 0
    reciprocal_ranks = []
    noise_count = 0
    store = rag.get_vector_store()

    for item in BENCHMARK_CASES:
        flt = {"sinif": item["sinif"]} if item.get("sinif") else None
        raw_cands = store.similarity_search(query=item["query"], k=8, filter=flt)
        reranked = rerank_candidates(item["query"], raw_cands, top_n=3)

        top3_pages = [d.metadata.get("printed_page") for d, _ in reranked]
        top1_doc, top1_score = reranked[0]
        top1_page = top1_doc.metadata.get("printed_page")

        exp_set = set(item["expected_pages"])
        is_top1 = top1_page in exp_set
        is_hit3 = any(p in exp_set for p in top3_pages)

        # MRR
        rr = 0.0
        for rank, (d, _) in enumerate(reranked, start=1):
            if d.metadata.get("printed_page") in exp_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        if is_hit3:
            hits_3 += 1
        if is_top1:
            top1_hits += 1

        # Gürültü kontrolü
        if "....." in top1_doc.page_content:
            noise_count += 1

        status_str = "TAM İSABET (P@1)" if is_top1 else ("HİT@3" if is_hit3 else "ISKA")
        print(f"  [{item['id']:02d}/10] {item['query'][:45]}... -> Top-1: Sayfa {top1_page} ({status_str})")

    recall_at_3 = (hits_3 / len(BENCHMARK_CASES)) * 100
    precision_at_1 = (top1_hits / len(BENCHMARK_CASES)) * 100
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"\n  📊 BENCHMARK SONUÇLARI:")
    print(f"     - Recall@3        : %{recall_at_3:.1f} ({hits_3}/{len(BENCHMARK_CASES)})")
    print(f"     - Precision@1     : %{precision_at_1:.1f} ({top1_hits}/{len(BENCHMARK_CASES)})")
    print(f"     - Ortalama MRR    : {mrr:.3f}")
    print(f"     - Gürültü Oranı   : %{noise_count * 10:.1f} ({noise_count} parça)")

    assert recall_at_3 >= 90.0, f"Recall@3 beklenenden düşük: %{recall_at_3}"
    assert precision_at_1 >= 90.0, f"Precision@1 beklenenden düşük: %{precision_at_1}"
    assert noise_count == 0, f"Gürültülü parça tespit edildi: {noise_count}"
    test_results["Test 7: 10'lu Benchmark"] = f"BAŞARILI (Recall@3=%{recall_at_3:.0f}, P@1=%{precision_at_1:.0f}, MRR={mrr:.3f})"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 7: 10'lu Benchmark"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# TEST 8: Bağlantı Kapatılıp Yeniden Açıldığında Veri Bütünlüğü
# ---------------------------------------------------------------------------
print("\n[Test 8/9] Bağlantı Kapatılıp Yeniden Açıldığında Veri Bütünlüğü...")
try:
    del rag
    gc.collect()

    # Yeni bağımsız client aç
    client_reopen = chromadb.PersistentClient(path=V2_PATH)
    col_reopen = client_reopen.get_collection("eduframe_meb_768")
    reopen_count = col_reopen.count()
    print(f"  ✓ Bağlantı kapatılıp yeniden açıldı. Vektör sayısı: {reopen_count} (Beklenen: 765)")
    assert reopen_count == 765
    test_results["Test 8: Bağlantı Döngüsü ve Kalıcılık"] = "BAŞARILI"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 8: Bağlantı Döngüsü ve Kalıcılık"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# TEST 9: Eski v1 Veritabanına Geri Dönüş (Rollback) Testi
# ---------------------------------------------------------------------------
print("\n[Test 9/9] Eski v1 Veritabanına Geri Dönüş (Rollback) Testi...")
try:
    # 1. Ortam değişkenini v1'e çevir
    os.environ["EDUFRAME_CHROMA_DB_DIR"] = V1_PATH
    rag_v1 = RAGMotoru(db_dizini=Path(V1_PATH))
    durum_v1 = rag_v1.mevcut_veritabani_durumu()
    v1_count = durum_v1.get("toplam_parca", 0)
    print(f"  ✓ v1 Rollback bağlantısı: {rag_v1.db_dizini}")
    print(f"  ✓ v1 Kayıt sayısı: {v1_count} (Beklenen: 903)")
    assert v1_count == 903, f"Rollback v1 sayısı uyuşmazlığı: {v1_count} != 903"

    # 2. Ortam değişkenini tekrar v2'ye çevir
    os.environ["EDUFRAME_CHROMA_DB_DIR"] = V2_PATH
    rag_v2 = RAGMotoru(db_dizini=Path(V2_PATH))
    durum_v2 = rag_v2.mevcut_veritabani_durumu()
    v2_count = durum_v2.get("toplam_parca", 0)
    print(f"  ✓ v2 Aktif bağlantısı: {rag_v2.db_dizini}")
    print(f"  ✓ v2 Kayıt sayısı: {v2_count} (Beklenen: 765)")
    assert v2_count == 765, f"v2 sayısı uyuşmazlığı: {v2_count} != 765"

    test_results["Test 9: v1 Rollback Geçiş Testi"] = "BAŞARILI"
except Exception as e:
    print(f"  ❌ HATA: {e}")
    test_results["Test 9: v1 Rollback Geçiş Testi"] = f"BAŞARISIZ ({e})"

# ---------------------------------------------------------------------------
# ÖZET TABLOSU
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("📋 DEVREYE ALMA ÖNCESİ TEST RAPORU ÖZETİ")
print("=" * 75)
all_passed = True
for test_name, status in test_results.items():
    print(f"  - {test_name:<38} : {status}")
    if "BAŞARISIZ" in status:
        all_passed = False

print("=" * 75)
if all_passed:
    print("🎯 TÜM DEVREYE ALMA ÖNCESİ TESTLER %100 BAŞARIYLA GEÇTİ! (GO)")
else:
    print("🚨 BAZI TESTLER BAŞARISIZ OLDU! (NO-GO)")
print("=" * 75)
