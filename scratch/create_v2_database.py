"""
EduFrame AI - Aşama 2: v2 Veritabanını Hazırlama ve Doğrulama Scripti
Kaynak: data/chroma_db_v2_dryrun (765 vektör)
Hedef : data/chroma_db_v2 (765 vektör)
"""

import sys
import shutil
import sqlite3
import hashlib
from pathlib import Path
import chromadb

ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT_DIR / "data" / "chroma_db_v2_dryrun"
TARGET_DIR = ROOT_DIR / "data" / "chroma_db_v2"
LIVE_DIR = ROOT_DIR / "data" / "chroma_db"

print("=" * 70)
print("📦 EduFrame AI - Aşama 2: v2 Veritabanını Devreye Almaya Hazırlama")
print("=" * 70)
print(f"Kaynak Dizin : {SOURCE_DIR}")
print(f"Hedef Dizin  : {TARGET_DIR}")

# 1. Kaynak ve hedef kontrolleri
if not SOURCE_DIR.exists():
    print(f"❌ [HATA] Kaynak dizin bulunamadı: {SOURCE_DIR}")
    sys.exit(1)

if TARGET_DIR.exists():
    print(f"❌ [HATA] Hedef dizin zaten mevcut: {TARGET_DIR}. Otomatik üzerine yazma engellendi!")
    sys.exit(1)

# 2. SQLite WAL kontrolü ve flush
sqlite_path = SOURCE_DIR / "chroma.sqlite3"
wal_path = SOURCE_DIR / "chroma.sqlite3-wal"
if wal_path.exists():
    print("ℹ️ Kaynakta WAL dosyası tespit edildi, checkpoint yapılıyor...")
    con = sqlite3.connect(str(sqlite_path))
    con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    con.close()
    print("✓ WAL checkpoint tamamlandı.")

# 3. Bağımsız kopya oluşturma
print("📋 Kopyalama işlemi başlatılıyor (shutil.copytree)...")
shutil.copytree(str(SOURCE_DIR), str(TARGET_DIR))
print("✓ Kopyalama tamamlandı.")

# 4. Doğrulama
print("\n🔍 Hedef Veritabanı (data/chroma_db_v2) Derinlemesine Denetleniyor...")
client = chromadb.PersistentClient(path=str(TARGET_DIR))
try:
    col = client.get_collection("eduframe_meb_768")
except Exception as e:
    print(f"❌ [HATA] Koleksiyon bulunamadı: {e}")
    sys.exit(1)

total_count = col.count()
print(f" -> Toplam Vektör Sayısı: {total_count} (Beklenen: 765)")
if total_count != 765:
    print(f"❌ [HATA] Kayıt sayısı uyuşmazlığı: {total_count} != 765")
    sys.exit(1)

# Tüm kayıtları çek ve doğrula
records = col.get(include=["metadatas", "documents", "embeddings"])
ids = records["ids"]
metas = records["metadatas"]
docs = records["documents"]
embs = records["embeddings"]

book_counts = {
    "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf": 0,
    "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf": 0,
    "ilkokul_fenbilimleri_4.sinif.pdf": 0,
}

expected_book_counts = {
    "ilkokul_fenbilimleri_3.sinif_1.kitap.pdf": 205,
    "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf": 186,
    "ilkokul_fenbilimleri_4.sinif.pdf": 374,
}

issues = []
for cid, meta, doc, emb in zip(ids, metas, docs, embs):
    src = meta.get("source")
    if src in book_counts:
        book_counts[src] += 1
    else:
        issues.append(f"{cid}: Beklenmeyen kaynak kitap: {src}")

    # Sürüm etiketi
    if meta.get("version") != "v2":
        issues.append(f"{cid}: Sürüm metadata 'v2' değil ({meta.get('version')})")
    if ":v2:" not in cid:
        issues.append(f"{cid}: ID içinde :v2: etiketi eksik")

    # Sayfa ve referans
    if not meta.get("physical_page") or meta.get("physical_page") <= 0:
        issues.append(f"{cid}: Geçersiz physical_page: {meta.get('physical_page')}")

    # SHA-256 Hash kontrolü
    calc_hash = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:12]
    saved_hash = meta.get("content_hash")
    if calc_hash != saved_hash:
        issues.append(f"{cid}: SHA-256 content_hash uyuşmazlığı! Kayıtlı={saved_hash}, Hesaplanan={calc_hash}")

    # Embedding boyutu
    if len(emb) != 768:
        issues.append(f"{cid}: Embedding boyutu {len(emb)} != 768")

print("\n📚 Kitap Bazında Dağılım:")
all_counts_match = True
for bname, count in book_counts.items():
    exp = expected_book_counts[bname]
    status = "✓ DOĞRU" if count == exp else "❌ HATALI"
    print(f"  - {bname}: {count} parça (Beklenen: {exp}) -> {status}")
    if count != exp:
        all_counts_match = False

if not all_counts_match:
    print("❌ [HATA] Kitap dağılımı beklentilerle uyuşmuyor!")
    sys.exit(1)

if issues:
    print(f"❌ [HATA] {len(issues)} bütünlük sorunu tespit edildi! İlk 3: {issues[:3]}")
    sys.exit(1)

print("\n✓ Bütünlük Denetimi: 765 parçanın TAMAMI v2 etiketli, 768 boyutlu ve SHA-256 content_hash ile %100 doğrulanmıştır.")

# Kaynak ve canlı kontrolü
dryrun_col = chromadb.PersistentClient(path=str(SOURCE_DIR)).get_collection("eduframe_meb_768")
live_col = chromadb.PersistentClient(path=str(LIVE_DIR)).get_collection("eduframe_meb_768")
print(f"✓ Kaynak DB (chroma_db_v2_dryrun) Korundu: {dryrun_col.count()} vektör")
print(f"✓ Canlı DB (chroma_db) Korundu: {live_col.count()} vektör")

print("=" * 70)
print("🎉 Aşama 2 Başarıyla Tamamlandı: data/chroma_db_v2 devreye alınmaya hazır!")
print("=" * 70)
