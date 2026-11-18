"""
EduFrame AI (AI Video Lesson Analyzer) - Merkezi Yapılandırma Dosyası
Tüm modüller için dosya yolları, model ayarları ve sabitler burada tanımlanır.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyası varsa otomatik yükle
load_dotenv()

# --- DİZİN YAPILANDIRMASI ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# ChromaDB Sürüm Dizinleri ve Aktif Yol Yapılandırması:
# EDUFRAME_CHROMA_DB_DIR ortam değişkeni tanımlıysa o yol kullanılır (örn: data/chroma_db_v2).
# Tanımlı değilse varsayılan yol güvenli v1 veritabanı (data/chroma_db) olarak kalır (Rollback uyumlu).
CHROMA_DB_V1_DIR = DATA_DIR / "chroma_db"
CHROMA_DB_V2_DIR = DATA_DIR / "chroma_db_v2"

_custom_db_dir = os.getenv("EDUFRAME_CHROMA_DB_DIR")
if _custom_db_dir:
    _custom_path = Path(_custom_db_dir)
    CHROMA_DB_DIR = _custom_path if _custom_path.is_absolute() else (ROOT / _custom_path)
else:
    CHROMA_DB_DIR = CHROMA_DB_V1_DIR

REFERENCE_BOOKS_DIR = DATA_DIR / "reference_books"

# Geçici medya klasörleri (Video indirme, kare yakalama, PDF çıktıları)
TEMP_DIR = ROOT / "temp_medya"
VIDEOS_DIR = TEMP_DIR / "videos"
FRAMES_DIR = TEMP_DIR / "frames"
OUTPUTS_DIR = TEMP_DIR / "outputs"

# Geriye dönük uyumluluk için eski değişken isimleri
DB_DIR = CHROMA_DB_DIR
COLLECTION = "eduframe_meb_768"

# --- EĞİTİM SEVİYELERİ ---
EGITIM_SEVIYELERI = ["İlkokul", "Ortaokul", "Lise", "Üniversite"]
DEFAULT_SEVIYE = "İlkokul"

# --- GOOGLE GEMINI API AYARLARI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# --- OLLAMA VE LOKAL MODEL AYARLARI (Birincil Model) ---
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
CHAT_MODEL = OLLAMA_MODEL

# --- VIDEO İNDİRME AYARLARI ---
DEFAULT_VIDEO_RESOLUTION = "720"  # 360 veya 720
ALLOWED_VIDEO_RESOLUTIONS = ["360", "720"]

# Klasörlerin varlığından emin ol
for path in [DATA_DIR, CHROMA_DB_DIR, REFERENCE_BOOKS_DIR, TEMP_DIR, VIDEOS_DIR, FRAMES_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
