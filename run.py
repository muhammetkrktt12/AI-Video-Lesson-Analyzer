"""
EduFrame AI - Hızlı Video Analiz ve PDF Üretim CLI Aracı (run.py)
-----------------------------------------------------------------
Kullanım:
    python run.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python run.py "https://www.youtube.com/watch?v=VIDEO_ID" --seviye "İlkokul"
"""

import sys
import argparse
from pathlib import Path

# Proje yolunu ekle
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.eduframe_agentic_core import EduFrameAgenticCore
from config import DEFAULT_SEVIYE

def main():
    parser = argparse.ArgumentParser(description="EduFrame AI - Video Ders Analizi ve PDF Fasikül Üreticisi")
    parser.add_argument("url", nargs="?", default="https://www.youtube.com/watch?v=1EPox1m2MPQ",
                        help="Analiz edilecek YouTube video bağlantısı veya yerel MP4 yolu")
    parser.add_argument("--seviye", default="İlkokul", choices=["İlkokul", "Ortaokul", "Lise", "Üniversite"],
                        help="Hedef eğitim kademesi (Varsayılan: İlkokul)")
    parser.add_argument("--soru-yok", action="store_true", help="Değerlendirme sorularını fasiküle dahil etme")

    args = parser.parse_args()

    print("=" * 65)
    print("🎓 EduFrame AI - Otonom Ders Notu ve Pedagojik Fasikül Motoru")
    print("=" * 65)
    print(f" -> Video Kaynağı : {args.url}")
    print(f" -> Eğitim Kademesi: {args.seviye}")
    print(f" -> Soru Üretimi  : {'Devre Dışı' if args.soru_yok else 'Aktif (10 Özgün Soru + Çözümler)'}")
    print("=" * 65)

    core = EduFrameAgenticCore()
    sonuc = core.calistir(
        kullanici_girisi=f"{args.seviye} seviyesi için ders notu fasikülü hazırla",
        video_kaynagi=args.url,
        egitim_seviyesi=args.seviye,
        soru_eklensin_mi=not args.soru_yok
    )

    pdf_yolu = sonuc.get("cikti_pdf_yolu")
    if pdf_yolu:
        print("\n" + "=" * 65)
        print("🎉 BAŞARILI! Ders fasikülü oluşturuldu:")
        print(f"📄 PDF Dosyası: {pdf_yolu}")
        print("=" * 65)
    else:
        print("\n❌ Bir hata oluştu veya PDF üretilemedi.")
        if sonuc.get("hata"):
            print("Hata Detayı:", sonuc.get("hata"))

if __name__ == "__main__":
    main()
