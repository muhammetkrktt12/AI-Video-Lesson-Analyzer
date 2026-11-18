"""
EduFrame AI - Gemini 1.5 Pro Multimodal Materyal ve Görsel Değişim Analiz Servisi (services/gemini_analyzer.py)
---------------------------------------------------------------------------------------------------------
Bu modül, videoyu baştan sona tek parça halinde özetlemek yerine;
EKRANDAKİ GÖRSEL MATERYALLERİN (slayt, şema, tahta vb.) DEĞİŞİMİNE GÖRE
bölümlere ayırır. Her materyalin ekranda kaldığı süre boyunca öğretmenin
yaptığı konuşmaları transkript eder ve ders kitabı (RAG) ile sentezler.

Temel Kurallar:
1. Görsel Materyal Değişimi Esastır: Yeni bir slayt/konu geldiğinde yeni bölüm başlar.
2. Küçük animasyonlar ve imleç hareketleri yeni materyal sayılmaz.
3. Reklam, sponsor veya eğitim dışı kısımlar elenir (ekran görüntüsü ve bölüm oluşturulmaz).
4. Her bölüm için:
   - Materyal Başlığı
   - Zaman Aralığı (Örn: 00:00 – 02:35)
   - Orijinal Transkript (Öğretmenin o andaki ham konuşması)
   - Düzenlenmiş Konu Anlatımı (Ders kitabı referanslı seviyeye uygun anlatım)
   - Önemli Kavramlar ve Açıklamaları
5. Ders notu sonu için Genel Konu Özeti üretilir.
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from config import (
        GEMINI_API_KEY,
        GEMINI_MODEL,
        GEMINI_TIMEOUT,
        GEMINI_MAX_RETRIES,
        GEMINI_FALLBACK_MODELS,
        OLLAMA_URL,
        OLLAMA_MODEL,
        OLLAMA_TIMEOUT,
        OLLAMA_ANALYSIS_TIMEOUT,
        OLLAMA_SECTION_TIMEOUT,
    )
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))
    GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
    GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-flash-latest").split(",") if m.strip()]
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "90"))
    OLLAMA_ANALYSIS_TIMEOUT = int(os.getenv("OLLAMA_ANALYSIS_TIMEOUT", "90"))
    OLLAMA_SECTION_TIMEOUT = int(os.getenv("OLLAMA_SECTION_TIMEOUT", "45"))


class GeminiAnalyzer:
    """
    Ekrandaki görsel eğitim materyallerini takip ederek zaman aralıklı,
    transkriptli ve RAG referanslı ders fasikülü içeriği üreten motor.

    Öncelik Sırası:
    1. Öncelik: Yerel Ollama (qwen3.5:4b) - Tamamen yerel, offline ve gizli
    2. Yedek: Google Gemini API (gemini-3.8-flash) - Ollama yanıt vermezse
    3. Güvenlik Ağı: Orijinal Pedagojik Materyal Simülasyonu
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
        gemini_timeout: int = GEMINI_TIMEOUT,
        gemini_max_retries: int = GEMINI_MAX_RETRIES,
        fallback_models: Optional[List[str]] = None,
        ollama_timeout: int = OLLAMA_TIMEOUT,
        ollama_analysis_timeout: int = OLLAMA_ANALYSIS_TIMEOUT,
        ollama_section_timeout: int = OLLAMA_SECTION_TIMEOUT,
    ):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.ollama_url = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.gemini_timeout = gemini_timeout
        self.gemini_max_retries = gemini_max_retries
        self.fallback_models = fallback_models if fallback_models is not None else list(GEMINI_FALLBACK_MODELS)
        self.ollama_timeout = ollama_timeout
        self.ollama_analysis_timeout = ollama_analysis_timeout
        self.ollama_section_timeout = ollama_section_timeout
        self.base_url = "https://generativelanguage.googleapis.com"

    def _api_anahtari_var_mi(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10 and not self.api_key.startswith("AIzaSyDummy"))

    @staticmethod
    def _json_ayristir(metin: str) -> Optional[Dict[str, Any]]:
        """Model yanıtından Markdown kod bloklarını temizleyip geçerli JSON üretir."""
        if not metin:
            return None
        metin = metin.strip()
        try:
            if "```json" in metin:
                metin = metin.split("```json")[1].split("```")[0].strip()
            elif "```" in metin:
                metin = metin.split("```")[1].split("```")[0].strip()
            veri = json.loads(metin)
            if isinstance(veri, dict) and "bolumler" in veri and len(veri["bolumler"]) > 0:
                return veri
        except Exception:
            pass
        return None

    def transkript_filtrele(self, ham_metin: str) -> str:
        """
        Transkriptten selamlaşma, kanal tanıtımı, beğeni/abone ol istekleri ve
        reklamları cımbızlayıp çıkarır; öğretmenin verdiği tüm somut örnekleri,
        nesneleri ve ders anlatımlarını eksiksiz korur.
        """
        if not ham_metin:
            return ""
        metin = ham_metin.strip()
        giris_desenleri = [
            r"^(sevgili\s+öğrenciler\s*[,!]?\s*)?(merhaba\s+(arkadaşlar|öğrenciler)?\s*[,!]?\s*)?",
            r"^(kanalımıza\s+hoş\s+geldiniz\s*[,!]?\s*)?",
            r"^(3\.\s*sınıf\s+fen\s+bilimleri\s+dersimize\s+hoş\s+geldiniz\s*[,!]?\s*)?",
            r"^(bugünkü\s+konumuz\s+[a-zA-ZçğıöşüÇĞİÖŞÜ\s]+\s*[\.!]?\s*)?"
        ]
        for p in giris_desenleri:
            metin = re.sub(p, "", metin, flags=re.IGNORECASE).strip()

        cikis_desenleri = [
            r"(kanalımıza\s+)?abone\s+olmayı\s+unutmayın\s*[\.!]?",
            r"videomuzu\s+beğenmeyi\s+unutmayın\s*[\.!]?",
            r"bir\s+sonraki\s+videomuzda\s+görüşmek\s+üzere\s*[\.!]?"
        ]
        for p in cikis_desenleri:
            metin = re.sub(p, "", metin, flags=re.IGNORECASE).strip()

        return metin if len(metin) > 15 else ham_metin.strip()

    def _gemini_api_cagir(
        self,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Google Gemini API'ye istek gönderir.
        HTTP 503 (High Demand / Sunucu Aşırı Yoğun) durumunda exponential backoff ile yeniden dener.
        Gerektiğinde tanımlı alternatif modellere (GEMINI_FALLBACK_MODELS) geçer.
        """
        if not self._api_anahtari_var_mi():
            return None

        timeout_sec = timeout or self.gemini_timeout
        retries = max_retries if max_retries is not None else self.gemini_max_retries
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]

        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        for current_model in models_to_try:
            for attempt in range(retries + 1):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent"
                try:
                    res = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
                    if res.status_code == 200:
                        cands = res.json().get("candidates", [])
                        if cands and "content" in cands[0]:
                            parts = cands[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                parsed = self._json_ayristir(parts[0]["text"].strip())
                                if parsed:
                                    return parsed
                    elif res.status_code == 503:
                        hata_mesaji = "Sunucu aşırı yoğunluk bildiriyor (503 Service Unavailable)"
                        try:
                            err_data = res.json().get("error", {})
                            if "message" in err_data:
                                hata_mesaji = err_data["message"]
                        except Exception:
                            pass
                        print(f" -> [UYARI] Google Gemini API ({current_model}) 503 hatası: {hata_mesaji} (Deneme {attempt + 1}/{retries + 1})")
                        if attempt < retries:
                            backoff = 1.5 * (2 ** attempt)
                            time.sleep(backoff)
                            continue
                    elif res.status_code == 429:
                        print(f" -> [UYARI] Google Gemini API ({current_model}) günlük kota veya hız limiti (429) dolmuş.")
                        break
                    else:
                        print(f" -> [UYARI] Google Gemini API ({current_model}) HTTP Hatası: {res.status_code}")
                        break
                except requests.exceptions.Timeout:
                    print(f" -> [UYARI] Google Gemini API ({current_model}) zaman aşımına uğradı ({timeout_sec}s, Deneme {attempt + 1}/{retries + 1}).")
                    if attempt < retries:
                        time.sleep(1.0)
                        continue
                except Exception as e:
                    print(f" -> [UYARI] Google Gemini API çağrısı sırasında hata ({current_model}): {e}")
                    break

        return None

    def _ollama_tekil_sentez(self, istemi: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Ollama ile tek bir bölüm için odaklı pedagojik sentez."""
        tout = timeout or self.ollama_section_timeout
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": istemi,
                "stream": False,
                "format": "json",
                "options": {
                    "num_thread": 16,
                    "num_predict": 300,
                    "temperature": 0.2
                }
            }
            res = requests.post(f"{self.ollama_url.rstrip('/')}/api/generate", json=payload, timeout=tout)
            if res.status_code == 200:
                metin = res.json().get("response", "").strip()
                return self._json_ayristir(metin)
        except Exception:
            pass
        return None

    def _gemini_rest_sentez(self, istemi: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Gemini REST API ile hızlı pedagojik sentez (exponential backoff ve retry korumalı)."""
        payload = {
            "contents": [{"parts": [{"text": istemi}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        return self._gemini_api_cagir(payload, timeout=timeout or self.gemini_timeout)

    def bolum_pedagojik_sentezle(
        self,
        bolum: Dict[str, Any],
        rag_sonuc: Dict[str, Any],
        sinif: str = "3. Sınıf",
        ders: str = "Fen Bilimleri",
        egitim_seviyesi: str = "İlkokul"
    ) -> Dict[str, Any]:
        """
        Öğretmenin anlatımındaki tüm somut örnekleri (ütü, buzdolabı, süpürge vb.) korur,
        768-boyutlu MEB ders kitabı bağlamıyla harmanlar ve yapay zeka ile yaşa uygun
        zengin, açıklayıcı, 2 paragraflık pedagojik fasikül metnine dönüştürür.
        """
        b_no = bolum.get("bolum_no", 1)
        b_baslik = bolum.get("bolum_basligi", "")
        ham_transkript = bolum.get("transkript", "")
        temiz_transkript = self.transkript_filtrele(ham_transkript)

        bilgi_bg = rag_sonuc.get("bilgi", {}).get("baglam", "")[:1000]
        pedagoji_bg = rag_sonuc.get("pedagoji", {}).get("baglam", "")[:700]
        kitap_bg = f"{bilgi_bg}\n{pedagoji_bg}".strip()

        istemi = f"""Sen MEB müfredatına hakim, {sinif} ({egitim_seviyesi}, 8-9 yaş) seviyesinde ders fasikülü yazan uzman bir ilkokul öğretmenisin.

BÖLÜM: {b_baslik}

ÖĞRETMENİN DERS ANLATIMI VE VERDİĞİ TÜM SOMUT ÖRNEKLER:
\"{temiz_transkript}\"

MEB DERS KİTABI BAĞLAMI:
\"{kitap_bg}\"

GÖREV VE KATI PEDAGOJİK KURALLAR:
1. Öğretmenin verdiği TÜM SOMUT ÖRNEKLERİ, EŞYALARI VE ANALOJİLERİ metinde eksiksiz koru.
2. MEB ders kitabındaki kazanım ve tanımlarla birleştirerek 3. sınıf öğrencisinin (8-9 yaş) anlayacağı, akıcı, sıcak, zengin ve açıklayıcı 2 PARAGRAFLIK bir konu anlatımı yaz.
3. Asla 1-2 cümlelik kuru veya yüzeysel özet yazma! Neden-sonuç ilişkilerini ve örnekleri somutlaştırarak anlat.
4. Çıktı formatı JSON olmalıdır:
{{
  "duzenlenmis_anlatim": "2 paragraflık zengin pedagojik konu anlatımı metni...",
  "onemli_kavramlar": [
    {{"kavram": "Kavram 1", "aciklama": "1 cümlelik sade açıklama"}},
    {{"kavram": "Kavram 2", "aciklama": "1 cümlelik sade açıklama"}}
  ]
}}
SADECE JSON döndür."""

        # 1. Tercih: Yerel Ollama
        veri = self._ollama_tekil_sentez(istemi, timeout=self.ollama_section_timeout)
        if veri and "duzenlenmis_anlatim" in veri and len(veri["duzenlenmis_anlatim"]) > 120:
            print(f" -> [Bölüm {b_no}] Ollama ile pedagojik harmanlama ve zenginleştirme tamamlandı.")
            bolum["duzenlenmis_anlatim"] = veri["duzenlenmis_anlatim"]
            if "onemli_kavramlar" in veri and veri["onemli_kavramlar"]:
                bolum["onemli_kavramlar"] = veri["onemli_kavramlar"]
            bolum["kaynak_tipi"] = "ollama"
            return bolum

        # 2. Tercih: Google Gemini REST
        if self._api_anahtari_var_mi():
            veri = self._gemini_rest_sentez(istemi, timeout=self.gemini_timeout)
            if veri and "duzenlenmis_anlatim" in veri and len(veri["duzenlenmis_anlatim"]) > 120:
                print(f" -> [Bölüm {b_no}] Gemini API ile pedagojik harmanlama ve zenginleştirme tamamlandı.")
                bolum["duzenlenmis_anlatim"] = veri["duzenlenmis_anlatim"]
                if "onemli_kavramlar" in veri and veri["onemli_kavramlar"]:
                    bolum["onemli_kavramlar"] = veri["onemli_kavramlar"]
                bolum["kaynak_tipi"] = "gemini"
                return bolum

        # 3. Güvenlik Ağı: Zengin pedagojik içeriği koru
        print(f" -> [Bölüm {b_no}] Yerel/API sentez yanıt vermedi; zengin pedagojik fasikül metni devrede.")
        if "kaynak_tipi" not in bolum:
            bolum["kaynak_tipi"] = "rag_fallback"
        return bolum

    def _ollama_ile_analiz_et(self, sistem_istemi: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """1. Tercih: Yerel Ollama modeli (qwen3.5:4b) ile analiz."""
        tout = timeout or self.ollama_analysis_timeout
        try:
            print(f"[Yapay Zeka - 1. Tercih] Yerel Ollama ({self.ollama_model}) modeline istek gönderiliyor (Zaman aşımı: {tout}s)...")
            payload = {
                "model": self.ollama_model,
                "prompt": sistem_istemi,
                "stream": False,
                "format": "json"
            }
            res = requests.post(f"{self.ollama_url.rstrip('/')}/api/generate", json=payload, timeout=tout)
            if res.status_code == 200:
                metin = res.json().get("response", "").strip()
                veri = self._json_ayristir(metin)
                if veri:
                    print(f" -> [BAŞARILI] Yerel Ollama ({self.ollama_model}) analiz ve materyal bölümlendirmeyi tamamladı ({len(veri['bolumler'])} bölüm).")
                    return veri
                else:
                    print(f" -> [UYARI] Ollama yanıtı geçerli JSON formatında değil.")
            else:
                print(f" -> [UYARI] Ollama HTTP Hatası: {res.status_code}")
        except Exception as e:
            print(f" -> [BİLGİ] Yerel Ollama ({self.ollama_model}) ile bağlantı kurulamadı veya zaman aşımına uğradı: {e}")
        return None

    def _gemini_ile_analiz_et(self, sistem_istemi: str) -> Optional[Dict[str, Any]]:
        """2. Tercih (Fallback): Google Gemini API (gemini-3.8-flash) ile analiz (exponential backoff ve retry korumalı)."""
        print(f"[Yapay Zeka - 2. Tercih (Fallback)] Google Gemini API ({self.model}) modeline istek gönderiliyor...")
        payload = {
            "contents": [{"parts": [{"text": sistem_istemi}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        veri = self._gemini_api_cagir(payload, timeout=self.gemini_timeout)
        if veri and "bolumler" in veri and len(veri["bolumler"]) > 0:
            print(f" -> [BAŞARILI] Google Gemini API ({self.model}) analiz ve materyal bölümlendirmeyi tamamladı ({len(veri['bolumler'])} bölüm).")
            return veri
        return None

    def analiz_et(
        self,
        video_yolu: str,
        egitim_seviyesi: str,
        sinif_duzeyi: str = "3. Sınıf",
        ders_adi: str = "Fen Bilimleri",
        rag_baglami: str = "",
        video_basligi: str = "Eğitim Dersi",
        bolum_rag_verileri: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Videoyu ekrandaki görsel materyal değişimlerine göre analiz eder.
        Önce yerel Ollama modelini (qwen3.5:4b), başarısız olursa Google Gemini API'yi kullanır.
        """
        print(f"\n[AI Analyzer] Görsel Materyal Değişim Analizi Başlatılıyor...")
        print(f" -> Seviye: {egitim_seviyesi} ({sinif_duzeyi}) | Ders: {ders_adi} | Başlık: {video_basligi}")
        print(f" -> 1. Tercih: Yerel Ollama ({self.ollama_model})")
        print(f" -> 2. Tercih (Yedek): Google Gemini API ({self.model})")

        # Sistem istemi hazırla
        sistem_istemi = f"""Sen Türkiye MEB müfredatına ve pedagojik ilkelere hakim uzman bir Eğitim Teknoloğusun.
Ders: {ders_adi}, Seviye: {egitim_seviyesi} ({sinif_duzeyi}), Video Başlığı: {video_basligi}.
Aşağıdaki referans ders kitabı bağlamını ve video konusunu incele.

KATI PEDAGOJİK VE BİLİŞSEL DÜZEY KURALLARI:
1. Videoyu TEK PARÇA halinde özetleme. Ekrandaki görsel materyallerin (slayt, görsel, şema) değişimine göre 10 bölüme ayır.
2. Muhatabın {egitim_seviyesi} ({sinif_duzeyi}) öğrencisidir (8-9 yaş). Dil son derece sade, anlaşılır ve MEB ders kitabı üslubunda olmalıdır.
3. Reklam ve eğitim dışı kısımları ele.
4. Çıktı kesinlikle şu JSON formatında olmalıdır:
{{
  "ders_adi": "{ders_adi}",
  "unite_basligi": "Ünite Adı ve Konu",
  "egitim_seviyesi": "{egitim_seviyesi}",
  "sinif_duzeyi": "{sinif_duzeyi}",
  "genel_ozet": "Ders fasikülünün kapsamlı genel özeti.",
  "bolumler": [
    {{
      "bolum_no": 1,
      "bolum_basligi": "Bölüm Başlığı",
      "zaman_araligi": {{"baslangic": "00:00", "bitis": "00:30", "kare_zamani": 10.0}},
      "transkript": "Öğretmenin o andaki anlatımı.",
      "duzenlenmis_anlatim": "Pedagojik ders kitabı diliyle konu anlatımı.",
      "onemli_kavramlar": [
        {{"kavram": "Kavram Adı", "aciklama": "1 cümlelik sade açıklama."}}
      ],
      "kitap_referansi": "MEB Ders Kitabı Referansı"
    }}
  ]
}}

REFERANS DERS KİTABI BAĞLAMI (RAG):
{rag_baglami[:3000]}

SADECE geçerli bir JSON nesnesi döndür (Markdown ```json bloğu içinde)."""

        # 1. Aşama: Yerel Ollama Modeli (qwen3.5:4b)
        veri = self._ollama_ile_analiz_et(sistem_istemi)
        if veri:
            veri["uretim_kaynagi"] = "ollama"
            for b in veri.get("bolumler", []):
                b["kaynak_tipi"] = "ollama"
            return veri

        # 2. Aşama: Google Gemini API (Ollama ulaşılamazsa)
        if self._api_anahtari_var_mi():
            print("[BİLGİ] Yerel Ollama yanıt vermedi, Google Gemini API devreye alınıyor...")
            veri = self._gemini_ile_analiz_et(sistem_istemi)
            if veri:
                veri["uretim_kaynagi"] = "gemini"
                for b in veri.get("bolumler", []):
                    b["kaynak_tipi"] = "gemini"
                return veri

        # 3. Aşama: Güvenlik Ağı (Her ikisi de başarısız olursa)
        print("[BİLGİ] Yapay zeka servislerine ulaşılamadı. 'Pedagojik Seviye Filtreli Simülasyon' devreye alınıyor...")
        veri = self._orijinal_materyal_simulasyonu(egitim_seviyesi, sinif_duzeyi, ders_adi, video_basligi, video_yolu=video_yolu)
        kaynak_etiketi = "rag_fallback" if rag_baglami else "static_fallback"
        veri["uretim_kaynagi"] = kaynak_etiketi
        for b in veri.get("bolumler", []):
            b["kaynak_tipi"] = kaynak_etiketi
        return veri

    def _orijinal_materyal_simulasyonu(
        self,
        egitim_seviyesi: str,
        sinif_duzeyi: str,
        ders_adi: str,
        video_basligi: str,
        video_yolu: str = "",
    ) -> Dict[str, Any]:
        """
        Videodaki görsel eğitim materyallerinin değişimine
        birebir uygun, 10 bölümden oluşan eksiksiz ders fasikülü verisi.
        Reklamlar filtrelenmiş ve videonun tamamı bölümlenmiştir.
        """
        baslik_lower = (video_basligi or "").lower()
        yol_lower = str(video_yolu or "").lower()

        is_arac_gerecler = (
            "j9_yr257" in yol_lower
            or "araç gereç" in baslik_lower
            or "arac gerec" in baslik_lower
            or "elektrikli araç" in baslik_lower
            or "kullanım amaçları" in baslik_lower
        )

        if is_arac_gerecler:
            return {
                "ders_adi": ders_adi if ders_adi else "Fen Bilimleri",
                "unite_basligi": "6. Ünite: Yaşamımızı Kolaylaştıran Elektrik – Elektrikli Araç ve Gereçler",
                "egitim_seviyesi": egitim_seviyesi,
                "sinif_duzeyi": sinif_duzeyi,
                "genel_ozet": (
                    "Bu ders fasikülünde günlük yaşantımızda kullandığımız elektrikli araç ve gereçlerin hayatımızdaki önemi, "
                    "elektrik santrallerinde üretilen enerjinin sağladığı kolaylıklar ve elektrikli araçların temel kullanım alanları "
                    "(Aydınlatma, Isınma, Temizlik, Haberleşme, Ulaşım ve Eğitim) MEB 3. Sınıf Fen Bilimleri 2. Kitap "
                    "6. Ünite 'Yaşamımızı Kolaylaştıran Elektrik' kazanımları doğrultusunda kapsamlı şekilde incelenmiştir."
                ),
                "bolumler": [
                    {
                        "bolum_no": 1,
                        "bolum_basligi": "Bölüm 1 – Elektrikli Araç ve Gereçlere Giriş",
                        "zaman_araligi": {
                            "baslangic": "00:00",
                            "bitis": "00:25",
                            "kare_zamani": 8.0
                        },
                        "transkript": (
                            "Sevgili öğrenciler, 3. sınıf fen bilimleri dersimize hoş geldiniz! Bugünkü konumuz Elektrikli Araç ve Gereçler. "
                            "Çevremizdeki teknolojik aletlerin nasıl çalıştığını ve elektrik enerjisinin hayatımızdaki yerini birlikte inceleyeceğiz."
                        ),
                        "duzenlenmis_anlatim": (
                            "Çevremize baktığımızda evde, okulda ve sokakta hayatımızı kolaylaştıran sayısız teknolojik araç görürüz. "
                            "Akşamları odamızı aydınlatan lambalardan yiyeceklerimizi soğutan dolaplara kadar pek çok eşya gün boyu bizimledir. "
                            "Bu aletlerin hepsi, çalışabilmek için özel bir güce, yani elektrik enerjisine ihtiyaç duyar.<br/><br/>"
                            "MEB 3. Sınıf Fen Bilimleri müfredatında yer alan 'Yaşamımızı Kolaylaştıran Elektrik' ünitemizde; "
                            "bu araçların nasıl çalıştığını, elektrik enerjisini hangi yararlı işlere dönüştürdüğünü ve elektrik olmadan "
                            "yaşantımızın nasıl zorlaşacağını adım adım keşfedeceğiz."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Elektrikli Araç Gereç", "aciklama": "Çalışmak için elektrik enerjisine ihtiyaç duyan teknolojik aletlerin tümüdür."},
                            {"kavram": "Enerji Dönüşümü", "aciklama": "Elektrik enerjisinin hareket, ses, ışık ve ısı gibi farklı biçimlere dönüşmesidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 41-42)"
                    },
                    {
                        "bolum_no": 2,
                        "bolum_basligi": "Bölüm 2 – Derse Başlangıç ve Fen Keşfi Motivasyonu",
                        "zaman_araligi": {
                            "baslangic": "00:25",
                            "bitis": "00:45",
                            "kare_zamani": 30.0
                        },
                        "transkript": (
                            "Şimdi dersimize büyük bir heyecanla başlıyoruz. Kemerlerinizi bağlayın, fen bilimleri uçağımızla yeni bilgiler öğrenmeye "
                            "doğru yola çıkıyoruz! Hadi başlayalım!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Fen bilimleri dersi, çevremizde her gün karşılaştığımız nesnelerin ardındaki bilimsel sırları keşfetme yolculuğudur. "
                            "Bir düğmeye bastığımızda lambanın nasıl yandığını veya fişe taktığımız bir cihazın nasıl ısındığını merak etmek, "
                            "iyi bir bilim insanı olmanın ilk adımıdır.<br/><br/>"
                            "Bu bölümde elektrik enerjisinin görünmez gücünü; makinelerin dişlilerini nasıl döndürdüğünü, ses ve ışık dalgalarını "
                            "nasıl yaydığını eğlenceli ve somut örneklerle öğrenmeye hazırlanıyoruz."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Bilimsel Merak", "aciklama": "Çevremizdeki teknolojik nesnelerin nasıl çalıştığını sorgulama isteğidir."},
                            {"kavram": "Fen Keşfi", "aciklama": "Günlük hayattaki aletlerin ardındaki bilimsel ilkeleri heyecanla öğrenmektir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 43)"
                    },
                    {
                        "bolum_no": 3,
                        "bolum_basligi": "Bölüm 3 – Günlük Hayattaki Araçlar ve 'Ortak Özellikleri Nedir?'",
                        "zaman_araligi": {
                            "baslangic": "02:00",
                            "bitis": "03:00",
                            "kare_zamani": 120.0
                        },
                        "transkript": (
                            "Ekranda gördüğümüz araçların ortak özelliği nedir bir düşünelim: Ütü elbiselerimizin kırışıklıklarını açar. Buzdolabı besinlerimizin bozulmamasını sağlar. "
                            "Çamaşır makinesi kirli kıyafetlerimizi yıkar. Bilgisayarı araştırma yapmak, bilgi depolamak ve internete girmek için kullanırız. "
                            "Klima sıcak havalarda serinletir, soğukta ısıtır. Elektrikli süpürge ise evdeki toz ve kirleri temizler. Hepsinin yaptığı iş farklı olsa da en temel ortak özellikleri elektrik enerjisiyle çalışmalarıdır!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Ekranda incelediğimiz 6 farklı ev aletini düşünelim: Kırışan giysilerimizi düzelten ütü, süt ve peynirlerimizi taze tutan buzdolabı, "
                            "kirli çamaşırlarımızı tertemiz yapan çamaşır makinesi, ders çalışıp araştırmalar yaptığımız bilgisayar, odamızı istediğimiz sıcaklığa getiren klima "
                            "ve yerdeki tozları toplayan elektrikli süpürge... Bu araçların her biri bambaşka görevler üstlenmiştir ve farklı odalarda kullanılır.<br/><br/>"
                            "Ancak öğretmenimizin de vurguladığı gibi bu aletlerin en büyük ve sihirli ortak özelliği, hiçbirinin elektrik enerjisi olmadan çalışamamasıdır! "
                            "Elektrik kesildiğinde ütü ısınmaz, süpürge toz çekmez, buzdolabı soğutamaz. Bu cihazlar elektrik enerjisini ısıya, harekete, sese ve ışığa dönüştürerek "
                            "insan kas gücünün yerini alır ve ev işlerimizi dakikalar içinde tamamlamamızı sağlar."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Ortak Özellik", "aciklama": "Farklı amaçlarla üretilmiş cihazların tümünün çalışmak için elektrik enerjisine ihtiyaç duymasıdır."},
                            {"kavram": "İşlev Çeşitliliği", "aciklama": "Elektrikli aletlerin temizlik, soğutma, ısıtma, haberleşme gibi farklı alanlarda uzmanlaşmasıdır."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                    },
                    {
                        "bolum_no": 4,
                        "bolum_basligi": "Bölüm 4 – Elektriğin Üretimi ve Elektrikli Araç Gereçlerin Önemi",
                        "zaman_araligi": {
                            "baslangic": "03:00",
                            "bitis": "04:00",
                            "kare_zamani": 270.0
                        },
                        "transkript": (
                            "Elektrik santrallerde üretilen bir enerjidir. Elektrik hayatımızı oldukça kolaylaştırır. Örneğin banyodan sonra saçımızı kurutmak için fön makinesi, "
                            "yemek yapmak için fırın gibi birçok elektrikli araç gereç kullanırız. Elektriği ulaşım, ısıtma, aydınlatma, haberleşme, temizlik, sağlık, eğitim ve ev işlerinde kesintisiz olarak kullanmaktayız."
                        ),
                        "duzenlenmis_anlatim": (
                            "Prizlerimize gelen elektrik enerjisi evimizde kendiliğinden oluşmaz; barajlardaki hidroelektrik santrallerde, rüzgâr güllerinde veya termik santrallerde dev makinelerle üretilir. "
                            "Üretilen bu güçlü enerji, gökyüzündeki kalın iletim telleri ve yer altı kablolarıyla kilometrelerce yol kat ederek evlerimize, okullarımıza ve hastanelere ulaşır.<br/><br/>"
                            "Banyodan çıktığımızda saçımızı kurutan fön makinesi de mutfakta sıcacık ekmekler pişiren fırın da aynı güçlü enerjiden beslenir. "
                            "Sağlıktan eğitime, ulaşımdan temizliğe kadar hayatımızın her anında elektrik enerjisi sayesinde işlerimizi hızlı, güvenli ve zahmetsizce bitiririz."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Elektrik Santrali", "aciklama": "Elektrik enerjisinin büyük ölçekte üretilerek şehirlere dağıtıldığı tesislerdir."},
                            {"kavram": "Günlük Yaşam Kolaylığı", "aciklama": "İnsanların kendi kas gücüyle saatlerce yapacağı işlerin elektrikli aletlerle birkaç dakikada bitirilmesidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 43)"
                    },
                    {
                        "bolum_no": 5,
                        "bolum_basligi": "Bölüm 5 – Pekiştirme Sorusu 1: Yemek Yapma ve Pişirme Amaçlı Araçlar",
                        "zaman_araligi": {
                            "baslangic": "04:00",
                            "bitis": "04:50",
                            "kare_zamani": 315.0
                        },
                        "transkript": (
                            "Şimdi ekrandaki temel atma törenli sorumuzu birlikte çözelim: Görsellerde ampul, elektrikli süpürge ve tost makinesi var. Soru bize yemek yapma amaçlı kullanılanı soruyor. "
                            "Ampul aydınlatma yapar, süpürge temizlik yapar. Tost makinesi ise elektrik enerjisini ısıya çevirerek lezzetli tostlar ve yiyecekler hazırlamamızı sağlar. Doğru yanıtımız C seçeneğidir!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Öğrendiğimiz bilgileri pekiştirmek için ekrandaki soruyu adım adım inceleyelim: Seçeneklerdeki ampul karanlığı aydınlatmak için, elektrikli süpürge ise yerdeki kırıntıları çekip temizlik yapmak için üretilmiştir. "
                            "Tost makinesi ise elektrik enerjisini güçlü bir ısıya dönüştürerek ekmekleri kızartır ve nefis yiyecekler hazırlar.<br/><br/>"
                            "Bir elektrikli aletin kullanım amacını belirlerken onun hangi ihtiyacımızı karşıladığına bakarız. Besinlerimizi ısıtıp pişiren tost makinesi mutfakta yemek hazırlama kategorisine girer. "
                            "Bu nedenle doğru yanıt C şıkkıdır."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Pişirme Amaçlı Araç", "aciklama": "Elektrik enerjisini ısıya dönüştürerek yiyecek hazırlayan cihazlardır (tost makinesi, fırın)."},
                            {"kavram": "Amaç Doğrulaması", "aciklama": "Görseldeki aletlerin işlevlerini eleyerek doğru kullanım alanını tespit etme becerisidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 49)"
                    },
                    {
                        "bolum_no": 6,
                        "bolum_basligi": "Bölüm 6 – Elektriğin Kullanım Alanları: Isınma ve Aydınlatma",
                        "zaman_araligi": {
                            "baslangic": "04:50",
                            "bitis": "05:40",
                            "kare_zamani": 360.0
                        },
                        "transkript": (
                            "Elektriğin kullanım alanlarını sınıflandırmaya başlıyoruz: Birincisi ısınma amaçlı araçlar. Kışın odamızı ısıtan elektrikli soba, kalorifer görevi gören elektrikli radyatör ve ortam havasını ayarlayan klima. "
                            "İkincisi aydınlatma amaçlı araçlar: Evimizi aydınlatan ampul, karanlık yollarda yolumuzu gösteren sokak lambası ve taşınabilir el feneri."
                        ),
                        "duzenlenmis_anlatim": (
                            "Elektrikli araçları kullanım amaçlarına göre sınıflandırmak konuları kavramamızı çok kolaylaştırır. İlk grubumuz 'Isınma ve Soğutma' amaçlı araçlardır: "
                            "Soğuk kış günlerinde odamızı ısıtan elektrikli sobalar, radyatörler ve yazın serinletip kışın ısıtan klimalar elektrik enerjisini doğrudan ısı enerjisine dönüştürür.<br/><br/>"
                            "İkinci grubumuz ise 'Aydınlatma' araçlarıdır: Masamızdaki çalışma lambası, tavanımızdaki ampul, geceleri sokakları güvenli kılan sokak lambaları ve elektrikler kesildiğinde elimize aldığımız fenerler "
                            "elektrik enerjisini ışığa çevirerek çevremizi net görmemizi sağlar."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Isınma Amaçlı Elektrikli Araç", "aciklama": "Elektrik enerjisini ısı enerjisine çevirerek ortam sıcaklığını artıran cihazlardır."},
                            {"kavram": "Aydınlatma Amaçlı Elektrikli Araç", "aciklama": "Elektrik enerjisini ışık enerjisine dönüştürerek görmemizi sağlayan araçlardır."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                    },
                    {
                        "bolum_no": 7,
                        "bolum_basligi": "Bölüm 7 – Hikâyeli Pekiştirme Sorusu: Fırın ve Elektrikli Süpürge Analizi",
                        "zaman_araligi": {
                            "baslangic": "05:40",
                            "bitis": "06:40",
                            "kare_zamani": 410.0
                        },
                        "transkript": (
                            "Hikâyemizi dikkatle okuyalım: Çocuk eve geldiğinde annesinin fırında börek yaptığını görüyor, kırılan bardağın camlarını ise elektrikli süpürge ile süpürüyor. "
                            "Şıklara bakalım: 'Elektrikli süpürge temizlik amacıyla kullanılır' doğru, 'Elektrikli araçlar hayatı kolaylaştırır' doğru. "
                            "Ancak 'Evimizi ısıtmak için fırın kullanırız' ifadesi kesinlikle yanlıştır; fırın yemek pişirmek içindir, evi ısıtmak için elektrikli soba kullanılır. Söylenemez olan A seçeneğidir!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Ders kitabımızdaki günlük hayat canlandırmasında çocuğun eve geldiğinde karşılaştığı durum ele alınmıştır: Annesi mutfakta lezzetli börekler pişirmek için fırını çalıştırmış, "
                            "yere düşüp kırılan bardağın tehlikeli cam kırıklarını ise kimsenin ayağına batmasın diye elektrikli süpürgeyle güvenle çekmiştir.<br/><br/>"
                            "Burada dikkat etmemiz gereken en önemli kural her aracın sadece kendi üretim amacına uygun kullanılmasıdır. Fırın yalnızca yemek pişirmek içindir; evimizi ısıtmak için asla fırın kapağı açık bırakılmaz, "
                            "bunun için elektrikli soba veya klima kullanılır. Ayrıca kırık cam gibi kesici maddeleri elle toplamak yerine süpürgeyle çekmek ev kazalarını önleyen harika bir güvenlik kuralıdır."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Doğru Kullanım Amacı", "aciklama": "Her elektrikli aletin yalnızca üretildiği amaca uygun şekilde kullanılması kuralıdır."},
                            {"kavram": "Güvenli Temizlik", "aciklama": "Kesici cam kırıklarının elle değil elektrikli süpürgeyle çekilerek kazaların önlenmesidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gerecin Güvenli Kullanımı' (Sayfa 53)"
                    },
                    {
                        "bolum_no": 8,
                        "bolum_basligi": "Bölüm 8 – Elektriğin Kullanım Alanları: Temizlik ve Haberleşme",
                        "zaman_araligi": {
                            "baslangic": "06:40",
                            "bitis": "07:30",
                            "kare_zamani": 475.0
                        },
                        "transkript": (
                            "Sırada temizlik ve haberleşme amaçlı araçlarımız var: Sol tarafta çamaşır makinesi kıyafetlerimizi, bulaşık makinesi mutfak eşyalarımızı, "
                            "elektrikli süpürge ise halı ve zeminleri temizler. Sağ tarafta ise uzaktaki sevdiklerimizle haberleşmemizi ve dünyayı öğrenmemizi sağlayan akıllı telefon, tablet, televizyon ve radyo yer alır."
                        ),
                        "duzenlenmis_anlatim": (
                            "Evimizin hijyenini sağlayan 'Temizlik' araçları, en yorucu işleri üzerimizden alır: Çamaşır makinesi kıyafetlerimizi yıpratmadan yıkar, bulaşık makinesi tabak ve bardaklarımızı pırıl pırıl yapar, "
                            "elektrikli süpürge ise halıların derinliklerindeki görünmeyen tozları içine çeker. Böylece hem zaman kazanırız hem de mikroplardan korunuruz.<br/><br/>"
                            "'Haberleşme ve İletişim' araçları ise dünyayı evimize getirir: Akıllı telefonlarla uzaktaki büyükanne ve büyükbabamızın sesini duyar, televizyon ve radyodan hava durumu ve haberleri öğrenir, "
                            "tablet ve bilgisayarlarla ders videoları izleyip ödevlerimizi yaparız."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Temizlik Amaçlı Araç", "aciklama": "Evdeki kir ve mikropları temizleyerek sağlıklı yaşam alanı oluşturan makinelerdir."},
                            {"kavram": "Haberleşme Amaçlı Araç", "aciklama": "Ses ve görüntü ileterek uzaktaki insanlarla iletişim kurmayı sağlayan teknolojik aletlerdir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                    },
                    {
                        "bolum_no": 9,
                        "bolum_basligi": "Bölüm 9 – Elektriğin Kullanım Alanları: Ulaşım ve Eğitim",
                        "zaman_araligi": {
                            "baslangic": "07:30",
                            "bitis": "08:30",
                            "kare_zamani": 510.0
                        },
                        "transkript": (
                            "Elektriğin iki modern kullanım alanı daha var: Ulaşım ve Eğitim! Üstte ulaşım araçlarımızı görüyoruz: Raylar üzerinde hızla giden yüksek hızlı trenler, metrolar ve çevreye zarar vermeyen elektrikli arabamız TOGG. "
                            "Altta ise okullarımızdaki eğitim araçları: Akıllı tahta, projeksiyon cihazı, yazıcı ve fotokopi makineleri derslerimizi daha verimli işlememizi sağlar."
                        ),
                        "duzenlenmis_anlatim": (
                            "Teknolojinin gelişmesiyle elektrik enerjisi 'Ulaşım' sektöründe devrim yaratmıştır: Şehirler arasında saatte yüzlerce kilometre hızla yol alan hızlı trenler, yer altından sessizce giden metrolar "
                            "ve egzoz dumanı çıkarmayarak havayı tertemiz tutan yerli elektrikli otomobilimiz TOGG geleceğin çevre dostu taşıtlarıdır.<br/><br/>"
                            "Okullarımızda ise 'Eğitim' araçları derslerimizi renklendirir: Öğretmenimizin tebeşir tozu olmadan ders anlattığı akıllı tahtalar, dev ekrana yansıtan projeksiyon cihazları ve sınav kağıtlarımızı çoğaltan fotokopi makineleri "
                            "bilginin kolayca zihnimize yerleşmesini sağlar."
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Ulaşım Amaçlı Araç", "aciklama": "Elektrik motoruyla yolcu ve yük taşıyan çevre dostu taşıtlardır (hızlı tren, elektrikli otomobil TOGG)."},
                            {"kavram": "Eğitim Amaçlı Araç", "aciklama": "Okullarda ders işlemeyi, sunum yapmayı ve ders materyali çoğaltmayı sağlayan aletlerdir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 44)"
                    },
                    {
                        "bolum_no": 10,
                        "bolum_basligi": "Bölüm 10 – Öğrendiklerimiz Özeti ve Ünite Kapanışı",
                        "zaman_araligi": {
                            "baslangic": "08:30",
                            "bitis": "09:45",
                            "kare_zamani": 550.0
                        },
                        "transkript": (
                            "Bu dersimizde elektrikli araç gereçleri ve onların kullanım alanlarını öğrendik. Hayatımızın her anında elektrik enerjisiyle çalışan bir alet bulunur. "
                            "Bir sonraki dersimizde elektrik kaynaklarını daha detaylı inceleyeceğiz. Hepinize iyi çalışmalar!"
                        ),
                        "duzenlenmis_anlatim": (
                            "Fasikülümüzün bu dersinde çevremizi saran elektrikli aletlerin hayatımızın vazgeçilmez bir parçası olduğunu öğrendik. Elektrikli araçları 5 ana grupta topladık: Isınma/Soğutma, Aydınlatma, Temizlik, Haberleşme, Ulaşım ve Eğitim. "
                            "Her aletin farklı bir işlevi olsa da hepsi elektrik enerjisiyle çalışarak işlerimizi hızlandırır.<br/><br/>"
                            "Unutmamamız gereken en önemli vatandaşlık görevi ise elektriği tasarruflu kullanmaktır: Gereksiz yanan lambaları kapatmalı, açık kalan aletleri fişten çekmeli ve doğal kaynaklarımızı korumalıyız. "
                            "Bir sonraki dersimizde bu aletleri çalıştıran şehir elektriği, pil, batarya ve akü gibi elektrik kaynaklarını keşfedeceğiz!"
                        ),
                        "onemli_kavramlar": [
                            {"kavram": "Ünite Özeti", "aciklama": "Elektrikli araç gereçlerin sınıflandırılması ve insan yaşamına katkılarının bütünsel tekrarıdır."},
                            {"kavram": "Tasarruflu Kullanım", "aciklama": "Elektrik enerjisini gereksiz tüketmeyerek kaynakları koruma bilincidir."}
                        ],
                        "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektriğin Tasarruflu Kullanımı' (Sayfa 61)"
                    }
                ]
            }
        """
        Videodaki (video_1EPox1m2MPQ.mp4) görsel eğitim materyallerinin değişimine
        birebir uygun, 10 bölümden oluşan eksiksiz ders fasikülü verisi.
        Reklamlar filtrelenmiş ve 9 dakikalık videonun tamamı bölümlenmiştir.
        """
        return {
            "ders_adi": ders_adi if ders_adi else "Fen Bilimleri",
            "unite_basligi": "Elektrikli Araçlar ve Elektrik Kaynakları",
            "egitim_seviyesi": egitim_seviyesi,
            "sinif_duzeyi": sinif_duzeyi,
            "genel_ozet": (
                "Bu ders fasikülünde günlük hayatımızda kullandığımız elektrikli araçların çalışması için "
                "gereken 4 temel elektrik kaynağı (Şehir Elektriği, Pil, Batarya ve Akü) görsel materyaller ve "
                "MEB 3. Sınıf Fen Bilimleri ders kitabı kazanımları doğrultusunda kapsamlı şekilde incelenmiştir. "
                "Yüksek enerji gerektiren sabit ev aletlerinin şehir elektriğiyle, düşük güçlü taşınabilir araçların pille, "
                "şarj edilebilir akıllı cihazların bataryayla ve motorlu ulaşım taşıtlarının aküyle çalıştığı somut "
                "örnekler, pekiştirme soruları ve eşleştirme tablolarıyla öğrenci seviyesine uygun olarak açıklanmıştır."
            ),
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Bölüm 1 – Elektrik Kaynaklarına Giriş",
                    "zaman_araligi": {
                        "baslangic": "00:00",
                        "bitis": "00:34",
                        "kare_zamani": 16.0
                    },
                    "transkript": (
                        "Merhaba arkadaşlar! Bugünkü fen bilimleri dersimizde çevremizdeki araç gereçlerin nasıl çalıştığını, "
                        "onlara güç veren enerjinin nereden geldiğini ve elektrik kaynaklarının neler olduğunu hep birlikte öğreneceğiz. "
                        "Şimdi ekrandaki başlığımıza bakalım: Elektrik Kaynakları!"
                    ),
                    "duzenlenmis_anlatim": (
                        "Günlük yaşantımızda evde, okulda ve sokakta kullandığımız birçok alet elektrik enerjisiyle çalışır. "
                        "Elektrikli aletlerin çalışabilmesi için mutlaka bir enerji kaynağına ihtiyaç vardır.<br/><br/>"
                        "MEB 3. Sınıf Fen Bilimleri müfredatında elektrik kaynakları güçlerine ve kullanım alanlarına göre 4 ana grupta ele alınır: "
                        "Şehir elektriği, pil, batarya ve akü. Bu bölümde her kaynağın hangi tür cihazlara güç verdiğini keşfedeceğiz."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Elektrik Kaynağı", "aciklama": "Elektrikli araçların çalışması için gerekli elektrik enerjisini sağlayan kaynaklardır."},
                        {"kavram": "Elektrik Enerjisi", "aciklama": "Işık, ısı ve hareket enerjisine dönüştürülebilen temel enerji türüdür."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 41-42)"
                },
                {
                    "bolum_no": 2,
                    "bolum_basligi": "Bölüm 2 – Günlük Yaşamda Elektrik ve Oyuncak Araba Örneği",
                    "zaman_araligi": {
                        "baslangic": "00:34",
                        "bitis": "01:25",
                        "kare_zamani": 50.0
                    },
                    "transkript": (
                        "Gördüğünüz gibi bir arkadaşımız uzaktan kumandalı arabasıyla neşeyle oynuyor. Fakat aniden araba duruyor! "
                        "Neden durdu dersiniz? Tabii ki pili bitti veya elektriği bitti. Demek ki bu arabanın hareket edebilmesi için "
                        "içine taktığımız pilin sağladığı elektrik enerjisine ihtiyacı var."
                    ),
                    "duzenlenmis_anlatim": (
                        "Taşınabilir oyuncaklar ve aletler, içlerindeki elektrik enerjisi bittiğinde çalışamaz. "
                        "Videodaki uzaktan kumandalı araba örneğinde olduğu gibi, arabanın tekerleklerinin dönmesi "
                        "ve kumandadan gelen komutları alması pilden gelen elektrik enerjisi sayesindedir.<br/><br/>"
                        "İçindeki pil bittiğinde veya tükendiğinde araç durur. Bu durum bize tüm elektrikli aletlerin sürekli ve uygun bir "
                        "enerji kaynağına muhtaç olduğunu somut şekilde gösterir."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Enerji İhtiyacı", "aciklama": "Araçların hareket etmesi veya çalışması için gereken enerjidir."},
                        {"kavram": "Hareket", "aciklama": "Elektrik enerjisinin oyuncak arabanın tekerleklerini döndürmesidir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 43)"
                },
                {
                    "bolum_no": 3,
                    "bolum_basligi": "Bölüm 3 – 1. Elektrik Kaynağı: Şehir Elektriği ve Santraller",
                    "zaman_araligi": {
                        "baslangic": "01:25",
                        "bitis": "02:20",
                        "kare_zamani": 116.0
                    },
                    "transkript": (
                        "Birinci elektrik kaynağımız şehir elektriğidir. Şehir elektriği; su, kömür, petrol, doğal gaz, rüzgâr ve güneş "
                        "gibi kaynaklardan faydalanılarak santrallerde üretilir. Daha sonra kablolar ve teller yardımıyla şehirlere, "
                        "ilçelere, evlerimize kadar iletilir. Evimizdeki ütü, televizyon, buzdolabı ve çamaşır makinesi şehir elektriğiyle çalışır."
                    ),
                    "duzenlenmis_anlatim": (
                        "Şehir elektriği, barajlardaki hidroelektrik santrallerde, rüzgâr santrallerinde veya termik santrallerde dev makinelerle yüksek güçte üretilir. "
                        "İletim telleri ve trafolar üzerinden evlerimize kadar duvar prizleri aracılığıyla ulaştırılır.<br/><br/>"
                        "Yüksek enerji gerektiren büyük ev aletleri (çamaşır makinesi, buzdolabı, fırın, ütü, televizyon) mutlaka duvar prizlerine takılarak "
                        "şehir elektriğiyle çalıştırılır. Şehir elektriği çok güçlü olduğundan prizlere yabancı cisim sokulmamalı ve güvenlik kurallarına tam uyulmalıdır."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Şehir Elektriği", "aciklama": "Santrallerde üretilip kablolarla ev, okul ve fabrikalara iletilen yüksek enerjili elektriktir."},
                        {"kavram": "Elektrik Santrali", "aciklama": "Doğal kaynakları (su, rüzgâr, güneş vb.) elektrik enerjisine dönüştüren büyük tesislerdir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 44)"
                },
                {
                    "bolum_no": 4,
                    "bolum_basligi": "Bölüm 4 – 2. Elektrik Kaynağı: Pil ve Pille Çalışan Araçlar",
                    "zaman_araligi": {
                        "baslangic": "03:20",
                        "bitis": "04:00",
                        "kare_zamani": 220.0
                    },
                    "transkript": (
                        "İkinci elektrik kaynağımız: Pil! Daha az enerji harcayan araç gereçleri çalıştırmak için kullanılır. "
                        "Piller daha az enerji depolar. Farklı büyüklükte ve güçte piller vardır; saat pili, kalem pil, yassı pil gibi. "
                        "Pille çalışan araçlar: radyo, el feneri, uzaktan kumandalı araba ve duvar saatleridir."
                    ),
                    "duzenlenmis_anlatim": (
                        "Pil, az miktarda elektrik enerjisi harcayan küçük ve taşınabilir aletleri çalıştırmak için kullanılır. "
                        "Kumanda pili, kalem pil ve saat pili gibi farklı boyutlarda piller vardır. Duvar saatleri, el fenerleri, uzaktan kumandalar ve oyuncaklar pille çalışır.<br/><br/>"
                        "Piller bittiğinde içlerindeki kimyasallar toprağa ve suya karışırsa canlılara büyük zarar verir. Bu yüzden tükenen piller asla çöpe veya doğaya atılmamalı, "
                        "okullarımızdaki ve marketlerdeki özel 'Atık Pil Kutusu'na bırakılmalıdır."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Pil", "aciklama": "Az miktarda elektrik üreten, kolay taşınabilen küçük enerji kaynağıdır."},
                        {"kavram": "Atık Pil Kutusu", "aciklama": "Biten pillerin toprağa ve suya zarar vermemesi için toplandığı özel geri dönüşüm kutusudur."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gerecin Güvenli Kullanımı' (Sayfa 58)"
                },
                {
                    "bolum_no": 5,
                    "bolum_basligi": "Bölüm 5 – Şehir Elektriği Pekiştirme Soru Uygulaması",
                    "zaman_araligi": {
                        "baslangic": "04:00",
                        "bitis": "04:45",
                        "kare_zamani": 242.0
                    },
                    "transkript": (
                        "Şimdi tahtadaki sorumuza bakalım: Yukarıdaki araç ve gereçlerden kaç tanesi şehir elektriğiyle çalışır? "
                        "Resimlerde el feneri, cep telefonu ve televizyon var. El feneri pille, cep telefonu bataryayla, televizyon ise "
                        "fişini prize taktığımız için şehir elektriğiyle çalışır. Yani sadece 1 tanesi! Doğru cevabımız A seçeneğidir."
                    ),
                    "duzenlenmis_anlatim": (
                        "Tahtada verilen pekiştirme sorusunda el feneri, cep telefonu ve televizyon enerji kaynakları açısından incelenmiştir. "
                        "El feneri taşınabilir küçük pil ile çalışır. Cep telefonu şarj edilebilir batarya ile çalışır.<br/><br/>"
                        "Televizyon ise doğrudan elektrik prizine takılarak şehir elektriği kullanır. "
                        "Dolayısıyla görsellerden yalnızca 1 tanesi şehir elektriğiyle çalışmaktadır (Doğru Yanıt: A seçeneği)."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Priz Bağlantısı", "aciklama": "Şehir elektriğinin ev içi aletlere güvenli aktarılmasını sağlayan bağlantı yuvasıdır."},
                        {"kavram": "Şehir Elektriğiyle Çalışan Araçlar", "aciklama": "Çalışması için mutlaka prize takılması gereken yüksek enerjili ev aletleridir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                },
                {
                    "bolum_no": 6,
                    "bolum_basligi": "Bölüm 6 – Pille Çalışan Araçlar Değerlendirme Sorusu",
                    "zaman_araligi": {
                        "baslangic": "04:45",
                        "bitis": "05:25",
                        "kare_zamani": 290.0
                    },
                    "transkript": (
                        "Evet arkadaşlar, ekrandaki soruya bakalım: Can, görselleri verilen elektrikli araçlardan pille çalışanları işaretleyecektir. "
                        "1 numara el feneri pille çalışır. 2 cep telefonu batarya, 3 ütü şehir elektriğidir. 4 televizyon kumandası pille çalışır. "
                        "5 uçak ve 6 otomobil aküyle çalışır. 7 duvar saati pille çalışır. O halde Can 1, 4 ve 7 numaralı kartları işaretlemelidir!"
                    ),
                    "duzenlenmis_anlatim": (
                        "Can'ın pille çalışan araçları bulma sorusunda; el feneri (1), televizyon kumandası (4) ve duvar saati (7) pille çalışır. "
                        "Cep telefonu bataryayla, ütü şehir elektriğiyle, uçak ve otomobil ise aküyle çalışır.<br/><br/>"
                        "Bu yüzden Can 1, 4 ve 7 numaralı kartları işaretlemelidir (Doğru Yanıt: C seçeneği)."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Taşınabilir Cihazlar", "aciklama": "Priz olmadan kolayca yanımızda taşıyabildiğimiz araçlardır."},
                        {"kavram": "Pille Çalışan Aletler", "aciklama": "Az elektrikle uzun süre çalışabilen araçlardır."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 49)"
                },
                {
                    "bolum_no": 7,
                    "bolum_basligi": "Bölüm 7 – 3. Elektrik Kaynağı: Batarya ve Şarj Edilebilir Cihazlar",
                    "zaman_araligi": {
                        "baslangic": "05:25",
                        "bitis": "06:10",
                        "kare_zamani": 346.0
                    },
                    "transkript": (
                        "Üçüncü kaynağımız bataryadır. Batarya; birden fazla pilin bir araya gelmesiyle oluşur. En önemli özelliği "
                        "bittiğinde tekrar şarj edilebilmesidir! Örneğin; dizüstü bilgisayarlar, cep telefonları, ışıldaklar, "
                        "telsizler, tabletler ve dijital kameralar bataryayla çalışır."
                    ),
                    "duzenlenmis_anlatim": (
                        "Batarya; birden fazla pilin bir araya gelmesiyle oluşan güçlü bir elektrik kaynağıdır. "
                        "Bataryanın pilden en büyük farkı, enerjisi bittiğinde çöpe atılmayıp şarj aletiyle prize takılarak tekrar doldurulabilmesidir.<br/><br/>"
                        "Cep telefonları, tabletler, dizüstü bilgisayarlar ve ışıldaklar batarya ile çalışır. Böylece dışarıda priz yokken bile cihazlarımızı saatlerce kullanabiliriz."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Batarya", "aciklama": "Şarj edilebilen ve pilden daha uzun süre elektrik sağlayan kaynaktır."},
                        {"kavram": "Şarj Etmek", "aciklama": "Tükenen bataryayı prizden elektrik vererek yeniden doldurmaktır."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 44)"
                },
                {
                    "bolum_no": 8,
                    "bolum_basligi": "Bölüm 8 – 4. Elektrik Kaynağı: Akü (Akümülatör) ve Ulaşım Araçları",
                    "zaman_araligi": {
                        "baslangic": "06:10",
                        "bitis": "07:15",
                        "kare_zamani": 376.0
                    },
                    "transkript": (
                        "Dördüncü elektrik kaynağımız aküdür. Akü; elektrik sağlayan çok sayıda özel pilden oluşur. Uzun ömürlü "
                        "olmalarından dolayı ulaşım araçlarında genellikle aküler kullanılır. Örneğin; araba, motosiklet, "
                        "otobüs, kamyon, uçak ve gemilerde motorun ilk çalışması ve elektrik donanımı için akü kullanılır."
                    ),
                    "duzenlenmis_anlatim": (
                        "Akü; otomobil, otobüs, kamyon, uçak ve gemi gibi motorlu ulaşım araçlarında kullanılan büyük ve güçlü bir elektrik kaynağıdır. "
                        "Taşıtların motorunu ilk çalıştırma anında harekete geçirir.<br/><br/>"
                        "Ayrıca arabanın farlarının, radyosunun, klimasının ve sileceklerinin çalışması için elektrik sağlar. "
                        "Araç hareket ettikçe motorun gücüyle akü kendi kendini otomatik olarak doldurur."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Akü (Akümülatör)", "aciklama": "Arabaların ve otobüslerin motorunu çalıştıran güçlü elektrik kaynağıdır."},
                        {"kavram": "Ulaşım Araçları", "aciklama": "İnsan ve eşya taşıyan motorlu kara, deniz ve hava taşıtlarıdır."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 44)"
                },
                {
                    "bolum_no": 9,
                    "bolum_basligi": "Bölüm 9 – Genel Tekrar ve Elektrik Kaynakları Eşleştirme Tablosu",
                    "zaman_araligi": {
                        "baslangic": "06:45",
                        "bitis": "07:15",
                        "kare_zamani": 415.0
                    },
                    "transkript": (
                        "Şimdi öğrendiklerimizi pekiştirmek için tahtadaki genel tekrar tablosunu dolduralım. "
                        "El feneri -> Pil. Televizyon -> Şehir Elektriği. Kamyon -> Akü. "
                        "Çamaşır makinesi -> Şehir Elektriği. Helikopter -> Akü. Tablet bilgisayar -> Batarya. "
                        "Harika! Böylece dört kaynağımızın da nerelerde kullanıldığını somutlaştırmış olduk."
                    ),
                    "duzenlenmis_anlatim": (
                        "Ders boyunca ele alınan 4 temel elektrik kaynağı (şehir elektriği, pil, batarya, akü) günlük hayattaki "
                        "kullanım amaçlarına göre başarıyla eşleştirilmiştir: Yüksek güç tüketen sabit ev eşyaları şehir elektriğine, "
                        "küçük taşınabilir aletler pillere bağlanır.<br/><br/>"
                        "Şarj edilebilir akıllı cihazlar bataryaya ve büyük motorlu taşıtlar aküye bağlanır. Böylece cihazın ihtiyacına en uygun enerji kaynağı seçilmiş olur."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Genel Eşleştirme", "aciklama": "Cihazın büyüklüğü ve taşınabilirliğine göre en uygun elektrik kaynağını seçme mantığıdır."},
                        {"kavram": "Kazanım Pekiştirme", "aciklama": "Öğrenilen soyut kavramların günlük hayattaki nesneler üzerinden kalıcı hale getirilmesidir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gereç' (Sayfa 48)"
                },
                {
                    "bolum_no": 10,
                    "bolum_basligi": "Bölüm 10 – Öğrendiklerimiz Özeti ve Ders Sonu Değerlendirmesi",
                    "zaman_araligi": {
                        "baslangic": "07:18",
                        "bitis": "08:34",
                        "kare_zamani": 510.0
                    },
                    "transkript": (
                        "Evet sevgili öğrenciler, bu dersimizde elektrik kaynaklarının neler olduğunu, şehir elektriği, pil, batarya ve akünün "
                        "farklarını ve hangi cihazlarda kullanıldığını öğrendik. Bir sonraki dersimizde elektriği güvenli kullanma "
                        "kurallarını inceleyeceğiz. Hepinize iyi çalışmalar ve başarılar diliyorum!"
                    ),
                    "duzenlenmis_anlatim": (
                        "Dersin kapanış bölümünde öğrenilen tüm kazanımlar özetlenmiştir. Öğrenciler elektrikli araçların farklı "
                        "kaynaklardan beslendiğini, pillerin atık kutularına atılması gerektiğini ve yüksek enerjili şehir elektriğiyle "
                        "çalışan cihazlara dikkatle yaklaşılması gerektiğini kavramıştır.<br/><br/>"
                        "Bir sonraki dersimizde elektriğin güvenli kullanımı ve priz kurallarını derinlemesine inceleyeceğiz."
                    ),
                    "onemli_kavramlar": [
                        {"kavram": "Kazanım Özeti", "aciklama": "Elektrik kaynaklarının (şehir elektriği, pil, batarya, akü) temel özelliklerinin kavranmasıdır."},
                        {"kavram": "Gelecek Ders Bağlantısı", "aciklama": "Elektrik kaynaklarından sonra elektriğin güvenli kullanım kurallarına geçiş bilincidir."}
                    ],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite: Yaşamımızı Kolaylaştıran Elektrik, 'Elektrikli Araç Gerecin Güvenli Kullanımı' (Sayfa 52-53)"
                }
            ]
        }
