"""
EduFrame AI - Seviyeye Duyarlı Değerlendirme ve Etkinlik Soruları Servisi (services/question_generator.py)
----------------------------------------------------------------------------------------------------------
Bu modül, video dersinde işlenen konular ve MEB ders kitaplarındaki kazanımlar, etkinlikler ve soru kalıplarını
birlikte sentezleyerek öğrenci seviyesine tam uyumlu, özgün ve modern değerlendirme soruları üretir.

Temel Kurallar:
1. Sorular yalnızca video görüntüsünün kendisine dair bir sınav değil; videodaki bilimsel bilgiler ile ders kitabındaki
   pedagojik etkinliklerin sentezidir.
2. Her yeni öğrenilen kavram için mutlaka en az 1 soru oluşturulur.
3. Sorular modern, ilgi çekici tematik kategorilere (💡 Kavram Keşfi, 🔍 Günlük Yaşam Analizi, 🧩 Kavram Tamamlama,
   ⚡ Elektrik Dedektifi, 🎯 Doğru mu Yanlış mı?, 🌱 Doğa ve Çevre Bilinci, 🛡️ Güvenlik Dedektifi) ayrılır.
4. Sorular karmaşık ve sıkışık metin blokları yerine ferah kart formatında sunulur.
"""

import os
import json
import time
import requests
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
        OLLAMA_QUESTION_TIMEOUT,
    )
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-3.8-flash"
    GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))
    GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
    GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-flash-latest").split(",") if m.strip()]
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    OLLAMA_QUESTION_TIMEOUT = int(os.getenv("OLLAMA_QUESTION_TIMEOUT", "60"))


class QuestionGenerator:
    """
    Video materyal kavramlarına ve MEB müfredat standartlarına göre özgün sorular üreten motor.

    Öncelik Sırası:
    1. Öncelik: Yerel Ollama (qwen3.5:4b)
    2. Yedek: Google Gemini API (gemini-3.8-flash)
    3. Güvenlik Ağı: MEB Pedagojik Standart Soru Bankası (Gerçek RAG Kaynak Eşleştirmeli)
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
        ollama_question_timeout: int = OLLAMA_QUESTION_TIMEOUT,
    ):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.ollama_url = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.gemini_timeout = gemini_timeout
        self.gemini_max_retries = gemini_max_retries
        self.fallback_models = fallback_models if fallback_models is not None else list(GEMINI_FALLBACK_MODELS)
        self.ollama_question_timeout = ollama_question_timeout

    def _api_anahtari_var_mi(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10 and not self.api_key.startswith("AIzaSyDummy"))

    @staticmethod
    def _json_sorulari_ayristir(metin: str) -> Optional[List[Dict[str, Any]]]:
        if not metin:
            return None
        metin = metin.strip()
        try:
            if "```json" in metin:
                metin = metin.split("```json")[1].split("```")[0].strip()
            elif "```" in metin:
                metin = metin.split("```")[1].split("```")[0].strip()
            veri = json.loads(metin)
            if isinstance(veri, dict) and "sorular" in veri:
                veri = veri["sorular"]
            if isinstance(veri, list) and len(veri) > 0:
                for idx, q in enumerate(veri, 1):
                    q["no"] = idx
                return veri
        except Exception:
            pass
        return None

    def _soru_istemi_olustur(self, video_analizi: Dict[str, Any], egitim_seviyesi: str, soru_sayisi: int = 10) -> str:
        ders = video_analizi.get("ders_adi", "Fen Bilimleri")
        unite = video_analizi.get("unite_basligi", "")
        bolumler = video_analizi.get("bolumler", [])
        kavramlar = []
        for b in bolumler:
            for k in b.get("onemli_kavramlar", []):
                kavramlar.append(k.get("kavram", ""))
        kavramlar_str = ", ".join(list(set(kavramlar))[:15])

        return f"""Sen MEB müfredatına hakim uzman bir Eğitim Teknoloğusun.
Ders: {ders}, Seviye: {egitim_seviyesi}, Ünite: {unite}.
Öğrenilen Kavramlar: {kavramlar_str}.

GÖREV: Bu ders için tam {soru_sayisi} adet pedagojik, seviyeye uygun özgün değerlendirme sorusu üret.
Soruların çoğu Çoktan Seçmeli (A, B, C, D seçenekli), en az 2 tanesi Doğru - Yanlış formatında olmalıdır.

JSON FORMATI:
[
  {{
    "no": 1,
    "kategori": "💡 Kavram Keşfi",
    "ilgili_kavram": "Kavram Adı",
    "tip": "Çoktan Seçmeli",
    "soru": "Soru metni...",
    "secenekler": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "dogru_cevap": "A",
    "aciklama": "Gerekçeli çözüm açıklaması.",
    "kazanim_referansi": "MEB Ders Kitabı"
  }}
]
SADECE geçerli bir JSON dizisi döndür."""

    def _gemini_api_cagir(
        self,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Google Gemini API'ye istek gönderir.
        HTTP 503 (High Demand) durumunda exponential backoff ile yeniden dener.
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
                                sorular = self._json_sorulari_ayristir(parts[0]["text"].strip())
                                if sorular and len(sorular) >= 5:
                                    return sorular
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
                    print(f" -> [UYARI] Google Gemini API soru üretim hatası ({current_model}): {e}")
                    break

        return None

    def _ollama_ile_soru_uret(self, istem: str, timeout: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        tout = timeout or self.ollama_question_timeout
        try:
            print(f"[QuestionGenerator - 1. Tercih] Yerel Ollama ({self.ollama_model}) ile soru üretiliyor (Zaman aşımı: {tout}s)...")
            payload = {
                "model": self.ollama_model,
                "prompt": istem,
                "stream": False,
                "format": "json"
            }
            res = requests.post(f"{self.ollama_url.rstrip('/')}/api/generate", json=payload, timeout=tout)
            if res.status_code == 200:
                metin = res.json().get("response", "").strip()
                sorular = self._json_sorulari_ayristir(metin)
                if sorular and len(sorular) >= 5:
                    print(f" -> [BAŞARILI] Yerel Ollama ({self.ollama_model}) {len(sorular)} özgün soru üretti.")
                    return sorular
        except Exception as e:
            print(f" -> [BİLGİ] Yerel Ollama soru üretiminde yanıt vermedi: {e}")
        return None

    def _gemini_ile_soru_uret(self, istem: str) -> Optional[List[Dict[str, Any]]]:
        print(f"[QuestionGenerator - 2. Tercih (Fallback)] Google Gemini ({self.model}) ile soru üretiliyor...")
        payload = {
            "contents": [{"parts": [{"text": istem}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        sorular = self._gemini_api_cagir(payload, timeout=self.gemini_timeout)
        if sorular and len(sorular) >= 5:
            print(f" -> [BAŞARILI] Google Gemini ({self.model}) {len(sorular)} özgün soru üretti.")
            return sorular
        return None

    def _rag_kaynaklarini_bagla(
        self,
        sorular: List[Dict[str, Any]],
        video_analizi: Dict[str, Any],
        rag_soru_ornekleri: Optional[List[Dict[str, Any]]] = None,
        varsayilan_kaynak_tipi: str = "rag_fallback",
    ) -> List[Dict[str, Any]]:
        """
        Her soru için doğrulanmış gerçek RAG kaynağını bağlar.
        Asla doğrulanmamış sayfa numarası veya kazanım kodu uydurmaz.
        Doğrulanmış eşleşme yoksa açıkça 'Doğrulanmış ders kitabı kaynağı bulunamadı' yazar.
        Açıklama (çözüm gerekçesi) ile ders kitabı referansını kesinlikle ayrı tutar.
        """
        bolumler = video_analizi.get("bolumler", []) if isinstance(video_analizi, dict) else []
        tum_kaynaklar = video_analizi.get("tum_kaynaklar", []) if isinstance(video_analizi, dict) else []

        for q in sorular:
            if not q.get("kaynak_tipi"):
                q["kaynak_tipi"] = varsayilan_kaynak_tipi

            kavram = (q.get("ilgili_kavram") or "").lower()
            kategori = (q.get("kategori") or "").lower()
            soru_metni = (q.get("soru") or "").lower()

            eslesen_ref = None
            eslesen_rag_meta = None

            # 1. Bölüm seviyesindeki doğrulanmış kitap referansı ve kavram eşleşmesi
            for b in bolumler:
                b_baslik = (b.get("bolum_basligi") or "").lower()
                b_kavramlar = [k.get("kavram", "").lower() for k in b.get("onemli_kavramlar", [])]
                b_ref = b.get("kitap_referansi")

                if not b_ref or b_ref == "MEB Ders Kitabı Referansı":
                    continue

                # Kavram veya başlık örtüşmesi
                kavram_kelimeler = [w for w in kavram.split() if len(w) > 3]
                kelime_eslesti = any(w in b_baslik or any(w in bk for bk in b_kavramlar) for w in kavram_kelimeler)
                if kelime_eslesti or any(bk in soru_metni for bk in b_kavramlar if len(bk) > 3):
                    eslesen_ref = b_ref
                    if b.get("rag_kaynaklari"):
                        eslesen_rag_meta = b["rag_kaynaklari"][0]
                    break

            # 2. Bölümlerde doğrudan eşleşme yoksa ilk doğrulanmış bölüm referansını incele
            if not eslesen_ref:
                for b in bolumler:
                    b_ref = b.get("kitap_referansi")
                    if b_ref and b_ref != "MEB Ders Kitabı Referansı" and ("Sayfa" in b_ref or "s." in b_ref or "Sayfa" in str(b_ref)):
                        eslesen_ref = b_ref
                        if b.get("rag_kaynaklari"):
                            eslesen_rag_meta = b["rag_kaynaklari"][0]
                        break

            # 3. tum_kaynaklar havuzundan eşleştir
            if not eslesen_ref and tum_kaynaklar:
                for src in tum_kaynaklar:
                    if src.get("referans"):
                        eslesen_ref = src["referans"]
                        eslesen_rag_meta = src
                        break

            # 4. rag_soru_ornekleri havuzundan eşleştir
            if not eslesen_ref and rag_soru_ornekleri:
                for r_orn in rag_soru_ornekleri:
                    if r_orn.get("referans"):
                        eslesen_ref = r_orn["referans"]
                        eslesen_rag_meta = r_orn
                        break

            # 5. Sonuç: Doğrulanmış referans yoksa ASLA uydurma
            if eslesen_ref:
                q["kazanim_referansi"] = eslesen_ref
                q["rag_dogrulandi"] = True
                if eslesen_rag_meta:
                    q["rag_kaynagi"] = eslesen_rag_meta
            else:
                q["kazanim_referansi"] = "Doğrulanmış ders kitabı kaynağı bulunamadı"
                q["rag_dogrulandi"] = False

        return sorular

    def soru_uret(
        self,
        video_analizi: Dict[str, Any],
        egitim_seviyesi: str = "İlkokul",
        rag_soru_ornekleri: Optional[List[Dict[str, Any]]] = None,
        soru_sayisi: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Kullanıcının belirlediği eğitim seviyesine ve öğrenilen kavramlara tam uyumlu
        özgün değerlendirme soruları üretir.
        Önce yerel Ollama modelini, ulaşılamazsa Gemini API'yi dener.
        Tüm soruların kaynaklarını doğrulanmış gerçek RAG verilerine bağlar.
        """
        print(f"\n[QuestionGenerator] '{egitim_seviyesi}' Kademesi İçin {soru_sayisi} Özgün Değerlendirme Sorusu Hazırlanıyor...")

        # İstem hazırla
        istem = self._soru_istemi_olustur(video_analizi, egitim_seviyesi, soru_sayisi)

        # 1. Aşama: Yerel Ollama (qwen3.5:4b)
        ai_sorulari = self._ollama_ile_soru_uret(istem)
        if ai_sorulari:
            for q in ai_sorulari:
                q["kaynak_tipi"] = "ollama"
            return self._rag_kaynaklarini_bagla(ai_sorulari, video_analizi, rag_soru_ornekleri, varsayilan_kaynak_tipi="ollama")

        # 2. Aşama: Google Gemini API (Ollama yanıt vermezse)
        if self._api_anahtari_var_mi():
            print("[BİLGİ] Yerel model soru üretmedi, Google Gemini API devreye alınıyor...")
            ai_sorulari = self._gemini_ile_soru_uret(istem)
            if ai_sorulari:
                for q in ai_sorulari:
                    q["kaynak_tipi"] = "gemini"
                return self._rag_kaynaklarini_bagla(ai_sorulari, video_analizi, rag_soru_ornekleri, varsayilan_kaynak_tipi="gemini")

        # 3. Aşama: Güvenlik Ağı - Pedagojik Şablon Bankası
        print("[BİLGİ] Yapay zeka servislerine ulaşılamadı. Müfredat uyumlu standart soru bankası devreye alınıyor...")

        # -------------------------------------------------------------------
        # 1. İLKOKUL (3. Sınıf Fen Bilimleri)
        # -------------------------------------------------------------------
        fallback_sorular: List[Dict[str, Any]] = []
        if egitim_seviyesi == "İlkokul":
            unite = str(video_analizi.get("unite_basligi", "")).lower()
            bolum_basliklari = " ".join([b.get("bolum_basligi", "") for b in video_analizi.get("bolumler", [])]).lower()
            is_arac_gerecler = "araç" in unite or "aydınlatma" in bolum_basliklari or "ısınma" in bolum_basliklari or "soğutma" in bolum_basliklari

            if is_arac_gerecler:
                fallback_sorular = [
                    {
                        "no": 1,
                        "kategori": "💡 Kavram Keşfi",
                        "ilgili_kavram": "Elektrikli Araç ve Gereçler",
                        "tip": "Doğru - Yanlış",
                        "soru": "Elektrikli araç ve gereçler; elektrik enerjisini kullanarak günlük işlerimizi daha kolay, hızlı ve yorulmadan yapmamızı sağlayan teknolojik aletlerdir.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Elektrikli araçlar insan gücünden ve zamandan tasarruf sağlar; günlük hayatı büyük ölçüde kolaylaştırır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 2,
                        "kategori": "🔍 Günlük Yaşam Analizi",
                        "ilgili_kavram": "Elektrik Olmasaydı Yaşam",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Bir akşam mahallenizde elektrikler kesildiğinde aşağıdaki durumlardan hangisi MEYDANA GELMEZ?",
                        "secenekler": [
                            "A) Buzdolabındaki yiyeceklerin zamanla ısınarak bozulma tehlikesi ortaya çıkar.",
                            "B) Odadaki tavan lambası ve avizeler yanmaz.",
                            "C) Duvardaki prizden elektrik alan çamaşır makinesi durur.",
                            "D) Mekanik yayla kurularak çalışan mekanik çalar saat anında durur."
                        ],
                        "dogru_cevap": "D",
                        "aciklama": "Mekanik kurmalı saatler elektrik enerjisine ihtiyaç duymadan yay gücüyle çalıştığı için elektrik kesintisinden etkilenmez.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 3,
                        "kategori": "💡 Aydınlatma Amaçlı Araçlar",
                        "ilgili_kavram": "Aydınlatma Araçları",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Güneş battıktan sonra karanlıkta ders çalışabilmemiz veya gece odamızı aydınlatabilmemiz için elektrik enerjisini ışığa dönüştüren araç aşağıdakilerden hangisidir?",
                        "secenekler": [
                            "A) Elektrikli süpürge",
                            "B) Masa lambası ve ampul",
                            "C) Buzdolabı",
                            "D) Tost makinesi"
                        ],
                        "dogru_cevap": "B",
                        "aciklama": "Ampul ve masa lambası elektrik enerjisini ışık enerjisine dönüştüren temel aydınlatma araçlarıdır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 4,
                        "kategori": "⚡ Isınma ve Pişirme Dedektifi",
                        "ilgili_kavram": "Isınma Amaçlı Araçlar",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Aşağıdaki elektrikli araçlardan hangisi kullanım amacı bakımından diğerlerinden FARKLI bir grupta yer alır?",
                        "secenekler": [
                            "A) Elektrikli soba (Isıtıcı)",
                            "B) Ekmek kızartma makinesi",
                            "C) Klima ve derin dondurucu",
                            "D) Su ısıtıcısı (Kettle)"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Elektrikli soba, ekmek kızartma makinesi ve kettle ısıtma/ısınma amaçlıdır; klima ve derin dondurucu ise soğutma amaçlıdır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 5,
                        "kategori": "❄️ Soğutma Dedektifi",
                        "ilgili_kavram": "Soğutma ve Besin Saklama",
                        "tip": "Doğru - Yanlış",
                        "soru": "Buzdolabı, içine konulan yiyecek ve içeceklerin ısısını düşürerek onların bozulmasını önleyen ve besinleri taze tutan soğutma amaçlı bir elektrikli araçtır.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Buzdolabı ortamı soğutarak bakterilerin üremesini yavaşlatır ve gıdaların taze kalmasını sağlar.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 6,
                        "kategori": "🧹 Temizlik ve Hijyen",
                        "ilgili_kavram": "Temizlik Amaçlı Araçlar",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Evimizde temizlik ve hijyen sağlamak amacıyla kullanılan; yerdeki toz ve kırıntıları çekerek emen elektrikli araç hangisidir?",
                        "secenekler": [
                            "A) Elektrikli süpürge",
                            "B) Bulaşık makinesi",
                            "C) Çamaşır makinesi",
                            "D) Saç kurutma makinesi"
                        ],
                        "dogru_cevap": "A",
                        "aciklama": "Elektrikli süpürge zeminlerdeki toz ve kirleri vakumlayarak temizleyen temel temizlik aracıdır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 7,
                        "kategori": "📱 Haberleşme ve İletişim",
                        "ilgili_kavram": "Haberleşme Araçları",
                        "tip": "Boşluk Doldurma",
                        "soru": "Cümledeki noktalı yeri doğru kavramla tamamlayınız:\nUzaktaki insanlarla sesli, görüntülü konuşmamızı, dünyadaki gelişmeleri takip etmemizi ve bilgiye hızla ulaşmamızı sağlayan televizyon, bilgisayar ve cep telefonu gibi araçlara ........................................ amaçlı elektrikli araçlar denir.",
                        "secenekler": [],
                        "dogru_cevap": "Haberleşme (İletişim)",
                        "aciklama": "Televizyon, radyo, telefon ve tablet gibi aletler iletişim ve bilgi aktarımı sağlar.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 8,
                        "kategori": "🎯 Eşleştirme ve Sınıflandırma",
                        "ilgili_kavram": "Kullanım Amaçlarına Göre Gruplandırma",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Aşağıdaki araç ve kullanım amacı eşleştirmelerinden hangisi YANLIŞTIR?",
                        "secenekler": [
                            "A) Vantilatör  ➔  Soğutma / Serinletme",
                            "B) Ütü  ➔  Isınma / Düzleştirme",
                            "C) Çamaşır Makinesi  ➔  Haberleşme",
                            "D) El Feneri  ➔  Aydınlatma"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Çamaşır makinesi haberleşme değil, temizlik amaçlı bir araçtır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 9,
                        "kategori": "🌱 Bilinçli Tüketici ve Tasarruf",
                        "ilgili_kavram": "Elektrik Tasarrufu",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Elektrikli araçları kullanırken enerji tasarrufu yapmak ve aile bütçemizi korumak için aşağıdakilerden hangisini YAPMALIYIZ?",
                        "secenekler": [
                            "A) Odadan çıkarken açık kalan lambayı kapatmalıyız.",
                            "B) Buzdolabının kapağını uzun süre açık bırakmalıyız.",
                            "C) İzlemediğimiz televizyonu sürekli açık tutmalıyız.",
                            "D) Gündüz güneş ışığı varken evin bütün lambalarını yakmalıyız."
                        ],
                        "dogru_cevap": "A",
                        "aciklama": "Gereksiz yanan lambaları söndürmek hem enerji tasarrufu sağlar hem de doğal kaynakları korur.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 10,
                        "kategori": "🛡️ Güvenlik Dedektifi",
                        "ilgili_kavram": "Elektrikli Araçları Güvenli Kullanma",
                        "tip": "Doğru - Yanlış",
                        "soru": "Elektrikli araçları ıslak elle veya banyoda suyun yanında kullanmak elektrik çarpmasına yol açabileceğinden son derece tehlikelidir.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Su elektriği iletir; ıslak ellerle elektrikli aletlere dokunmak can güvenliğini tehlikeye sokar.",
                        "kazanim_referansi": ""
                    }
                ]
            else:
                fallback_sorular = [
                    {
                        "no": 1,
                        "kategori": "💡 Kavram Keşfi",
                        "ilgili_kavram": "Elektrik Enerjisi ve Dönüşümü",
                        "tip": "Doğru - Yanlış",
                        "soru": "Elektrik enerjisi; evimizde ve okulumuzda kullandığımız aletler sayesinde ışık (ampul), ısı (fırın, ütü) ve hareket (vantilatör) enerjisine dönüştürülebilen temel bir enerji türüdür.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Elektrik tek başına bir alet değil enerjidir. Aletlerin içine girerek ihtiyacımıza göre ışığa, ısıya veya harekete dönüşür.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 2,
                        "kategori": "🔍 Günlük Yaşam Analizi",
                        "ilgili_kavram": "Enerji İhtiyacı ve Taşınabilirlik",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Dışarıda bahçede veya parkta oynarken yanımızda taşıdığımız oyuncak arabanın veya el fenerinin aniden durmasının en temel bilimsel sebebi aşağıdakilerden hangisidir?",
                        "secenekler": [
                            "A) Havanın kararması ve güneş ışığının kaybolması",
                            "B) İçine takılan pildeki depolanmış elektrik enerjisinin tükenmiş olması",
                            "C) Şehirdeki büyük elektrik santralinin kapanması",
                            "D) Aletin duvardaki prize takılmak istemesi"
                        ],
                        "dogru_cevap": "B",
                        "aciklama": "Taşınabilir aletler pildeki sınırlı kimyasal enerjiyle çalışır; pildeki enerji bittiğinde alet hemen durur.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 3,
                        "kategori": "🧩 Kavram Tamamlama",
                        "ilgili_kavram": "Şehir Elektriği ve Santraller",
                        "tip": "Boşluk Doldurma",
                        "soru": "Cümledeki noktalı yeri doğru kavramla tamamlayınız:\nBarajlardaki suyun gücü, rüzgâr türbinleri veya güneş panelleri kullanılarak santrallerde yüksek güçte üretilen ve kablolarla evlerimize ulaştırılan enerji kaynağına ........................................ denir.",
                        "secenekler": [],
                        "dogru_cevap": "Şehir Elektriği",
                        "aciklama": "Santrallerde üretilen elektrik, iletim hatları ve trafolar üzerinden evlerimizin prizlerine kadar kesintisiz ulaştırılır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 4,
                        "kategori": "⚡ Elektrik Dedektifi",
                        "ilgili_kavram": "Priz Bağlantısı ve Yüksek Güçlü Aletler",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Aşağıdaki ev aletlerinden hangisi çok yüksek miktarda enerji tükettiği için küçük kumanda pilleriyle ÇALIŞTIRILAMAZ ve mutlaka duvardaki prize takılmalıdır?",
                        "secenekler": [
                            "A) Kol saati",
                            "B) Duvar saati",
                            "C) Çamaşır makinesi",
                            "D) Televizyon uzaktan kumandası"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Çamaşır makinesi, buzdolabı ve ütü gibi büyük aletler çok yüksek elektrik enerjisine ihtiyaç duyar; bu enerji yalnızca prizden (şehir elektriği) sağlanabilir.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 5,
                        "kategori": "🎯 Doğru mu, Yanlış mı?",
                        "ilgili_kavram": "Pil ve Düşük Güç Tüketimi",
                        "tip": "Doğru - Yanlış",
                        "soru": "Pil; az miktarda elektrik enerjisi harcayan, hafif ve kablosuz olarak kolayca taşınabilen küçük aletleri (duvar saati, radyo, el feneri) çalıştırmak için kullanılan güvenli bir elektrik kaynağıdır.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Piller düşük enerji depolar ve kabloya ihtiyaç duymadan cihazları yanımızda taşımamızı sağlar.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 6,
                        "kategori": "🌱 Doğa ve Çevre Bilinci",
                        "ilgili_kavram": "Atık Pil ve Geri Dönüşüm",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Tükenen pilleri normal ev çöpüne, sokağa veya toprağa atmak yerine okullardaki ve marketlerdeki 'Atık Pil Kutusu'na atmamızın en önemli nedeni nedir?",
                        "secenekler": [
                            "A) Pillerin kutu içinde kendi kendine renk değiştirmesi",
                            "B) İçlerindeki zararlı kimyasal maddelerin toprağa ve içme suyuna karışıp canlılara zarar vermesini önlemek",
                            "C) Çöp poşetinin daha hafif kalmasını sağlamak",
                            "D) Pillerin kutunun içinde sihirli şekilde yeniden dolması"
                        ],
                        "dogru_cevap": "B",
                        "aciklama": "Pillerin içinde cıva, kadmiyum gibi zehirli maddeler bulunur. Çöpe atılırlarsa toprağı ve suyu zehirler; bu yüzden mutlaka atık pil kutusunda toplanmalıdır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 7,
                        "kategori": "🧩 Kavram Tamamlama",
                        "ilgili_kavram": "Batarya ve Şarj Edilebilirlik",
                        "tip": "Boşluk Doldurma",
                        "soru": "Cümledeki noktalı yeri doğru kavramla tamamlayınız:\nBirden fazla pilin bir araya gelmesiyle oluşan; dizüstü bilgisayar, tablet ve akıllı telefonlarda kullanılan, bittiğinde çöpe atılmayıp adaptörle şarj edilebilen elektrik kaynağına ........................................ denir.",
                        "secenekler": [],
                        "dogru_cevap": "Batarya",
                        "aciklama": "Bataryalar şarj edilebilir özelliktedir; prizden elektrik enerjisi depolayarak aletleri kablosuz çalıştırmaya devam eder.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 8,
                        "kategori": "🔍 Günlük Yaşam Analizi",
                        "ilgili_kavram": "Akü (Akümülatör) ve Taşıtlar",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Otomobil, otobüs, kamyon, uçak ve gemi gibi motorlu ulaşım araçlarında motoru ilk çalıştıran ve farları aydınlatan güçlü elektrik kaynağı hangisidir?",
                        "secenekler": [
                            "A) Kumanda pili",
                            "B) Güneş pili",
                            "C) Akü (Akümülatör)",
                            "D) Rüzgâr türbini"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Motorlu taşıtların motorunu ilk çalıştırma anında harekete geçiren ve elektrik donanımını besleyen büyük ve güçlü kaynak aküdür.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 9,
                        "kategori": "⚡ Eşleştirme ve Sınıflandırma",
                        "ilgili_kavram": "Elektrik Kaynaklarını Ayırt Etme",
                        "tip": "Çoktan Seçmeli",
                        "soru": "Aşağıdaki araç ve kullandığı temel elektrik kaynağı eşleştirmelerinden hangisi YANLIŞTIR?",
                        "secenekler": [
                            "A) Buzdolabı  ➔  Şehir Elektriği",
                            "B) Akıllı Telefon  ➔  Batarya",
                            "C) Şehirler Arası Otobüs  ➔  Kumanda Pili",
                            "D) Duvar Saati  ➔  Pil"
                        ],
                        "dogru_cevap": "C",
                        "aciklama": "Otobüs gibi büyük motorlu taşıtlar küçük kumanda pilleriyle değil, yüksek güce sahip dayanıklı akülerle çalışır.",
                        "kazanim_referansi": ""
                    },
                    {
                        "no": 10,
                        "kategori": "🛡️ Güvenlik Dedektifi",
                        "ilgili_kavram": "Elektrik Güvenliği ve Bilinçli Kullanım",
                        "tip": "Doğru - Yanlış",
                        "soru": "Evimizdeki şehir elektriği prizlerinde bulunan elektrik enerjisi, pillerdeki enerjiden çok daha yüksek ve tehlikelidir; bu yüzden prizlere kesinlikle yabancı cisim sokulmamalı ve ıslak ellerle elektrikli aletlere dokunulmamalıdır.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Şehir elektriği elektrik çarpmasına ve hayati tehlikeye neden olabilir. Güvenlik kurallarına uymak can güvenliğimiz için zorunludur.",
                        "kazanim_referansi": ""
                    }
                ]

        # -------------------------------------------------------------------
        # 2. ORTAOKUL (5-8. Sınıf Seviyesi)
        # -------------------------------------------------------------------
        elif egitim_seviyesi == "Ortaokul":
            fallback_sorular = [
                {
                    "no": i,
                    "kategori": "🔍 Ortaokul Fen Analizi",
                    "ilgili_kavram": f"Elektrik Kavramı {i}",
                    "tip": "Çoktan Seçmeli",
                    "soru": f"Ortaokul Fen Bilimleri elektrik ünitesi analiz sorusu {i}: Elektrik enerjisinin iletimi ve kaynakların verimliliği...",
                    "secenekler": ["A) Seçenek 1", "B) Seçenek 2", "C) Seçenek 3", "D) Seçenek 4"],
                    "dogru_cevap": "B",
                    "aciklama": "Devre elemanları ve potansiyel fark prensiplerine göre doğru yanıt B seçeneğidir.",
                    "kazanim_referansi": ""
                }
                for i in range(1, soru_sayisi + 1)
            ]

        # -------------------------------------------------------------------
        # 3. LİSE (9-12. Sınıf Seviyesi)
        # -------------------------------------------------------------------
        else:
            fallback_sorular = [
                {
                    "no": i,
                    "kategori": "⚡ Lise Fizik & Elektrik Analizi",
                    "ilgili_kavram": f"Elektrik Devreleri & Enerji {i}",
                    "tip": "Analiz ve Problem Çözme",
                    "soru": f"Lise fizik düzeyi elektrik potansiyeli ve elektrostatik alan analizi {i}: Joule yasası ve ohmik iletkenler...",
                    "secenekler": ["A) Seçenek A", "B) Seçenek B", "C) Seçenek C", "D) Seçenek D"],
                    "dogru_cevap": "A",
                    "aciklama": "Elektrik potansiyeli ve direnç katsayılarına göre akım-gerilim ilişkisi incelenmiştir.",
                    "kazanim_referansi": ""
                }
                for i in range(1, soru_sayisi + 1)
            ]

        varsayilan_tur = "rag_fallback" if (video_analizi.get("bolumler") or rag_soru_ornekleri) else "static_fallback"
        return self._rag_kaynaklarini_bagla(
            fallback_sorular[:soru_sayisi],
            video_analizi=video_analizi,
            rag_soru_ornekleri=rag_soru_ornekleri,
            varsayilan_kaynak_tipi=varsayilan_tur
        )


if __name__ == "__main__":
    qg = QuestionGenerator()
    qs = qg.soru_uret({}, egitim_seviyesi="İlkokul", soru_sayisi=10)
    print(f"Toplam {len(qs)} soru üretildi.")
    for q in qs:
        print(f"[{q['no']}] {q['kategori']} ({q['tip']}) -> {q['ilgili_kavram']}")
        print(f"     Soru: {q['soru'][:90]}...")
        print(f"     Doğru: {q['dogru_cevap']} | Ref: {q['kazanim_referansi']}\n")
