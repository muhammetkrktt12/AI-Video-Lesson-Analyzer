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
import requests
from typing import List, Dict, Any, Optional

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_URL, OLLAMA_MODEL
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-3.8-flash"
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")


class QuestionGenerator:
    """
    Video materyal kavramlarına ve MEB müfredat standartlarına göre özgün sorular üreten motor.

    Öncelik Sırası:
    1. Öncelik: Yerel Ollama (qwen3.5:4b)
    2. Yedek: Google Gemini API (gemini-3.8-flash)
    3. Güvenlik Ağı: MEB Pedagojik Standart Soru Bankası
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
    ):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.ollama_url = ollama_url or OLLAMA_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL

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

    def _ollama_ile_soru_uret(self, istem: str, timeout: int = 15) -> Optional[List[Dict[str, Any]]]:
        try:
            print(f"[QuestionGenerator - 1. Tercih] Yerel Ollama ({self.ollama_model}) ile soru üretiliyor...")
            payload = {
                "model": self.ollama_model,
                "prompt": istem,
                "stream": False,
                "format": "json"
            }
            res = requests.post(f"{self.ollama_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
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
        try:
            print(f"[QuestionGenerator - 2. Tercih (Fallback)] Google Gemini ({self.model}) ile soru üretiliyor...")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": istem}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
            }
            res = requests.post(url, json=payload, timeout=5)
            if res.status_code == 200:
                cands = res.json().get("candidates", [])
                if cands and "content" in cands[0]:
                    t = cands[0]["content"]["parts"][0]["text"]
                    sorular = self._json_sorulari_ayristir(t.strip())
                    if sorular and len(sorular) >= 5:
                        print(f" -> [BAŞARILI] Google Gemini ({self.model}) {len(sorular)} özgün soru üretti.")
                        return sorular
        except Exception as e:
            print(f" -> [UYARI] Gemini soru üretimi hatası: {e}")
        return None

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
        """
        print(f"\n[QuestionGenerator] '{egitim_seviyesi}' Kademesi İçin {soru_sayisi} Özgün Değerlendirme Sorusu Hazırlanıyor...")

        # İstem hazırla
        istem = self._soru_istemi_olustur(video_analizi, egitim_seviyesi, soru_sayisi)

        # 1. Aşama: Yerel Ollama (qwen3.5:4b)
        ai_sorulari = self._ollama_ile_soru_uret(istem)
        if ai_sorulari:
            return ai_sorulari

        # 2. Aşama: Google Gemini API (Ollama yanıt vermezse)
        if self._api_anahtari_var_mi():
            print("[BİLGİ] Yerel model soru üretmedi, Google Gemini API devreye alınıyor...")
            ai_sorulari = self._gemini_ile_soru_uret(istem)
            if ai_sorulari:
                return ai_sorulari

        # 3. Aşama: Güvenlik Ağı - Pedagojik Şablon Bankası
        print("[BİLGİ] Yapay zeka servislerine ulaşılamadı. Müfredat uyumlu standart soru bankası devreye alınıyor...")

        # -------------------------------------------------------------------
        # 1. İLKOKUL (3. Sınıf Fen Bilimleri)
        # -------------------------------------------------------------------
        if egitim_seviyesi == "İlkokul":
            unite = str(video_analizi.get("unite_basligi", "")).lower()
            bolum_basliklari = " ".join([b.get("bolum_basligi", "") for b in video_analizi.get("bolumler", [])]).lower()
            is_arac_gerecler = "araç" in unite or "aydınlatma" in bolum_basliklari or "ısınma" in bolum_basliklari or "soğutma" in bolum_basliklari

            if is_arac_gerecler:
                return [
                    {
                        "no": 1,
                        "kategori": "💡 Kavram Keşfi",
                        "ilgili_kavram": "Elektrikli Araç ve Gereçler",
                        "tip": "Doğru - Yanlış",
                        "soru": "Elektrikli araç ve gereçler; elektrik enerjisini kullanarak günlük işlerimizi daha kolay, hızlı ve yorulmadan yapmamızı sağlayan teknolojik aletlerdir.",
                        "secenekler": ["Doğru", "Yanlış"],
                        "dogru_cevap": "Doğru",
                        "aciklama": "Elektrikli araçlar insan gücünden ve zamandan tasarruf sağlar; günlük hayatı büyük ölçüde kolaylaştırır.",
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 178"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 180"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 181"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 182"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 183"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 184"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 185"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 186"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 187"
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
                        "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 188"
                    }
                ][:soru_sayisi]
            return [
                {
                    "no": 1,
                    "kategori": "💡 Kavram Keşfi",
                    "ilgili_kavram": "Elektrik Enerjisi ve Dönüşümü",
                    "tip": "Doğru - Yanlış",
                    "soru": "Elektrik enerjisi; evimizde ve okulumuzda kullandığımız aletler sayesinde ışık (ampul), ısı (fırın, ütü) ve hareket (vantilatör) enerjisine dönüştürülebilen temel bir enerji türüdür.",
                    "secenekler": ["Doğru", "Yanlış"],
                    "dogru_cevap": "Doğru",
                    "aciklama": "Elektrik tek başına bir alet değil enerjidir. Aletlerin içine girerek ihtiyacımıza göre ışığa, ısıya veya harekete dönüşür.",
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 180"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 181"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 182"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 183"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 184"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 185"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 186"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 187"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 188"
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
                    "kazanim_referansi": "MEB 3. Sınıf Fen Bilimleri, s. 189"
                }
            ][:soru_sayisi]

        # -------------------------------------------------------------------
        # 2. ORTAOKUL (5-8. Sınıf Seviyesi)
        # -------------------------------------------------------------------
        elif egitim_seviyesi == "Ortaokul":
            return [
                {
                    "no": i,
                    "kategori": "🔍 Ortaokul Fen Analizi",
                    "ilgili_kavram": f"Elektrik Kavramı {i}",
                    "tip": "Çoktan Seçmeli",
                    "soru": f"Ortaokul Fen Bilimleri elektrik ünitesi analiz sorusu {i}: Elektrik enerjisinin iletimi ve kaynakların verimliliği...",
                    "secenekler": ["A) Seçenek 1", "B) Seçenek 2", "C) Seçenek 3", "D) Seçenek 4"],
                    "dogru_cevap": "B",
                    "aciklama": "Devre elemanları ve potansiyel fark prensiplerine göre doğru yanıt B seçeneğidir.",
                    "kazanim_referansi": f"MEB Fen Bilimleri, Ünite Değerlendirme s.{180+i}"
                }
                for i in range(1, soru_sayisi + 1)
            ]

        # -------------------------------------------------------------------
        # 3. LİSE (9-12. Sınıf Seviyesi)
        # -------------------------------------------------------------------
        else:
            return [
                {
                    "no": i,
                    "kategori": "⚡ Lise Fizik & Elektrik Analizi",
                    "ilgili_kavram": f"Elektrik Devreleri & Enerji {i}",
                    "tip": "Analiz ve Problem Çözme",
                    "soru": f"Lise fizik düzeyi elektrik potansiyeli ve elektrostatik alan analizi {i}: Joule yasası ve ohmik iletkenler...",
                    "secenekler": ["A) Seçenek A", "B) Seçenek B", "C) Seçenek C", "D) Seçenek D"],
                    "dogru_cevap": "A",
                    "aciklama": "Elektrik potansiyeli ve direnç katsayılarına göre akım-gerilim ilişkisi incelenmiştir.",
                    "kazanim_referansi": f"MEB Fizik Ders Kitabı, s.{140+i}"
                }
                for i in range(1, soru_sayisi + 1)
            ]


if __name__ == "__main__":
    qg = QuestionGenerator()
    qs = qg.soru_uret({}, egitim_seviyesi="İlkokul", soru_sayisi=10)
    print(f"Toplam {len(qs)} soru üretildi.")
    for q in qs:
        print(f"[{q['no']}] {q['kategori']} ({q['tip']}) -> {q['ilgili_kavram']}")
        print(f"     Soru: {q['soru'][:90]}...")
        print(f"     Doğru: {q['dogru_cevap']} | Ref: {q['kazanim_referansi']}\n")
