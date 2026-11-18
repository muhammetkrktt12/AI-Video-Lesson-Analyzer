# EduFrame AI — RAG Sınıf İzolasyonu, Anlamsal Uygunluk ve Kaynak Doğrulama Raporu

**Tarih:** 19 Kasım 2026  
**Dal:** `refactor/project-cleanup`  
**Veritabanı:** `data/chroma_db_v2` (Koleksiyon: `eduframe_meb_768`, 765 vektör)  
**Doğrulama Durumu:** Başarılı (105 / 105 Birim & Entegrasyon Testi Başarılı)

---

## 1. Yönetici Özeti (Executive Summary)

EduFrame AI projesinde RAG (Retrieval-Augmented Generation) hattında tespit edilen kritik sınıf sızıntısı (cross-grade bleed), anlamsal ilgisizlik, sahte skorlama (`vektor_skoru: 1.0`) ve kör soru–kaynak ataması sorunları giderilmiştir.

Sistem mimarisi korunarak şu temel iyileştirmeler gerçekleştirilmiştir:
1. **Sıkı Sınıf İzolasyonu (Strict Grade Isolation):** `3. Sınıf` sorgularının `4. Sınıf` içeriğine sızması en erken aşamada (`where={"sinif": "3. Sınıf"}`) engellendi. `allow_cross_grade=False` varsayılan kılındı.
2. **Kavramsal ve Ünite Düzeyinde Uygunluk (Topic & Unit Relevance):** Sadece "oyuncak" veya "pil" gibi yüzeysel anahtar kelimelere dayanarak 4. Sınıf Besinlerimiz veya Kuvvetin Etkileri ünitelerinden chunk çekilmesi engellendi. Kaynaklar `ana_kaynak`, `yardimci_kaynak` ve `konu_disi` rollerine ayrıldı.
3. **Dürüst Skorlama ve Destek Durumu (Genuine Scoring):** Sahte `vektor_skoru: 1.0` kaldırıldı. Çevrimdışı/fallback durumunda skorlar dürüstçe `None` olarak işaretlendi. `tam_destekli` statüsü ancak hedef sınıf ve üniteyle tam uyuşan doğrulanmış bir `ana_kaynak` bulunduğunda verilecek şekilde sıkılaştırıldı.
4. **Soru–Kaynak Bağlama Doğrulaması (Question Citation Overhaul):** Soru üreticinin (`_rag_kaynaklarini_bagla`) eşleşmeyen sorulara ilk bölümü veya genel referansı körlemesine ataması kaldırıldı. Eşleşmeyen sorular dürüstçe `kaynak_dogrulanamadi` ve `kazanim_referansi = "Doğrulanmış ders kitabı kaynağı bulunamadı"` olarak etiketlendi.
5. **PDF ve Kaynakça Entegrasyonu:** Kaynakça derleyicisi (`_dinamik_kaynakca_derle`) `[Ana Kaynak]` ve `[Yardımcı Kaynak]` ayrımını görselleştirdi, fiziksel ve basılı sayfa ayrımını korudu.

---

## 2. ChromaDB v2 Metadata Şeması ve Koleksiyon Doğrulaması

`data/chroma_db_v2` veritabanı doğrudan incelenmiş ve değişmezliği korunmuştur.

- **Koleksiyon Adı:** `eduframe_meb_768`
- **Toplam Vektör Sayısı:** 765
- **Sınıf Dağılımı:**
  - `3. Sınıf`: 391 chunk (MEB 3. Sınıf Fen Bilimleri 1. ve 2. Kitap)
  - `4. Sınıf`: 374 chunk (MEB 4. Sınıf Fen Bilimleri)
- **Kritik Metadata Alanları:**
  - `sinif`: `'3. Sınıf'` / `'4. Sınıf'`
  - `unite`: Örn. `'6. Ünite: Yaşamımızı Kolaylaştıran Elektrik'`, `'4. Ünite: Maddenin Özellikleri'`, `'3. Ünite: Kuvvetin Etkileri'`
  - `konu`: Örn. `'Elektrikli Araç Gereç'`, `'Elektrikli Araçların Güvenli Kullanımı'`
  - `alt_konu`: Örn. `'Elektrikli Araçların Kullanım Alanları'`
  - `physical_page`: PDF fiziksel sayfa indeksi (1-tabanlı)
  - `printed_page`: Kitap üzerindeki gerçek sayfa numarası
  - `chunk_id`: Tekil tanımlayıcı (örn. `ilkokul_fenbilimleri_3.sinif_2.kitap.pdf:421c34396672:v2:phys_0049:c000`)
  - `referans`: Standartlaştırılmış MEB kitap atıfı

---

## 3. Sınıf İzolasyonu Mimarisi

Önceki mimaride vektör araması sonuçsuz kaldığında veya geniş filtre uygulandığında `{"egitim_seviyesi": "İlkokul"}` filtresi kullanılıyor, bu da 3. Sınıf analizine 4. Sınıf ünitelerinin sızmasına yol açıyordu.

### Yapılan Düzenlemeler:
1. **`RAGMotoru.sinif_normalize(sinif)` Eklendi:**
   - `"3"`, `"3.sınıf"`, `"3. Sınıf"`, `" 3. SINIF "` girdileri kararlı biçimde `"3. Sınıf"` formatına dönüştürülür.
   - `"4"` türevleri `"4. Sınıf"` formatına dönüştürülür.
2. **`seviye_filtreli_arama()` Güncellendi:**
   - `allow_cross_grade: bool = False` parametresi eklendi.
   - Vektör aramasında doğrudan `where={"sinif": norm_sinif}` uygulandı.
   - Fallback (lexical) aramasında da aynı `{"sinif": norm_sinif}` filtresi korunarak diğer sınıfların chunk'larının listeye girmesi engellendi.
   - Post-filter aşamasında `allow_cross_grade=False` ise hedef sınıf dışındaki tüm dokümanlar elenir.

---

## 4. Konu, Ünite ve Pedagojik Uygunluk Analizi

Yalnızca "oyuncak" veya "pil" gibi ortak kelimeler üzerinden alakasız ünitelerden chunk seçilmesini engellemek için `chunk_alaka_puanla` fonksiyonuna semantik ve pedagojik filtreler eklendi:

### Rol Dağılımı ve Skorlama Kriterleri:
- **`ana_kaynak`:**
  - Hedef sınıfla (`3. Sınıf`) birebir uyuşan,
  - İlgili ünite (`"Yaşamımızı Kolaylaştıran Elektrik"`, `"Elektrikli Araçlar"` vb.) altında yer alan,
  - Konu anlatımının doğrudan temelini oluşturan chunk'lar (örn. 3. Sınıf 2. Kitap Sayfa 48).
- **`yardimci_kaynak`:**
  - Hedef sınıfta yer alan ancak güvenlik, yan kavram veya destekleyici bağlam sunan chunk'lar (örn. 3. Sınıf 2. Kitap Sayfa 57 - Elektrikli Araçların Güvenli Kullanımı).
- **`konu_disi` (Elenenler):**
  - Hedef sınıf uyuşmazlığı (`4. Sınıf` chunk'ları),
  - Konu dışı ünite cezası: `"Besinlerimiz"`, `"Kuvvetin Etkileri"`, `"Maddenin Özellikleri"` ve `"Dünya'mızın Hareketleri"` üniteleri için -10.0 puan ceza verilerek anında elenir.

---

## 5. Dürüst Skorlama ve Kaynak Destek Durumu

Önceki kodda fallback aramalarında atanan sahte `vektor_skoru: 1.0` değeri tamamen kaldırıldı.

- **Vektör Araması Başarılı Olduğunda:**
  - `vektor_mesafesi`: ChromaDB'den dönen gerçek L2 mesafesi.
  - `vektor_skoru`: \(1.0 / (1.0 + \text{mesafe})\) formülü ile hesaplanan gerçek benzerlik skoru.
  - `arama_yontemi`: `"vektor"`
- **Fallback (Çevrimdışı / Sandbox) Olduğunda:**
  - `vektor_skoru`: `None`
  - `vektor_mesafesi`: `None`
  - `arama_yontemi`: `"lexical_fallback"`
  - `lexical_skor`: Gerçek 3-gram ve terim örtüşme skoru.
- **Kaynak Destek Durumu:**
  - `tam_destekli`: Ancak ve ancak hedef sınıfa ait en az bir `ana_kaynak` bulunduğunda verilir.
  - `kismi_destekli`: Sadece `yardimci_kaynak` bulunduğunda verilir.
  - `desteksiz`: Hiçbir geçerli kaynak bulunamadığında verilir.

---

## 6. Soru–Kaynak Bağlama Hattı İyileştirmesi

`services/question_generator.py` dosyasındaki `_rag_kaynaklarini_bagla()` fonksiyonu yeniden yapılandırıldı:
- **Eski Durum:** Soru içeriği hiçbir RAG kaynağıyla eşleşmese dahi ilk bölümün kitap referansı (`bolumler[0]["kitap_referansi"]`) körlemesine soruya atanıyordu.
- **Yeni Durum:**
  - Soru metni, şıklar ve kavramlar doğrulanmış aday chunk'ların metinleriyle n-gram leksikal benzerlik ve kavram örtüşmesiyle karşılaştırılır.
  - Eşik değer (3.0 puan) aşılamazsa yapay atama yapılmaz.
  - Soruya açıkça şu alanlar yazılır:
    - `rag_dogrulandi`: `False`
    - `kaynak_durumu`: `"kaynak_dogrulanamadi"`
    - `kazanim_referansi`: `"Doğrulanmış ders kitabı kaynağı bulunamadı"`
    - `eslesen_chunk_id`: `None`

---

## 7. Bölüm 2 Özelinde Eski ve Yeni RAG Karşılaştırma Tablosu

**Bölüm 2 Konusu:** Günlük Yaşamda Elektrik ve Oyuncak Araba Örneği (3. Sınıf Fen Bilimleri)

| Kriter | Eski RAG Davranışı | Yeni RAG Davranışı |
| :--- | :--- | :--- |
| **Sınıf Sızıntısı** | 4. Sınıf p.76 (Kuvvetin Etkileri) veya 4. Sınıf p.46 (Besinlerimiz) getiriliyordu | Sıfır sızıntı. Yalnızca 3. Sınıf değerlendirildi |
| **Seçilen Ana Kaynak** | Yanlış ünite veya "Doğrulanmış RAG eşleşmesi bulunamadı" | **3. Sınıf 2. Kitap Sayfa 48** (Elektrikli Araç Gereç / Pil) |
| **Seçilen Yardımcı Kaynak** | Yok veya konu dışı | **3. Sınıf 2. Kitap Sayfa 57** (Elektrikli Araçların Güvenli Kullanımı) |
| **Elenen Kaynaklar** | Elenenler raporlanmıyordu | 4 chunk şeffafça elendi (sayfa 69 ses, sayfa 71 ışık kirliliği vb.) |
| **Vektör Skoru** | Sahte `1.0` | Fallback modunda dürüstçe `None`, leksikal skor: `0.487` |
| **Destek Durumu** | Sahte tam destek veya genel başarısızlık | Dürüst `tam_destekli` (Doğrulanmış `ana_kaynak` mevcut) |

---

## 8. Test Sonuçları ve Doğrulama Matrisi

`tests/test_rag_grade_isolation_and_quality.py` dosyasında oluşturulan 7 yeni senaryo testi ve mevcut test paketinin tamamı çalıştırılmıştır.

| Test | Senaryo | Hedef | Durum |
| :--- | :--- | :--- | :--- |
| **Test A** | `test_scenario_a_sinif_normalization` | "3", "3.sınıf", "4. Sınıf" formatlarının doğru normalizasyonu | **BAŞARILI** |
| **Test B** | `test_scenario_b_cross_grade_isolation_in_search` | 3. Sınıf aramasında 4. Sınıf chunk'larının elenmesi | **BAŞARILI** |
| **Test C** | `test_scenario_c_off_topic_unit_penalization` | 4. Sınıf Besinlerimiz ve Kuvvet ünitelerinin ceza alıp elenmesi | **BAŞARILI** |
| **Test D** | `test_scenario_d_honest_vector_scoring` | Fallback modunda vektör skorunun `None` olması | **BAŞARILI** |
| **Test E** | `test_scenario_e_no_silent_cross_grade_bleed` | Sadece 4. Sınıfa özgü konuda 3. Sınıf araması yapıldığında sızıntı olmaması | **BAŞARILI** |
| **Test F** | `test_scenario_f_question_binding_verified_match` | Uyumlu sorunun doğru chunk_id ve sayfaya bağlanması | **BAŞARILI** |
| **Test G** | `test_scenario_g_question_binding_unverified_marked` | Alakasız sorunun sahte referans almayıp dürüstçe işaretlenmesi | **BAŞARILI** |

### Genel Test Koşumu Çıktısı:
```text
Ran 105 tests in 25.899s
OK
```
Tüm 105 birim ve entegrasyon testi gerileme (regression) olmadan başarıyla tamamlanmıştır.

---

## 9. Kullanıcı Doğrulama Komutu (PowerShell)

Kullanıcı yerel terminalinde sistemi test etmek için aşağıdaki komutları çalıştırabilir:

### A) Yalnızca RAG Sınıf İzolasyonu Testlerini Çalıştırmak İçin:
```powershell
.venv\Scripts\python.exe -m unittest tests/test_rag_grade_isolation_and_quality.py -v
```

### B) Tüm Test Paketini Çalıştırmak İçin:
```powershell
.venv\Scripts\python.exe -m unittest discover tests -v
```

### C) Ollama `embeddinggemma` Çevrimiçi İken Bölüm 2 Analizini Canlı Doğrulamak İçin:
```powershell
.venv\Scripts\python.exe -c "
from services.rag_motoru import RAGMotoru
rag = RAGMotoru(chroma_yolu='data/chroma_db_v2')
res = rag.bolum_bazli_cok_yonlu_arama(
    bolum_baslik='Bölüm 2 – Günlük Yaşamda Elektrik ve Oyuncak Araba Örneği',
    bolum_icerik='Öğretmen oyuncak arabanın pille çalıştığını anlatıyor. Elektrik enerjisi pilden sağlanır.',
    ders='Fen Bilimleri',
    sinif='3. Sınıf',
    gorsel_kavramlar=['oyuncak araba', 'pil', 'elektrik devresi']
)
print('Durum:', res['kaynak_destek_durumu'])
print('Ana Kaynak Sayısı:', len(res['ana_kaynaklar']))
for ak in res['ana_kaynaklar']:
    print(f' - {ak[\"referans\"]} (Skor: {ak[\"puan\"]}, Yontem: {ak[\"arama_yontemi\"]})')
"
```

---

## 10. Teknik Riskler ve Mimari Sınırlar

1. **Embedding Boyutu ve Model Uyum Sınırı:**
   - ChromaDB v2 koleksiyonu (`eduframe_meb_768`), 768 boyutlu `embeddinggemma` vektörleriyle indekslenmiştir.
   - Ollama servisi kapalıyken veya sandbox içinde network kısıtındayken vektör araması yerine leksikal fallback devreye girer. Fallback modu tüm metadata filtrelerini (`sinif`, `unite`) tam olarak uygular ancak anlamsal mesafe yerine leksikal örtüşme kullanır.
2. **Koleksiyonun Değişmezliği (Immutability):**
   - `data/chroma_db_v2` dizinine hiçbir yeni kayıt yazılmamış ve eski kayıt silinmemiştir (765 kayıt sabit tutulmuştur).
3. **Soru–Cevap Anahtarı Tutarlılığı:**
   - Kitapta doğrudan eşleşmesi olmayan yapay zeka soruları artık ders kitabına zorla bağlanmamaktadır. Bu durum öğrenci ve öğretmene şeffaf şekilde "Doğrulanmış ders kitabı kaynağı bulunamadı" olarak yansıtılmaktadır.
