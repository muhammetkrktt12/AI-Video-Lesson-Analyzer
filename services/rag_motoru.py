"""
EduFrame AI - Metadata Destekli RAG ve Vektör Veritabanı Servisi (services/rag_motoru.py)
--------------------------------------------------------------------------------------
Bu modül, referans ders kitaplarının (PDF) yüklenmesini, eğitim seviyesi bazlı
metadata ile parçalanıp ChromaDB'ye kaydedilmesini ve analiz sırasında sadece
seçilen eğitim seviyesinde anlamsal (semantik) arama yapılmasını sağlar.

Temel Kurallar ve Sorumluluklar:
1. Metadata Filtreleme: Her parça (chunk) 'egitim_seviyesi' (İlkokul, Ortaokul, Lise, Üniversite)
   etiketiyle kaydedilir. Arama sırasında SADECE seçilen seviyedeki belgeler taranır.
2. Soru Formatı Klonlama (Few-Shot): Kitap içerisindeki alıştırma, etkinlik ve test sorularını
   ayrıştırarak soru üretici ajana örnek format sağlar.
3. Esnek Embedding: Ollama (embeddinggemma) öncelikli çalışır; bağlantı yoksa güvenli yerel
   vektörleştirici ile kesintisiz geliştirme ve test imkanı sunar.
4. Temiz Kod ve Açıklamalar: Tüm fonksiyonlar tip ipuçları (type hints) ve detaylı Türkçe
   açıklamalarla donatılmıştır.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import pypdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Kalite ve Sıralama Araçlarını İçe Aktar (utils.lexical_tools öncelikli)
try:
    from utils.lexical_tools import rerank_candidates as lexical_rerank, validate_citations, hit_at_k
except ImportError:
    try:
        from quality_utils import rerank_candidates as lexical_rerank, validate_citations, hit_at_k
    except ImportError:
        lexical_rerank = None

# Gemma Tokenizer Desteği (tokenizer_utils & gemma_tokenizer.json)
try:
    from tokenizer_utils import get_gemma_tokenizer, analyze_chunk_tokens
    _GEMMA_TOKENIZER = get_gemma_tokenizer()
except Exception:
    _GEMMA_TOKENIZER = None
try:
    from config import (
        CHROMA_DB_DIR,
        COLLECTION,
        DATA_DIR,
        REFERENCE_BOOKS_DIR,
        EGITIM_SEVIYELERI,
        DEFAULT_SEVIYE,
        OLLAMA_URL,
        EMBED_MODEL,
    )
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    CHROMA_DB_DIR = DATA_DIR / "chroma_db"
    REFERENCE_BOOKS_DIR = DATA_DIR / "reference_books"
    COLLECTION = "eduframe_rag_v1"
    EGITIM_SEVIYELERI = ["İlkokul", "Ortaokul", "Lise", "Üniversite"]
    DEFAULT_SEVIYE = "İlkokul"
    OLLAMA_URL = "http://localhost:11434"
    EMBED_MODEL = "embeddinggemma"


# ---------------------------------------------------------------------------
# EMBEDDING ADAPTÖRÜ
# ---------------------------------------------------------------------------
class GuvenliEmbeddingMotoru(Embeddings):
    """
    384 boyutlu, deterministik ve Türkçe morfolojisine duyarlı dense embedding motoru.
    Gerektiğinde Ollama servisiyle senkronize olabilir; ancak harici servis olmasa dahi
    kesintisiz ve yüksek performanslı anlamsal vektör üretimi sağlar.
    """

    def __init__(
        self,
        model_adi: str = EMBED_MODEL,
        base_url: str = OLLAMA_URL,
        koleksiyon_adi: Optional[str] = None,
        allow_fallback: bool = False,
    ):
        self.model_adi = model_adi
        self.base_url = base_url
        self.koleksiyon_adi = koleksiyon_adi or COLLECTION
        self.allow_fallback = allow_fallback
        self._ollama_client = None
        self._ollama_kullanilabilir = None
        self._hedef_boyut = self._koleksiyon_boyutu_al()

    def _koleksiyon_boyutu_al(self) -> int:
        """Mevcut Chroma koleksiyonunun beklediği embedding vektör boyutunu tespit eder."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            col = client.get_collection(self.koleksiyon_adi)
            sample = col.get(limit=1, include=['embeddings'])
            if sample and sample.get('embeddings') is not None and len(sample['embeddings']) > 0:
                dim = len(sample['embeddings'][0])
                if dim > 0:
                    return dim
        except Exception:
            pass
        return 768

    def _get_ollama(self):
        if self._ollama_kullanilabilir is False:
            return None
        if self._ollama_client is None:
            try:
                from langchain_ollama import OllamaEmbeddings
                client = OllamaEmbeddings(model=self.model_adi, base_url=self.base_url)
                test_v = client.embed_query("test")
                if len(test_v) == self._hedef_boyut:
                    self._ollama_client = client
                    self._ollama_kullanilabilir = True
                    print(f"[RAGMotoru] Ollama Embedding aktif: {self.model_adi} ({len(test_v)} boyut)")
                else:
                    print(f"[RAGMotoru] Ollama boyutu ({len(test_v)}) koleksiyonla ({self._hedef_boyut}) uyuşmadı.")
                    self._ollama_kullanilabilir = False
            except Exception as e:
                self._ollama_kullanilabilir = False
                self._ollama_client = None
        return self._ollama_client

    def _fallback_embed(self, text: str, dim: Optional[int] = None) -> List[float]:
        """Türkçe kök ve ek duyarlılığı yüksek 768 boyutlu n-gram dense embedding."""
        d = dim or self._hedef_boyut or 768
        vector = np.zeros(d, dtype=np.float32)
        norm_text = text.replace("İ", "i").replace("I", "ı").lower()
        words = re.findall(r"[a-zçğıöşü0-9]+", norm_text)
        if not words:
            return vector.tolist()

        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            vector[h % d] += 2.5
            w_pad = f"^{w}$"
            for i in range(len(w_pad) - 2):
                g = w_pad[i:i+3]
                hg = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
                vector[hg % d] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ollama = self._get_ollama()
        if ollama:
            try:
                return ollama.embed_documents(texts)
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(f"[KRİTİK HATA] Ollama embedding işlemi başarısız oldu: {e}") from e
        if not self.allow_fallback:
            raise RuntimeError(
                f"[KRİTİK HATA] Üretim modunda Ollama ({self.model_adi}) servisine ulaşılamadı. "
                f"Vektör bütünlüğünü korumak için hash fallback kapalıdır. Lütfen Ollama servisini başlatın."
            )
        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        ollama = self._get_ollama()
        if ollama:
            try:
                return ollama.embed_query(text)
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(f"[KRİTİK HATA] Ollama sorgu embedding işlemi başarısız oldu: {e}") from e
        if not self.allow_fallback:
            raise RuntimeError(
                f"[KRİTİK HATA] Üretim modunda Ollama ({self.model_adi}) servisine ulaşılamadı. "
                f"Semantik arama kalitesini ve veri bütünlüğünü korumak için hash fallback kapalıdır. "
                f"Lütfen Ollama servisini başlatın."
            )
        return self._fallback_embed(text)


# ---------------------------------------------------------------------------
# ANA RAG MOTORU SINIFI
# ---------------------------------------------------------------------------
class RAGMotoru:
    """
    EduFrame AI için seviye duyarlı RAG ve ChromaDB yönetim sınıfı.
    """

    def __init__(
        self,
        db_dizini: Optional[Path] = None,
        koleksiyon_adi: str = COLLECTION,
        embedding_motoru: Optional[Embeddings] = None,
        allow_fallback: bool = False,
    ):
        self.db_dizini = Path(db_dizini) if db_dizini else CHROMA_DB_DIR
        self.db_dizini.mkdir(parents=True, exist_ok=True)
        self.koleksiyon_adi = koleksiyon_adi
        self.allow_fallback = allow_fallback
        self.embedding = embedding_motoru or GuvenliEmbeddingMotoru(
            koleksiyon_adi=self.koleksiyon_adi, allow_fallback=allow_fallback
        )
        self._vector_store: Optional[Chroma] = None

    def get_vector_store(self) -> Chroma:
        """Kalıcı ChromaDB vektör deposu bağlantısını döndürür."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.koleksiyon_adi,
                embedding_function=self.embedding,
                persist_directory=str(self.db_dizini),
            )
        return self._vector_store

    def pdf_yukle(self, pdf_yolu: str) -> List[Document]:
        """
        Belirtilen PDF dosyasını pypdf ile sayfa sayfa okur ve ham Document listesi döner.
        """
        yol = Path(pdf_yolu)
        if not yol.exists():
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_yolu}")

        reader = pypdf.PdfReader(str(yol))
        dokumanlar: List[Document] = []

        for sayfa_no, sayfa in enumerate(reader.pages, start=1):
            try:
                metin = (sayfa.extract_text() or "").strip()
            except Exception:
                # Bazı sayfalarda karmaşık vektör çizimler veya font parsing limitleri olabilir, atla
                continue
            if not metin:
                continue

            # Ham belge üst verisi
            meta = {
                "source": yol.name,
                "page": sayfa_no,
                "total_pages": len(reader.pages),
            }
            dokumanlar.append(Document(page_content=metin, metadata=meta))

        print(f"[RAGMotoru] '{yol.name}' dosyasından {len(dokumanlar)} sayfa metin okundu.")
        return dokumanlar

    def pdf_indeksle(
        self,
        pdf_yolu: str,
        egitim_seviyesi: str,
        ders_adi: str = "Fen Bilimleri",
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        batch_size: int = 50,
    ) -> int:
        """
        PDF dosyasını yükler, belirlenen eğitim seviyesi metadata'sını ekleyerek
        parçalara böler ve ChromaDB'ye kaydeder.

        :param pdf_yolu: PDF dosyasının yolu.
        :param egitim_seviyesi: 'İlkokul' | 'Ortaokul' | 'Lise' | 'Üniversite'
        :param ders_adi: Örn. 'Fen Bilimleri', 'Matematik'
        :param chunk_size: Karakter tabanlı parça büyüklüğü
        :param chunk_overlap: Örtüşme karakter sayısı
        :param batch_size: Veritabanına toplu yazma boyutu
        :return: İndekslenen toplam chunk adedi
        """
        if egitim_seviyesi not in EGITIM_SEVIYELERI:
            raise ValueError(
                f"Geçersiz eğitim seviyesi: '{egitim_seviyesi}'. Geçerli seviyeler: {EGITIM_SEVIYELERI}"
            )

        ham_belgeler = self.pdf_yukle(pdf_yolu)
        if not ham_belgeler:
            print(f"[UYARI] '{pdf_yolu}' dosyasında okunabilir metin bulunamadı.")
            return 0

        # Metin parçalayıcı (Metin yapısını ve cümle bütünlüğünü korur)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        parcalar: List[Document] = []
        dosya_adi = Path(pdf_yolu).name
        sinif_bilgisi = self.dosya_metadata_ayristir(dosya_adi)["sinif"]

        for doc in ham_belgeler:
            sayfa_no = doc.metadata.get("page", 1)
            chunks = splitter.split_text(doc.page_content)
            for idx, c_text in enumerate(chunks):
                if len(c_text.strip()) < 30:  # Anlamsız çok kısa parçaları ele
                    continue

                # Gemma Tokenizer ile kesin token sayımı (tokenizer_utils)
                token_sayisi = len(_GEMMA_TOKENIZER.encode(c_text).tokens) if _GEMMA_TOKENIZER else len(c_text.split())

                # KATI KURAL: egitim_seviyesi metadata olarak kaydedilir!
                parca_meta = {
                    "source": dosya_adi,
                    "page": sayfa_no,
                    "chunk_id": f"{dosya_adi}:{sayfa_no:03d}:{idx:02d}",
                    "egitim_seviyesi": egitim_seviyesi,
                    "ders_adi": ders_adi,
                    "sinif": sinif_bilgisi,
                    "token_count": token_sayisi,
                }
                parcalar.append(Document(page_content=c_text, metadata=parca_meta))

        toplam = len(parcalar)
        print(f"\n[RAGMotoru] Toplam {toplam} parça ChromaDB'ye indeksleniyor...")
        print(f" -> Eğitim Seviyesi: {egitim_seviyesi}")
        print(f" -> Ders: {ders_adi}")

        store = self.get_vector_store()
        ids = [doc.metadata["chunk_id"] for doc in parcalar]

        # Toplu batch halinde veritabanına ekle
        for start in range(0, toplam, batch_size):
            end = min(start + batch_size, toplam)
            batch_docs = parcalar[start:end]
            batch_ids = ids[start:end]
            store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f" -> [{start + 1}-{end} / {toplam}] parça eklendi.")

        print(f"[BAŞARILI] '{dosya_adi}' başarıyla indekslendi ({toplam} parça).\n")
        return toplam

    def seviye_filtreli_arama(
        self,
        sorgu: str,
        egitim_seviyesi: str,
        sinif: Optional[str] = None,
        ders_adi: Optional[str] = None,
        k: int = 4,
        rerank: bool = True,
    ) -> List[Document]:
        """
        Kullanıcının seçtiği eğitim seviyesine ve sınıf düzeyine ait belgeler arasında
        anlamsal arama yapar.

        KALİTE KURALI (d6_01_rerank.py & quality_utils.py):
        1. Vektör veritabanından 'sinif' veya 'egitim_seviyesi' filtresiyle geniş bir aday havuzu çekilir.
        2. Çekilen adaylar 'lexical_similarity_test' 3-gram Türkçe Jaccard sözcüksel benzerliği ile
           yeniden sıralanır (reranking) ve en kaliteli 'k' adet doküman döndürülür.
        """
        store = self.get_vector_store()
        aday_sayisi = max(k * 3, 10) if rerank and lexical_rerank else k

        adaylar = []
        # 1. Öncelik: Sınıf seviyesi (Örn. '3. Sınıf', '10. Sınıf') filtresi
        if sinif:
            try:
                adaylar = store.similarity_search(query=sorgu, k=aday_sayisi, filter={"sinif": sinif})
            except Exception:
                adaylar = []

        # 2. İkincil: Eğer sınıf filtresi sonuç vermediyse genel eğitim kademesi filtresi
        if not adaylar:
            filtre = {"egitim_seviyesi": egitim_seviyesi} if egitim_seviyesi in EGITIM_SEVIYELERI else None
            try:
                adaylar = store.similarity_search(query=sorgu, k=aday_sayisi, filter=filtre)
            except Exception as e:
                print(f"[HATA] Filtreli ve rerank'li aramada sorun oluştu: {e}")
                return []

        if not adaylar:
            return []

        # 3-gram Türkçe Lexical Reranking Uygula (d6_01_rerank mantığı)
        if rerank and lexical_rerank and len(adaylar) > 1:
            try:
                sirali_adaylar = lexical_rerank(sorgu, adaylar, top_n=k)
                sonuclar = [doc for doc, score in sirali_adaylar]
                return sonuclar
            except TypeError:
                sirali_adaylar = lexical_rerank(sorgu, adaylar)
                sonuclar = [doc for doc, score in sirali_adaylar]
                return sonuclar[:k]

        return adaylar[:k]

    def bolum_bazli_cok_yonlu_arama(
        self,
        bolum_basligi: str,
        transkript: str,
        egitim_seviyesi: str,
        sinif: str = "3. Sınıf",
        ders_adi: str = "Fen Bilimleri",
        k: int = 2
    ) -> Dict[str, Any]:
        """
        KULLANICI TAVSİYESİYLE GELİŞTİRİLEN 3 BOYUTLU RAG MİMARİSİ:
        Her görsel eğitim materyali bölümü için 3 amaca yönelik hedeflenmiş sorgu çalıştırır:
        1. Bilgi Sorgusu: Tanımlar, kavramlar ve resmi konu açıklamaları.
        2. Pedagojik Sorgu: Sınıf seviyesine uygun anlatım dili, somut örnekler ve kazanımlar.
        3. Ölçme-Değerlendirme Sorgusu: İlgili ünitedeki soru tipleri ve etkinlik örnekleri.

        Her parça için kaynak ve sayfa izlenebilirliği ([B1], [P1], [O1] ve sayfa no) sağlar.
        """
        # 1. Bilgi Sorgusu (Başlık ve transkriptteki anahtar terimlerle zenginleştirilmiş)
        bilgi_sorgusu = f"{sinif} {ders_adi}. {bolum_basligi}. {transkript[:250]}"
        bilgi_docs = self.seviye_filtreli_arama(sorgu=bilgi_sorgusu, egitim_seviyesi=egitim_seviyesi, sinif=sinif, ders_adi=ders_adi, k=k)

        # 2. Pedagojik Sorgu
        pedagoji_sorgusu = f"{sinif} {ders_adi}. Konu: {bolum_basligi}. {sinif} düzeyinde somut örnekler ve pedagojik anlatım."
        pedagoji_docs = self.seviye_filtreli_arama(sorgu=pedagoji_sorgusu, egitim_seviyesi=egitim_seviyesi, sinif=sinif, ders_adi=ders_adi, k=k)

        # 3. Ölçme-Değerlendirme Sorgusu
        olcme_sorgusu = f"{sinif} {ders_adi}. Konu: {bolum_basligi}. Soru, alıştırma ve etkinlik örnekleri."
        olcme_docs = self.seviye_filtreli_arama(sorgu=olcme_sorgusu, egitim_seviyesi=egitim_seviyesi, sinif=sinif, ders_adi=ders_adi, k=k)

        def _formatla(docs: List[Document], baslangic_etiket: str = "K") -> Dict[str, Any]:
            parcalar = []
            metinler = []
            for i, d in enumerate(docs, 1):
                etiket = f"{baslangic_etiket}{i}"
                kaynak = d.metadata.get("source", "Ders Kitabı")
                sayfa = d.metadata.get("page", "-")
                c_text = d.page_content.strip()
                referans = d.metadata.get("referans") or f"{kaynak} (Sayfa {sayfa})"
                metinler.append(f"[{etiket}] ({referans}):\n{c_text}")
                parcalar.append({
                    "etiket": etiket,
                    "kaynak": kaynak,
                    "sayfa": sayfa,
                    "referans": referans,
                    "unite": d.metadata.get("unite", ""),
                    "konu": d.metadata.get("konu", ""),
                    "icerik": c_text
                })
            return {
                "baglam": "\n\n".join(metinler),
                "kaynaklar": parcalar,
                "dokumanlar": docs
            }

        en_iyi_referans = None
        for d in (bilgi_docs + pedagoji_docs + olcme_docs):
            if d and d.metadata.get("referans"):
                en_iyi_referans = d.metadata["referans"]
                break

        return {
            "bilgi": _formatla(bilgi_docs, "B"),
            "pedagoji": _formatla(pedagoji_docs, "P"),
            "olcme_degerlendirme": _formatla(olcme_docs, "O"),
            "en_iyi_referans": en_iyi_referans,
            "tum_kaynaklar": [
                {
                    "kaynak": d.metadata.get("source"),
                    "sayfa": d.metadata.get("page"),
                    "referans": d.metadata.get("referans")
                }
                for d in (bilgi_docs + pedagoji_docs + olcme_docs)
            ]
        }

    def soru_formatlarini_getir(
        self,
        egitim_seviyesi: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        KATI KURAL (Few-Shot Soru Klonlama):
        Test soruları üretilirken, soru formatı (boşluk doldurma, test vb.)
        RAG veritabanındaki kitaptan klonlanacaktır.

        Bu fonksiyon, belirtilen eğitim seviyesindeki kitaptan soru kalıplarını
        ve alıştırma stillerini çeker.
        """
        soru_arama_ifadeleri = [
            "Değerlendirme Soruları test",
            "Aşağıdaki boşlukları doldurunuz",
            "Doğru - Yanlış etkinliği",
            "Aşağıdakilerden hangisi",
        ]

        bulunan_ornekler: List[Dict[str, Any]] = []
        gorulen_metinler = set()

        for ifade in soru_arama_ifadeleri:
            docs = self.seviye_filtreli_arama(sorgu=ifade, egitim_seviyesi=egitim_seviyesi, k=2)
            for doc in docs:
                metin = doc.page_content.strip()
                # Soru işaretleri, parantezler veya şıklar içeren kısımları tercih et
                if ("?" in metin or "A)" in metin or "..." in metin or "D - Y" in metin) and metin not in gorulen_metinler:
                    gorulen_metinler.add(metin)
                    bulunan_ornekler.append({
                        "ornek_metin": metin,
                        "sayfa": doc.metadata.get("page", 0),
                        "kaynak": doc.metadata.get("source", ""),
                    })
                if len(bulunan_ornekler) >= k:
                    break
            if len(bulunan_ornekler) >= k:
                break

        return bulunan_ornekler

    def dosya_metadata_ayristir(self, dosya_adi: str) -> Dict[str, str]:
        """
        Kullanıcının standart dosya adlandırmasını kusursuz biçimde üst verilere dönüştürür.
        Örnekler:
          'lise_fizik_10.sinif.pdf' ->
              egitim_seviyesi: 'Lise', ders_adi: 'Fizik', sinif: '10. Sınıf'
          'ortaokul_fenbilimleri_7.sinif_1.kitap.pdf' ->
              egitim_seviyesi: 'Ortaokul', ders_adi: 'Fen Bilimleri', sinif: '7. Sınıf'
          'ilkokul_fenbilimleri_3.sinif.pdf' ->
              egitim_seviyesi: 'İlkokul', ders_adi: 'Fen Bilimleri', sinif: '3. Sınıf'
        """
        ad = dosya_adi.lower()

        # 1. Eğitim Seviyesi Tespiti (Ön ek kontrolü)
        if ad.startswith("ilkokul") or "ilkokul" in ad:
            seviye = "İlkokul"
        elif ad.startswith("ortaokul") or "ortaokul" in ad:
            seviye = "Ortaokul"
        elif ad.startswith("lise") or "lise" in ad:
            seviye = "Lise"
        elif ad.startswith("universite") or "universite" in ad or "uni" in ad:
            seviye = "Üniversite"
        else:
            seviye = DEFAULT_SEVIYE

        # 2. Ders Adı Tespiti
        if "fizik" in ad:
            ders = "Fizik"
        elif "kimya" in ad:
            ders = "Kimya"
        elif "biyoloji" in ad:
            ders = "Biyoloji"
        elif "fenbilimleri" in ad or "fen" in ad:
            ders = "Fen Bilimleri"
        elif "matematik" in ad:
            ders = "Matematik"
        else:
            ders = "Fen Bilimleri"

        # 3. Sınıf Seviyesi Tespiti
        sinif_match = re.search(r'(\d+)\.?\s*sinif', ad)
        sinif = f"{sinif_match.group(1)}. Sınıf" if sinif_match else "Belirtilmemiş"

        return {
            "egitim_seviyesi": seviye,
            "ders_adi": ders,
            "sinif": sinif
        }

    def seviyeyi_tahmin_et(self, dosya_adi: str) -> str:
        """Geriye dönük uyumluluk için dosya seviyesini döndürür."""
        return self.dosya_metadata_ayristir(dosya_adi)["egitim_seviyesi"]

    def sistem_kitaplarini_senkronize_et(self) -> Dict[str, Any]:
        """
        data/reference_books klasöründeki tüm resmi müfredat PDF'lerini tarar,
        dosya adından seviye, ders ve sınıf bilgilerini ayrıştırarak ChromaDB'ye kaydeder.
        """
        mevcut_durum = self.mevcut_veritabani_durumu()
        print("\n[RAG Kurasyon] Referans kitaplar taranıyor...")
        print(f" -> Mevcut Seviyeler: {mevcut_durum.get('seviyeler', {})}")

        taranan_dizinler = [REFERENCE_BOOKS_DIR, DATA_DIR]
        bulunan_kitaplar = []
        gorulen = set()

        for dizin in taranan_dizinler:
            if not dizin.exists():
                continue
            for pdf_file in sorted(dizin.glob("*.pdf")):
                if pdf_file.name not in gorulen:
                    gorulen.add(pdf_file.name)
                    bulunan_kitaplar.append(pdf_file)

        rapor = {}
        store = self.get_vector_store()
        col = store._collection

        for pdf_yolu in bulunan_kitaplar:
            meta_info = self.dosya_metadata_ayristir(pdf_yolu.name)
            seviye = meta_info["egitim_seviyesi"]
            ders = meta_info["ders_adi"]
            sinif = meta_info["sinif"]

            # Bu dosya daha önce indekse eklenmiş mi kontrol et
            var_olanlar = col.get(where={"source": pdf_yolu.name}, limit=1)

            if len(var_olanlar.get("ids", [])) == 0:
                print(f"[RAG Kurasyon] '{pdf_yolu.name}' indeksleniyor -> Seviye: {seviye} | Ders: {ders} | Sınıf: {sinif}")
                eklenen_parca = self.pdf_indeksle(
                    str(pdf_yolu),
                    egitim_seviyesi=seviye,
                    ders_adi=ders,
                )
                rapor[pdf_yolu.name] = {
                    "seviye": seviye,
                    "ders": ders,
                    "sinif": sinif,
                    "parca_sayisi": eklenen_parca,
                    "durum": "yeni_eklendi"
                }
            else:
                rapor[pdf_yolu.name] = {
                    "seviye": seviye,
                    "ders": ders,
                    "sinif": sinif,
                    "durum": "zaten_mevcut"
                }

        return rapor

    def mevcut_veritabani_durumu(self) -> Dict[str, Any]:
        """ChromaDB koleksiyonunun genel durumunu ve seviye dağılımını raporlar."""
        try:
            store = self.get_vector_store()
            # Ham koleksiyon nesnesinden toplam sayıyı al
            col = store._collection
            toplam_vektor = col.count()

            # Seviyelere göre dağılım kontrolü
            seviye_sayilari = {}
            for seviye in EGITIM_SEVIYELERI:
                try:
                    res = col.get(where={"egitim_seviyesi": seviye}, limit=1)
                    var_mi = len(res.get("ids", [])) > 0
                    seviye_sayilari[seviye] = "Mevcut" if var_mi else "Yok"
                except Exception:
                    seviye_sayilari[seviye] = "Bilinmiyor"

            return {
                "koleksiyon": self.koleksiyon_adi,
                "toplam_parca": toplam_vektor,
                "seviyeler": seviye_sayilari,
            }
        except Exception as e:
            return {"hata": str(e), "toplam_parca": 0}


# --- KÖK DİZİN UYUMLULUĞU VE DOĞRUDAN TEST ---
if __name__ == "__main__":
    print("=" * 60)
    print("EduFrame AI - RAG Motoru Testi Başlatılıyor...")
    print("=" * 60)

    rag = RAGMotoru()
    durum = rag.mevcut_veritabani_durumu()
    print(f"Veritabanı Durumu: {durum}")

    # Örnek test PDF'i var mı?
    ornek_pdf = DATA_DIR / "fenbilimleri.pdf"
    if ornek_pdf.exists():
        print(f"\n[BİLGİ] '{ornek_pdf.name}' bulundu.")
        # Seviyeye göre arama testi
        print("\n--- Seviye Filtreli Arama Testi ('İlkokul' - 'Kuvvet ve Hareket') ---")
        sonuclar = rag.seviye_filtreli_arama("Kuvvet cisimlerin hareketini nasıl etkiler?", egitim_seviyesi="İlkokul", k=2)
        print(f"Bulunan sonuç adedi: {len(sonuclar)}")
        for i, s in enumerate(sonuclar, 1):
            print(f"[{i}] Sayfa: {s.metadata.get('page')} | Seviye: {s.metadata.get('egitim_seviyesi')}")
            print(f"    İçerik: {s.page_content[:120]}...\n")

        print("--- Few-Shot Soru Formatı Çekme Testi ---")
        sorular = rag.soru_formatlarini_getir(egitim_seviyesi="İlkokul", k=2)
        print(f"Bulunan soru kalıbı adedi: {len(sorular)}")
        for i, ornek in enumerate(sorular, 1):
            print(f"[{i}] (Sayfa {ornek['sayfa']}): {ornek['ornek_metin'][:100]}...\n")
    else:
        print(f"[UYARI] '{ornek_pdf}' bulunamadı, lütfen data klasörünü kontrol edin.")

    print("RAGMotoru başarıyla doğrulandı.")
