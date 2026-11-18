"""
Mevcut d5_05_search.py ve chroma_db ile 3-gram Türkçe lexical reranking modülü (Test ve Karşılaştırma).
Temel benzerlik fonksiyonları utils.lexical_tools içerisinden sağlanır.
"""

from utils.lexical_tools import (
    kelimelere_ayir,
    ucgramlar,
    lexical_similarity,
    rerank_candidates,
)

def ara_ve_sirala(soru, aday_sayisi=8, lexical_agirlik=0.4, sonuc_sayisi=3):
    """Chroma'dan arama yapar ve hibrit skorla yeniden sıralar: [(toplam, eski_sira, lexical_puan, belge), ...]"""
    if aday_sayisi < 1 or sonuc_sayisi < 1:
        raise ValueError("Aday ve sonuç sayısı pozitif olmalı")
    if not 0 <= lexical_agirlik <= 1:
        raise ValueError("Lexical ağırlık 0 ile 1 arasında olmalı")

    # d5_05_search yalnızca test amaçlı doğrudan arama çağrıldığında tembelce yüklenir
    from d5_05_search import search

    adaylar = search(soru, k=aday_sayisi)
    puanli = []
    for sira, belge in enumerate(adaylar, 1):
        anlamsal_sira_puani = 1 / sira
        lexical_puan = lexical_similarity(soru, belge.page_content)
        toplam = ((1 - lexical_agirlik) * anlamsal_sira_puani + lexical_agirlik * lexical_puan)
        puanli.append((toplam, sira, lexical_puan, belge))

    puanli.sort(key=lambda item: (-item[0], item[1]))
    return puanli[:sonuc_sayisi]


def main():
    soru = input("Soru : ").strip()
    soru = soru or "Maddenin halleri nelerdir, açıkla?"
    k_input = input("Chroma aday sayısı k [8]: ").strip()
    k = int(k_input) if k_input else 8
    agirlik_input = input("Lexical ağırlık 0-1 [0.4]: ").strip()
    agirlik = float(agirlik_input.replace(",", ".")) if agirlik_input else 0.4

    print("\n--- CHROMA'NIN İLK ANLAMSAL SIRALAMASI + LEXICAL PUANLARI ---")
    from d5_05_search import search
    adaylar = search(soru, k=k)

    for i, belge in enumerate(adaylar, 1):
        chunk_id = belge.metadata.get("chunk_id", "-")
        page = belge.metadata.get("page", "-")
        source = belge.metadata.get("source", "?")

        lexical = lexical_similarity(soru, belge.page_content)

        print(
            f"{i:>2}. {source} (Sayfa {page}) | {chunk_id} "
            f"| Lexical={lexical:.3f}"
        )

    print("\n--- 3-GRAM LEXICAL & HİBRİT YENİDEN SIRALAMA ---")
    sirali = ara_ve_sirala(soru, aday_sayisi=k, lexical_agirlik=agirlik)
    for i, (toplam, eski_sira, lexical, belge) in enumerate(sirali, 1):
        chunk_id = belge.metadata.get("chunk_id", "-")
        page = belge.metadata.get("page", "-")
        source = belge.metadata.get("source", "?")
        print(f"{i}. {source} (Sayfa {page}) | {chunk_id} | Eski Sıra={eski_sira} | Lexical={lexical:.3f} | Birleşik={toplam:.3f}")
        print("   Metin:", belge.page_content[:150].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()
