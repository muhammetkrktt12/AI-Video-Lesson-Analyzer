"""
EduFrame AI - RAG Motoru (Root Compatibility Wrapper)
Bu dosya kök dizindeki çağrılar ve geriye dönük uyumluluk için
services.rag_motoru modülünü sarmalar.
"""

from services.rag_motoru import RAGMotoru, GuvenliEmbeddingMotoru

__all__ = ["RAGMotoru", "GuvenliEmbeddingMotoru"]

if __name__ == "__main__":
    motor = RAGMotoru()
    print("[RAGMotoru Wrapper] Hazır.")
