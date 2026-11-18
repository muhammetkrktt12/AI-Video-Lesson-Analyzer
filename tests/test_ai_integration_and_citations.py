"""
EduFrame AI - Yapay Zeka Entegrasyonu, Hata Dayanıklılığı ve Dinamik RAG Kaynakça Testleri
(tests/test_ai_integration_and_citations.py)
-----------------------------------------------------------------------------------------
Bu test paketi 10 kritik senaryoyu doğrular:
1. Ollama Başarılı Yanıt Senaryosu
2. Ollama Zaman Aşımı -> Gemini Fallback
3. Gemini HTTP 503 Hatası -> Exponential Backoff & Model Fallback
4. İki Yapay Zeka Başarısız -> Pedagojik RAG Fallback
5. İki Yapay Zeka + RAG Yok -> Güvenlik Ağı Şablonu (static_fallback)
6. Dinamik Kaynakça Üretimi (Çoklu Benzersiz RAG Parçaları)
7. Dinamik Kaynakça Tekilleştirme (Deduplication - Bölüm & Soru Birleştirme)
8. Kaynak Bulunamadığında Fallback Mesajı
9. PDF Başlık ve Üst Bilgi Dosya Yolu Temizleme (Sanitization)
10. Cevap Anahtarı Açıklama/Kaynak Ayrımı ve Gerçek RAG Eşleştirmesi
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import requests

from services.gemini_analyzer import GeminiAnalyzer
from services.question_generator import QuestionGenerator
from services.pdf_generator import PDFGenerator


class TestAIIntegrationAndCitations(unittest.TestCase):
    """Yapay Zeka Dayanıklılığı ve Dinamik Kaynakça Testleri."""

    def setUp(self):
        self.analyzer = GeminiAnalyzer(api_key="AIzaTestKeyForMockingOnly123456789")
        self.qgen = QuestionGenerator(api_key="AIzaTestKeyForMockingOnly123456789")
        self.pdf_gen = PDFGenerator()

    # -----------------------------------------------------------------------
    # SENARYO 1: Ollama Başarılı Yanıt Senaryosu
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_1_ollama_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": '{"ders_adi": "Fen Bilimleri", "unite_basligi": "Elektrik", "bolumler": [{"bolum_no": 1, "bolum_basligi": "Giriş", "transkript": "Ders", "duzenlenmis_anlatim": "Konu", "onemli_kavramlar": []}]}'
        }
        mock_post.return_value = mock_resp

        sonuc = self.analyzer.analiz_et(
            video_yolu="test.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="Referans metin",
            video_basligi="Elektrikli Araçlar"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "ollama")
        self.assertEqual(len(sonuc.get("bolumler", [])), 1)
        self.assertEqual(sonuc["bolumler"][0].get("kaynak_tipi"), "ollama")

    # -----------------------------------------------------------------------
    # SENARYO 2: Ollama Zaman Aşımı -> Gemini Fallback
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_2_ollama_timeout_gemini_fallback(self, mock_post):
        def side_effect(url, **kwargs):
            if "localhost" in url or "11434" in url:
                raise requests.exceptions.Timeout("Ollama CPU inference timed out after 90s")
            # Gemini isteği
            m_resp = MagicMock()
            m_resp.status_code = 200
            m_resp.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": '{"ders_adi": "Fen Bilimleri", "unite_basligi": "Elektrik", "bolumler": [{"bolum_no": 1, "bolum_basligi": "Gemini Bölüm", "transkript": "Konuşma", "duzenlenmis_anlatim": "Gemini Anlatım", "onemli_kavramlar": []}]}'
                        }]
                    }
                }]
            }
            return m_resp

        mock_post.side_effect = side_effect

        sonuc = self.analyzer.analiz_et(
            video_yolu="test.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="Referans RAG",
            video_basligi="Elektrik"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "gemini")
        self.assertEqual(sonuc["bolumler"][0].get("kaynak_tipi"), "gemini")
        self.assertEqual(sonuc["bolumler"][0]["bolum_basligi"], "Gemini Bölüm")

    # -----------------------------------------------------------------------
    # SENARYO 3: Gemini 503 Hatası -> Exponential Backoff & Model Fallback
    # -----------------------------------------------------------------------
    @patch("time.sleep", return_value=None)  # Testi hızlandırmak için sleep'i atla
    @patch("requests.post")
    def test_scenario_3_gemini_503_retry_and_model_fallback(self, mock_post, mock_sleep):
        attempt_counts = {"gemini-3.8-flash": 0, "gemini-2.5-flash": 0}

        def side_effect(url, **kwargs):
            if "gemini-3.8-flash" in url:
                attempt_counts["gemini-3.8-flash"] += 1
                resp = MagicMock()
                resp.status_code = 503
                resp.json.return_value = {
                    "error": {
                        "code": 503,
                        "message": "This model is currently experiencing high demand.",
                        "status": "UNAVAILABLE"
                    }
                }
                return resp
            elif "gemini-2.5-flash" in url:
                attempt_counts["gemini-2.5-flash"] += 1
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": '{"bolumler": [{"bolum_no": 1, "bolum_basligi": "Yedek Modelden Başarılı", "transkript": "T", "duzenlenmis_anlatim": "D", "onemli_kavramlar": []}]}'
                            }]
                        }
                    }]
                }
                return resp
            raise ValueError(f"Beklenmeyen URL: {url}")

        mock_post.side_effect = side_effect

        analyzer = GeminiAnalyzer(
            api_key="AIzaValidKeyForTest1234567890",
            model="gemini-3.8-flash",
            gemini_max_retries=2,
            fallback_models=["gemini-2.5-flash"]
        )

        res = analyzer._gemini_ile_analiz_et("Test Prompt")

        self.assertIsNotNone(res)
        self.assertEqual(res["bolumler"][0]["bolum_basligi"], "Yedek Modelden Başarılı")
        # 3.8-flash modeli retries+1 defa (3 kez) 503 verdi, sonra 2.5-flash'a geçti
        self.assertEqual(attempt_counts["gemini-3.8-flash"], 3)
        self.assertEqual(attempt_counts["gemini-2.5-flash"], 1)

    # -----------------------------------------------------------------------
    # SENARYO 4: İki Yapay Zeka Başarısız -> Pedagojik RAG Fallback
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_4_both_ai_fail_rag_fallback(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Network down")

        sonuc = self.analyzer.analiz_et(
            video_yolu="test_video.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="MEB 3. Sınıf Fen Bilimleri 6. Ünite Elektrikli Araçlar...",
            video_basligi="Elektrikli Araç ve Gereçler"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "rag_fallback")
        self.assertTrue(len(sonuc.get("bolumler", [])) >= 10)
        self.assertEqual(sonuc["bolumler"][0].get("kaynak_tipi"), "rag_fallback")

    # -----------------------------------------------------------------------
    # SENARYO 5: İki Yapay Zeka + RAG Yok -> Güvenlik Ağı Şablonu (static_fallback)
    # -----------------------------------------------------------------------
    @patch("requests.post")
    def test_scenario_5_both_ai_and_rag_fail_static_fallback(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Network down")

        sonuc = self.analyzer.analiz_et(
            video_yolu="test_video.mp4",
            egitim_seviyesi="İlkokul",
            sinif_duzeyi="3. Sınıf",
            ders_adi="Fen Bilimleri",
            rag_baglami="",  # RAG bağlamı yok
            video_basligi="Elektrikli Araç ve Gereçler"
        )

        self.assertEqual(sonuc.get("uretim_kaynagi"), "static_fallback")
        self.assertEqual(sonuc["bolumler"][0].get("kaynak_tipi"), "static_fallback")

    # -----------------------------------------------------------------------
    # SENARYO 6: Dinamik Kaynakça Üretimi (Çoklu Benzersiz RAG Parçaları)
    # -----------------------------------------------------------------------
    def test_scenario_6_dynamic_bibliography_multiple_distinct_rag_chunks(self):
        video_analizi = {
            "sinif_duzeyi": "3. Sınıf",
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Bölüm 1 - Elektrikli Araçlar",
                    "rag_kaynaklari": [
                        {
                            "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
                            "dosya_adi": "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
                            "sinif": "3. Sınıf",
                            "unite": "6. Ünite",
                            "konu": "Elektrikli Araç Gereç",
                            "physical_page": 41,
                            "printed_page": 41,
                            "has_printed_page": True
                        }
                    ]
                },
                {
                    "bolum_no": 6,
                    "bolum_basligi": "Bölüm 6 - Aydınlatma Araçları",
                    "rag_kaynaklari": [
                        {
                            "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
                            "dosya_adi": "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
                            "sinif": "3. Sınıf",
                            "unite": "6. Ünite",
                            "konu": "Aydınlatma ve Isınma",
                            "physical_page": 48,
                            "printed_page": 48,
                            "has_printed_page": True
                        }
                    ]
                }
            ]
        }
        sorular = [
            {
                "no": 1,
                "rag_kaynagi": {
                    "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
                    "dosya_adi": "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
                    "sinif": "3. Sınıf",
                    "unite": "6. Ünite",
                    "konu": "Elektrikli Araç Gereç",
                    "physical_page": 41,
                    "printed_page": 41,
                    "has_printed_page": True
                }
            }
        ]

        kaynaklar = self.pdf_gen._dinamik_kaynakca_derle(video_analizi, sorular)

        self.assertEqual(len(kaynaklar), 2)
        # 1. parça (Sayfa 41) hem Bölüm 1 hem Soru 1'de kullanıldı
        sayfa_41_entry = [k for k in kaynaklar if "41" in k][0]
        self.assertIn("MEB 3. Sınıf Fen Bilimleri (2. Kitap)", sayfa_41_entry)
        self.assertIn("ilkokul_fenbilimleri_3.sinif_2.kitap.pdf", sayfa_41_entry)
        self.assertIn("Fiziksel Sayfa: 41", sayfa_41_entry)
        self.assertIn("Basılı Sayfa: 41", sayfa_41_entry)
        self.assertIn("Bölüm 1", sayfa_41_entry)
        self.assertIn("Soru 1", sayfa_41_entry)

        # 2. parça (Sayfa 48) Bölüm 6'da kullanıldı
        sayfa_48_entry = [k for k in kaynaklar if "48" in k][0]
        self.assertIn("Fiziksel Sayfa: 48", sayfa_48_entry)
        self.assertIn("Bölüm 6", sayfa_48_entry)

    # -----------------------------------------------------------------------
    # SENARYO 7: Dinamik Kaynakça Tekilleştirme (Deduplication)
    # -----------------------------------------------------------------------
    def test_scenario_7_dynamic_bibliography_deduplication(self):
        aynı_kaynak = {
            "kitap_adi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap)",
            "dosya_adi": "ilkokul_fenbilimleri_3.sinif_2.kitap.pdf",
            "sinif": "3. Sınıf",
            "unite": "6. Ünite",
            "konu": "Elektrikli Araç Gereç",
            "physical_page": 41,
            "printed_page": 41,
            "has_printed_page": True
        }
        video_analizi = {
            "bolumler": [
                {"bolum_no": 1, "rag_kaynaklari": [aynı_kaynak]},
                {"bolum_no": 2, "rag_kaynaklari": [aynı_kaynak]},
                {"bolum_no": 5, "rag_kaynaklari": [aynı_kaynak]}
            ]
        }
        sorular = [
            {"no": 1, "rag_kaynagi": aynı_kaynak},
            {"no": 4, "rag_kaynagi": aynı_kaynak}
        ]

        kaynaklar = self.pdf_gen._dinamik_kaynakca_derle(video_analizi, sorular)

        # 3 bölüm ve 2 soru aynı kaynağı referans verdi, tek bir kayıt olmalı
        self.assertEqual(len(kaynaklar), 1)
        entry = kaynaklar[0]
        self.assertIn("Bölüm 1", entry)
        self.assertIn("Bölüm 2", entry)
        self.assertIn("Bölüm 5", entry)
        self.assertIn("Soru 1", entry)
        self.assertIn("Soru 4", entry)

    # -----------------------------------------------------------------------
    # SENARYO 8: Kaynak Bulunamadığında Fallback Mesajı
    # -----------------------------------------------------------------------
    def test_scenario_8_fallback_message_when_rag_missing(self):
        video_analizi = {"bolumler": []}
        sorular = []

        kaynaklar = self.pdf_gen._dinamik_kaynakca_derle(video_analizi, sorular)

        self.assertEqual(len(kaynaklar), 1)
        self.assertIn("Bu bölüm için doğrulanmış ders kitabı kaynağı bulunamadı", kaynaklar[0])

    # -----------------------------------------------------------------------
    # SENARYO 9: PDF Başlık ve Üst Bilgi Dosya Yolu Temizleme (Sanitization)
    # -----------------------------------------------------------------------
    def test_scenario_9_pdf_header_title_path_sanitization(self):
        yol_1 = r"C:\Users\nurmu\OneDrive\Desktop\AIVideoLessonAnalyzer\temp_medya\videos\elektrikli_araclar_dersi.mp4"
        yol_2 = "C:/Users/test/workspace/videos/fen_3_sinif_video.mkv"
        sade_baslik = "3. Sınıf Fen Bilimleri Elektrik Ünitesi"

        self.assertEqual(self.pdf_gen._temizle_dosya_yolu(yol_1), "elektrikli araclar dersi")
        self.assertEqual(self.pdf_gen._temizle_dosya_yolu(yol_2), "fen 3 sinif video")
        self.assertEqual(self.pdf_gen._temizle_dosya_yolu(sade_baslik), sade_baslik)

    # -----------------------------------------------------------------------
    # SENARYO 10: Cevap Anahtarı Açıklama/Kaynak Ayrımı ve Gerçek RAG Eşleştirmesi
    # -----------------------------------------------------------------------
    def test_scenario_10_answer_key_source_separation_and_binding(self):
        # 1. Gerçek RAG referansı bulunan bölüm
        video_analizi = {
            "unite_basligi": "Elektrikli Araç ve Gereçler",
            "bolumler": [
                {
                    "bolum_no": 1,
                    "bolum_basligi": "Bölüm 1 - Elektrikli Araç ve Gereçler",
                    "onemli_kavramlar": [{"kavram": "Elektrikli Araç ve Gereçler"}],
                    "kitap_referansi": "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite (Sayfa 41-42)"
                }
            ]
        }

        # Soru üret
        sorular = self.qgen.soru_uret(video_analizi, egitim_seviyesi="İlkokul", soru_sayisi=1)
        q = sorular[0]

        # Açıklama salt gerekçe içermeli, uydurma 's. 178' gibi referans içermemeli
        self.assertNotIn("s. 178", q["aciklama"])
        self.assertNotIn("Sayfa 178", q["aciklama"])

        # Referans gerçek RAG kaynağından bağlanmış olmalı
        self.assertEqual(q["kazanim_referansi"], "MEB 3. Sınıf Fen Bilimleri (2. Kitap) - 6. Ünite (Sayfa 41-42)")
        self.assertTrue(q.get("rag_dogrulandi"))

        # 2. RAG eşleşmesi olmayan boş analiz senaryosu
        sorular_bos = self.qgen.soru_uret({}, egitim_seviyesi="İlkokul", soru_sayisi=1)
        q_bos = sorular_bos[0]

        # Asla sayfa uydurmamalı, doğrulanmadığını belirtmeli
        self.assertEqual(q_bos["kazanim_referansi"], "Doğrulanmış ders kitabı kaynağı bulunamadı")
        self.assertFalse(q_bos.get("rag_dogrulandi"))


if __name__ == "__main__":
    unittest.main()
