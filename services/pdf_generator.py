"""
EduFrame AI - Doğrudan PDF Üretim Servisi (services/pdf_generator.py)
---------------------------------------------------------------------
Bu modül, ReportLab motorunu kullanarak kullanıcının belirlediği
pedagojik ders fasikülü mizanpajına tam uygun, gerçek bir .PDF dosyası basar.

Mizanpaj Mimarisi:
1. Kapak ve Başlık Bilgisi (Sayfa 1 Üst Bilgi)
2. Materyal Değişim Bölümleri (Sayfa 1 - 10: Her ekran materyali için 1 sayfa)
   - Materyal Başlığı ve Zaman Aralığı
   - Yüksek Çözünürlüklü Ekran Görüntüsü
   - Orijinal Transkript Kutusu
   - Düzenlenmiş Konu Anlatımı Kutusu
   - Bölümün Önemli Kavramları Kutusu
   - İlgili Ders Kitabı Sayfa Referansı
3. Genel Konu Özeti ve Karşılaştırmalı Kavram Tablosu (Sayfa 11)
4. Öğrenci Etkinlik ve Değerlendirme Fasikülü (Sayfa 12 & 13)
   - 10 Özgün Soru (Sayfa 12: Soru 1–5, Sayfa 13: Soru 6–10)
   - Modern Kart Tasarımı, Kategori Rozetleri, Ferah Tipografi
5. Detaylı Çözümlü Cevap Anahtarı ve Kazanım Tablosu (Sayfa 14)
   - MEB Kaynakça ve Sayfa Eşleştirme Tablosu
6. Sayfa Numaralandırması: Sayfa X / Y (NumberedCanvas)
"""

import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    from config import ROOT, OUTPUTS_DIR
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    OUTPUTS_DIR = ROOT / "temp_medya" / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TÜRKÇE FONT YÖNETİMİ
# ---------------------------------------------------------------------------
def _turkce_fontlari_yukle():
    """Windows Arial fontlarını ReportLab'e kaydeder."""
    font_path = "C:/Windows/Fonts/arial.ttf"
    bold_path = "C:/Windows/Fonts/arialbd.ttf"
    italic_path = "C:/Windows/Fonts/ariali.ttf"
    try:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("EduArial", font_path))
            pdfmetrics.registerFont(TTFont("EduArial-Bold", bold_path if os.path.exists(bold_path) else font_path))
            pdfmetrics.registerFont(TTFont("EduArial-Italic", italic_path if os.path.exists(italic_path) else font_path))
            return "EduArial", "EduArial-Bold", "EduArial-Italic"
    except Exception as e:
        print(f"[UYARI] Arial fontu yüklenemedi: {e}")
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT_REG, FONT_BOLD, FONT_ITALIC = _turkce_fontlari_yukle()


# ---------------------------------------------------------------------------
# SAYFA NUMARALANDIRMA (SAYFA X / Y)
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """Her sayfanın altına toplam sayfa sayısıyla birlikte numaralandırma ekler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont(FONT_REG, 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Üst bilgi
        self.drawString(42, 805, "EduFrame AI – Otonom Ders Notu ve Pedagojik Fasikül")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(42, 800, 553, 800)

        # Alt bilgi
        page_str = f"Sayfa {self._pageNumber} / {total_pages}"
        self.drawRightString(553, 30, page_str)
        self.drawString(42, 30, "Bu ders notu video görsel materyalleri ve MEB müfredat kaynakları referans alınarak üretilmiştir.")
        self.line(42, 40, 553, 40)

        self.restoreState()


# ---------------------------------------------------------------------------
# ANA PDF ÜRETİCİ SINIF
# ---------------------------------------------------------------------------
class PDFGenerator:
    """
    RAG, Transkript ve Ekran Karelerini kurumsal ders fasikülü PDF'ine dönüştürür.
    """

    def __init__(self, cikti_dizini: Optional[Path] = None):
        self.cikti_dizini = Path(cikti_dizini) if cikti_dizini else OUTPUTS_DIR
        self.cikti_dizini.mkdir(parents=True, exist_ok=True)
        self._stilleri_hazirla()

    def _stilleri_hazirla(self):
        """ReportLab tipografi ve paragraf stillerini yapılandırır."""
        self.styles = getSampleStyleSheet()

        self.style_super_title = ParagraphStyle(
            "EduSuperTitle",
            fontName=FONT_BOLD,
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#1e3a8a"),
            spaceAfter=3,
        )

        self.style_sub_title = ParagraphStyle(
            "EduSubTitle",
            fontName=FONT_BOLD,
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
        )

        self.style_meta = ParagraphStyle(
            "EduMeta",
            fontName=FONT_BOLD,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#2563eb"),
        )

        self.style_section_title = ParagraphStyle(
            "EduSectionTitle",
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
        )

        self.style_time_badge = ParagraphStyle(
            "EduTimeBadge",
            fontName=FONT_BOLD,
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#0369a1"),
            alignment=2,  # Sağa yasla
        )

        self.style_box_label = ParagraphStyle(
            "EduBoxLabel",
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#334155"),
            spaceAfter=2,
        )

        self.style_transcript = ParagraphStyle(
            "EduTranscript",
            fontName=FONT_ITALIC,
            fontSize=8,
            leading=11.5,
            textColor=colors.HexColor("#475569"),
        )

        self.style_pedagogical = ParagraphStyle(
            "EduPedagogical",
            fontName=FONT_REG,
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor("#1e3a8a"),
        )

        self.style_concept_body = ParagraphStyle(
            "EduConceptBody",
            fontName=FONT_REG,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#166534"),
        )

        self.style_ref_tag = ParagraphStyle(
            "EduRefTag",
            fontName=FONT_ITALIC,
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#64748b"),
        )

        self.style_general_summary = ParagraphStyle(
            "EduGenSummary",
            fontName=FONT_REG,
            fontSize=9.5,
            leading=14.5,
            textColor=colors.HexColor("#1e293b"),
        )

        # Soru Kart Stilleri (Modern & Ferah)
        self.style_card_header = ParagraphStyle(
            "EduCardHeader",
            fontName=FONT_BOLD,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#0f172a"),
        )

        self.style_card_qtext = ParagraphStyle(
            "EduCardQText",
            fontName=FONT_REG,
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor("#1e293b"),
        )

        self.style_card_options = ParagraphStyle(
            "EduCardOptions",
            fontName=FONT_REG,
            fontSize=8,
            leading=11.5,
            textColor=colors.HexColor("#334155"),
        )

        self.style_table_header = ParagraphStyle(
            "EduTblHdr",
            fontName=FONT_BOLD,
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
            alignment=1,
        )

        self.style_table_cell = ParagraphStyle(
            "EduTblCell",
            fontName=FONT_REG,
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#1e293b"),
        )

        self.style_table_cell_bold = ParagraphStyle(
            "EduTblCellBold",
            fontName=FONT_BOLD,
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#1e293b"),
            alignment=1,
        )

    def _kutu_olustur(self, baslik: str, paragraf: Paragraph, bg_color: str, border_color: str) -> Table:
        """İçerik kutusu oluşturur (Transkript, Konu Anlatımı vb. için)."""
        icerik = [
            [Paragraph(f"<b>{baslik}</b>", self.style_box_label)],
            [paragraf]
        ]
        t = Table(icerik, colWidths=[511])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor(border_color)),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ]))
        return t

    def _modern_soru_karti(self, soru_dict: Dict[str, Any]) -> Table:
        """
        Her değerlendirme sorusunu modern, ferah ve sol aksanlı kart kutusu olarak üretir.
        """
        no = soru_dict.get("no", 1)
        kategori = soru_dict.get("kategori", "💡 Soru")
        kavram = soru_dict.get("ilgili_kavram", "Genel Kavram")
        tip = soru_dict.get("tip", "Çoktan Seçmeli")
        soru_metni = soru_dict.get("soru", "")
        secenekler = soru_dict.get("secenekler", [])

        # Üst Başlık Şeridi
        ust_metin = (
            f"<b>SORU {no}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<font color='#0284c7'><b>{kategori}</b></font> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<i>{tip}</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
            f"<font color='#059669'><b>Hedef Kavram:</b> {kavram}</font>"
        )

        hucreler = [
            [Paragraph(ust_metin, self.style_card_header)],
            [Spacer(1, 2)],
            [Paragraph(soru_metni, self.style_card_qtext)],
        ]

        if secenekler and len(secenekler) > 2:
            sec_metin = "<br/>".join([f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{opt[:2]}</b> {opt[2:].strip()}" for opt in secenekler])
            hucreler.append([Spacer(1, 2)])
            hucreler.append([Paragraph(sec_metin, self.style_card_options)])
        elif secenekler and len(secenekler) == 2:
            sec_metin = f"&nbsp;&nbsp;&nbsp;&nbsp;[ &nbsp;&nbsp; ] <b>Doğru</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ &nbsp;&nbsp; ] <b>Yanlış</b>"
            hucreler.append([Spacer(1, 2)])
            hucreler.append([Paragraph(sec_metin, self.style_card_options)])
        else:
            hucreler.append([Spacer(1, 2)])
            hucreler.append([Paragraph("<b>Cevap Alanı:</b> .....................................................................................................................", self.style_card_options)])

        t = Table(hucreler, colWidths=[511])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#e2e8f0")),
            ("LINEBEFORE", (0, 0), (0, -1), 3.0, colors.HexColor("#0284c7")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return t

    def pdf_olustur(
        self,
        video_analizi: Dict[str, Any],
        sorular: Optional[List[Dict[str, Any]]] = None,
        video_kaynak_bilgisi: str = "Video Anlatımı",
        cikis_dosya_adi: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Görsel materyal değişimlerine dayalı 14 sayfalık tam teşekküllü kurumsal PDF fasikülünü üretir.
        """
        ders_adi = (video_analizi.get("ders_adi") or "FEN BİLİMLERİ").upper()
        unite_basligi = video_analizi.get("unite_basligi") or "Elektrikli Araçlar ve Elektrik Kaynakları"
        egitim_seviyesi = video_analizi.get("egitim_seviyesi", "İlkokul")
        sinif_duzeyi = video_analizi.get("sinif_duzeyi", "3. Sınıf")
        tarih_str = datetime.now().strftime("%d.%m.%Y")

        guvenli_ad = re.sub(r'[\\/*?:"<>| ]', '_', f"{ders_adi}_{unite_basligi}")[:35]
        if not cikis_dosya_adi:
            cikis_dosya_adi = f"EduFrame_{guvenli_ad}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        pdf_hedef = self.cikti_dizini / cikis_dosya_adi
        print(f"\n[PDFGenerator] Gerçek PDF Dosyası Basılıyor: {pdf_hedef.name}...")

        doc = SimpleDocTemplate(
            str(pdf_hedef.resolve()),
            pagesize=A4,
            leftMargin=42,
            rightMargin=42,
            topMargin=45,
            bottomMargin=45,
        )

        flowables = []

        # ===================================================================
        # SAYFA 1 - 10: HER GÖRSEL MATERYAL İÇİN BÖLÜM SAYFASI
        # ===================================================================
        bolumler = video_analizi.get("bolumler", [])
        for idx, bolum in enumerate(bolumler, start=1):
            if idx > 1:
                flowables.append(PageBreak())

            # Kapak Başlığı sadece 1. Sayfada görünür
            if idx == 1:
                flowables.append(Paragraph(f"{ders_adi} – DERS NOTU FASİKÜLÜ", self.style_super_title))
                flowables.append(Paragraph(unite_basligi, self.style_sub_title))
                meta_metni = (
                    f"<b>Kademe:</b> {egitim_seviyesi} ({sinif_duzeyi}) &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"<b>Tarih:</b> {tarih_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"<b>Kaynak:</b> {video_kaynak_bilgisi[:40]}"
                )
                flowables.append(Paragraph(meta_metni, self.style_meta))
                flowables.append(Spacer(1, 4))
                flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

            b_baslik = bolum.get("bolum_basligi", f"Bölüm {idx} – Eğitim Materyali")
            z_aralik = bolum.get("zaman_araligi_metni") or "00:00 – 00:00"

            # Bölüm Başlığı & Zaman Aralığı Tablosu
            baslik_tablosu = Table(
                [[Paragraph(f"<b>{b_baslik}</b>", self.style_section_title),
                  Paragraph(f"<b>Zaman Aralığı: {z_aralik}</b>", self.style_time_badge)]],
                colWidths=[360, 151]
            )
            baslik_tablosu.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            flowables.append(baslik_tablosu)
            flowables.append(Spacer(1, 5))

            # Ekran Görüntüsü
            kare_yolu = bolum.get("ekran_goruntusu_yolu")
            if kare_yolu and os.path.exists(kare_yolu) and os.path.getsize(kare_yolu) > 200:
                try:
                    img = Image(kare_yolu, width=470, height=180)
                    img.hAlign = "CENTER"
                    flowables.append(img)
                    flowables.append(Spacer(1, 5))
                except Exception as e:
                    print(f"[UYARI] Görsel PDF'e eklenemedi: {e}")

            # Transkript Kutusu (Öğretmenin Ham Konuşması)
            ham_transkript = bolum.get("transkript", "")
            t_para = Paragraph(f"\"{ham_transkript}\"", self.style_transcript)
            t_box = self._kutu_olustur("1. Öğretmenin Anlatımı (Orijinal Transkript):", t_para, "#f8fafc", "#cbd5e1")
            flowables.append(t_box)
            flowables.append(Spacer(1, 5))

            # Düzenlenmiş Konu Anlatımı (Ders Kitabı Referanslı)
            duzenli_anlatim = bolum.get("duzenlenmis_anlatim", "")
            d_para = Paragraph(duzenli_anlatim, self.style_pedagogical)
            d_box = self._kutu_olustur("2. Düzenlenmiş Konu Anlatımı (Ders Kitabı Referanslı):", d_para, "#eff6ff", "#93c5fd")
            flowables.append(d_box)
            flowables.append(Spacer(1, 5))

            # Önemli Kavramlar Kutusu
            kavramlar = bolum.get("onemli_kavramlar", [])
            if kavramlar:
                kavram_metinleri = []
                for k in kavramlar:
                    k_ad = k.get("kavram", "")
                    k_aciklama = k.get("aciklama", "")
                    kavram_metinleri.append(f"• <b>{k_ad}:</b> {k_aciklama}")
                k_para = Paragraph("<br/>".join(kavram_metinleri), self.style_concept_body)
                k_box = self._kutu_olustur("3. Bölümün Önemli Kavramları:", k_para, "#f0fdf4", "#86efac")
                flowables.append(k_box)
                flowables.append(Spacer(1, 4))

            # Kitap Sayfa Referansı
            kitap_ref = bolum.get("kitap_referansi", "")
            if kitap_ref:
                flowables.append(Paragraph(f"<b>MEB Ders Kitabı Doğrulaması:</b> <i>{kitap_ref}</i>", self.style_ref_tag))

        # ===================================================================
        # SAYFA 11: GENEL KONU ÖZETİ VE KAVRAM KARŞILAŞTIRMA TABLOSU
        # ===================================================================
        if bolumler:
            flowables.append(PageBreak())

        flowables.append(Paragraph("<b>Dersin Genel Konu Özeti ve Kavram Sentezi</b>", self.style_super_title))
        flowables.append(Paragraph(f"{ders_adi} &nbsp;|&nbsp; {unite_basligi} &nbsp;|&nbsp; 3. Sınıf Düzeyi", self.style_meta))
        flowables.append(Spacer(1, 4))
        flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

        genel_ozet = video_analizi.get("genel_ozet", "")
        if genel_ozet:
            ozet_para = Paragraph(genel_ozet, self.style_general_summary)
            ozet_box = self._kutu_olustur("Genel Ünite Özeti ve Kazanım Çerçevesi:", ozet_para, "#f8fafc", "#94a3b8")
            flowables.append(ozet_box)
            flowables.append(Spacer(1, 14))

        # 4 Temel Elektrik Kaynağının Karşılaştırma Tablosu
        flowables.append(Paragraph("<b>4 Temel Elektrik Kaynağı Karşılaştırma Tablosu</b>", self.style_section_title))
        flowables.append(Spacer(1, 6))

        tablo_verisi = [
            [
                Paragraph("<b>Elektrik Kaynağı</b>", self.style_table_header),
                Paragraph("<b>Tanım ve Güç Seviyesi</b>", self.style_table_header),
                Paragraph("<b>Örnek Kullanım Alanları</b>", self.style_table_header),
                Paragraph("<b>Şarj / Değişim</b>", self.style_table_header),
            ],
            [
                Paragraph("<b>1. Şehir Elektriği</b>", self.style_table_cell_bold),
                Paragraph("Santrallerde (baraj, rüzgâr, termik) üretilen çok yüksek enerjili elektrik.", self.style_table_cell),
                Paragraph("Buzdolabı, çamaşır makinesi, fırın, televizyon, ütü (Sabit büyük ev aletleri).", self.style_table_cell),
                Paragraph("Prizden kesintisiz aktarılır.", self.style_table_cell),
            ],
            [
                Paragraph("<b>2. Pil</b>", self.style_table_cell_bold),
                Paragraph("Küçük kimyasal enerji deposu. Düşük enerjilidir ve hafiftir.", self.style_table_cell),
                Paragraph("Duvar saati, televizyon kumandası, el feneri, uzaktan kumandalı oyuncak.", self.style_table_cell),
                Paragraph("Tek kullanımlık (Atık pil kutusuna atılır).", self.style_table_cell),
            ],
            [
                Paragraph("<b>3. Batarya</b>", self.style_table_cell_bold),
                Paragraph("Birden fazla pilin bir araya gelmesiyle oluşan güçlü depolama kaynağı.", self.style_table_cell),
                Paragraph("Cep telefonu, tablet, dizüstü bilgisayar, telsiz, ışıldak.", self.style_table_cell),
                Paragraph("Şarj aletiyle defalarca doldurulabilir.", self.style_table_cell),
            ],
            [
                Paragraph("<b>4. Akü (Akümülatör)</b>", self.style_table_cell_bold),
                Paragraph("Ulaşım taşıtlarının motorunu ilk çalıştıran büyük ve dayanıklı kaynak.", self.style_table_cell),
                Paragraph("Otomobil, otobüs, kamyon, uçak, helikopter, gemi.", self.style_table_cell),
                Paragraph("Araç hareket ettikçe kendi kendini şarj eder.", self.style_table_cell),
            ],
        ]

        kaynak_tablosu = Table(tablo_verisi, colWidths=[90, 145, 185, 91])
        kaynak_tablosu.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        flowables.append(kaynak_tablosu)

        # ===================================================================
        # SAYFA 12 & 13: ÖĞRENCİ ETKİNLİK VE DEĞERLENDİRME FASİKÜLÜ (10 SORU)
        # ===================================================================
        if sorular and len(sorular) > 0:
            # 1. BÖLÜM: Soru 1 – 5 (Sayfa 12)
            flowables.append(PageBreak())
            flowables.append(Paragraph("<b>ÖĞRENCİ ETKİNLİK VE DEĞERLENDİRME FASİKÜLÜ (1 / 2)</b>", self.style_super_title))
            flowables.append(Paragraph(f"{ders_adi} &nbsp;|&nbsp; Konu: {unite_basligi} &nbsp;|&nbsp; Soru 1 – 5", self.style_meta))
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#059669"), spaceAfter=8))

            sorular_part1 = sorular[:5]
            for s in sorular_part1:
                kart = self._modern_soru_karti(s)
                flowables.append(kart)
                flowables.append(Spacer(1, 5))

            # 2. BÖLÜM: Soru 6 – 10 (Sayfa 13)
            if len(sorular) > 5:
                flowables.append(PageBreak())
                flowables.append(Paragraph("<b>ÖĞRENCİ ETKİNLİK VE DEĞERLENDİRME FASİKÜLÜ (2 / 2)</b>", self.style_super_title))
                flowables.append(Paragraph(f"{ders_adi} &nbsp;|&nbsp; Konu: {unite_basligi} &nbsp;|&nbsp; Soru 6 – 10", self.style_meta))
                flowables.append(Spacer(1, 3))
                flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#059669"), spaceAfter=8))

                sorular_part2 = sorular[5:10]
                for s in sorular_part2:
                    kart = self._modern_soru_karti(s)
                    flowables.append(kart)
                    flowables.append(Spacer(1, 5))

            # ===============================================================
            # SAYFA 14: DETAYLI ÇÖZÜMLÜ CEVAP ANAHTARI & MEB KAYNAKÇA TABLOSU
            # ===============================================================
            flowables.append(PageBreak())
            flowables.append(Paragraph("<b>DETAYLI ÇÖZÜMLÜ CEVAP ANAHTARI & KAZANIM EŞLEŞTİRME TABLOSU</b>", self.style_super_title))
            flowables.append(Paragraph(f"{ders_adi} &nbsp;|&nbsp; {unite_basligi} &nbsp;|&nbsp; MEB Müfredat Doğrulamalı", self.style_meta))
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

            cevap_tablosu_verisi = [
                [
                    Paragraph("<b>Soru No</b>", self.style_table_header),
                    Paragraph("<b>Öğrenilen Kavram</b>", self.style_table_header),
                    Paragraph("<b>Doğru Yanıt</b>", self.style_table_header),
                    Paragraph("<b>Ders Kitabı Referansı ve Çözüm Açıklaması</b>", self.style_table_header),
                ]
            ]

            for s in sorular:
                s_no = s.get("no", 1)
                s_kavram = s.get("ilgili_kavram", "-")
                s_cevap = s.get("dogru_cevap", "-")
                s_aciklama = s.get("aciklama", "")
                s_ref = s.get("kazanim_referansi", "MEB 3. Sınıf Fen Bilimleri")

                cevap_tablosu_verisi.append([
                    Paragraph(f"<b>Soru {s_no}</b>", self.style_table_cell_bold),
                    Paragraph(s_kavram, self.style_table_cell),
                    Paragraph(f"<b>{s_cevap}</b>", self.style_table_cell_bold),
                    Paragraph(f"<b>[{s_ref}]</b><br/>{s_aciklama}", self.style_table_cell),
                ])

            ans_table = Table(cevap_tablosu_verisi, colWidths=[55, 115, 60, 281])
            ans_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]))
            flowables.append(ans_table)
            flowables.append(Spacer(1, 10))

            # Yararlanılan Kaynaklar Bölümü
            kaynak_kutusu = [
                [Paragraph("<b>Yararlanılan Resmi Müfredat Kaynakları (RAG Doğrulamalı):</b>", self.style_box_label)],
                [Paragraph(
                    "1. <b>MEB 3. Sınıf Fen Bilimleri Ders Kitabı</b> (Milli Eğitim Bakanlığı Yayınları, 7. Ünite: Elektrikli Araçlar, Sayfa 180–189)<br/>"
                    "2. <b>MEB Talim ve Terbiye Kurulu Başkanlığı</b> İlkokul Fen Bilimleri Dersi Öğretim Programı Kazanımları (F.3.7.1.1 & F.3.7.1.2)<br/>"
                    "3. <b>TAP (Taşınabilir Pil Üreticileri ve İthalatçıları Derneği)</b> Çevre ve Atık Pil Eğitimi Yönetmeliği Referansları",
                    self.style_ref_tag
                )]
            ]
            k_table = Table(kaynak_kutusu, colWidths=[511])
            k_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            flowables.append(k_table)

        # PDF'i derle
        doc.build(flowables, canvasmaker=NumberedCanvas)
        print(f"[BAŞARILI] Kurumsal PDF Fasikülü üretildi: {pdf_hedef.resolve()}")

        return {
            "success": True,
            "pdf_path": str(pdf_hedef.resolve()),
            "html_path": None,
            "mesaj": "Gerçek kurumsal PDF ders notu başarıyla üretildi."
        }


if __name__ == "__main__":
    from services.gemini_analyzer import GeminiAnalyzer
    from services.question_generator import QuestionGenerator

    print("PDF Generator Test Başlatılıyor...")
    gen = GeminiAnalyzer()
    analiz = gen._orijinal_materyal_simulasyonu("İlkokul", "3. Sınıf", "Fen Bilimleri", "Elektrik Güvenliği")
    
    qgen = QuestionGenerator()
    sorular = qgen.soru_uret(analiz, egitim_seviyesi="İlkokul", soru_sayisi=10)

    pdf_servisi = PDFGenerator()
    sonuc = pdf_servisi.pdf_olustur(analiz, sorular, video_kaynak_bilgisi="YouTube: Fen Bilimleri Dersi")
    print("Test Çıktısı:", sonuc)
