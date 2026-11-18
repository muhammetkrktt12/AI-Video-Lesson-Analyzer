import requests
import json

prompt = """Sen Türkiye MEB müfredatına hakim, 3. sınıf ilkokul seviyesi için ders fasikülü hazırlayan uzman bir öğretmensin.

ÖĞRETMENİN VİDEODAKİ ANLATIMI VE ÖRNEKLERİ:
"Ekranda gördüğümüz araçların ortak özelliği nedir bir düşünelim: Ütü elbiselerimizin kırışıklıklarını açar. Buzdolabı besinlerimizin bozulmamasını sağlar. Çamaşır makinesi kirli kıyafetlerimizi yıkar. Bilgisayarı araştırma yapmak ve bilgi depolamak için kullanırız. Klima sıcak havalarda serinletir, soğukta ısıtır. Elektrikli süpürge ise evdeki toz ve kirleri temizler. Hepsinin yaptığı iş farklı olsa da en temel ortak özellikleri elektrik enerjisiyle çalışmalarıdır!"

MEB DERS KİTABI BAĞLAMI:
"Elektrikli Araç ve Gereçler: Evlerimizde, okullarımızda ve çevremizde elektrik enerjisiyle çalışan birçok araç gereç vardır. Bu araçlar aydınlatma, ısıtma, soğutma, temizlik ve haberleşme gibi amaçlarla kullanılır. Elektrik yaşamımızı büyük ölçüde kolaylaştırır."

GÖREV:
Yukarıdaki öğretmenin verdiği TÜM SOMUT ÖRNEKLERİ (ütü, buzdolabı, çamaşır makinesi, bilgisayar, klima, elektrikli süpürge) ve ders kitabındaki bilgileri birleştirerek; 3. sınıf öğrencisinin (8-9 yaş) zevkle okuyup kolayca anlayacağı, açıklayıcı, akıcı ve pedagojik bir konu anlatımı metni yaz (2-3 zengin paragraf). Asla 1-2 cümlelik kısa özet yapma, konuyu çocuğa somutlaştırarak öğret.
Düşünme zinciri veya etiket ekleme, sadece doğrudan çocuğa hitap eden fasikül konu anlatım metnini ver.
"""

payload = {
    "model": "qwen3.5:4b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.3
    }
}

res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
print("STATUS:", res.status_code)
print("RESPONSE:\n", res.json().get("response"))
