import requests
import json
import time

prompt = """3. sınıf fen bilgisi ders fasikülü için konu anlatımı:

ÖĞRETMENİN ÖRNEKLERİ:
Ütü elbiselerin kırışıklığını açar, buzdolabı besinleri korur, çamaşır makinesi kıyafetleri yıkar, bilgisayar araştırma yapar, klima serinletir/ısıtır, elektrikli süpürge tozları temizler. Ortak özellikleri elektrik enerjisiyle çalışmalarıdır.

DERS KİTABI:
Elektrikli Araç ve Gereçler aydınlatma, ısıtma, soğutma ve temizlik için kullanılır.

GÖREV:
Yukarıdaki tüm örnekleri içeren, 3. sınıf öğrencisinin anlayacağı 2 paragraflık sade ve zengin bir ders notu yaz.
"""

payload = {
    "model": "qwen3.5:4b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "num_thread": 16,
        "num_predict": 250,
        "temperature": 0.2
    }
}

t0 = time.time()
try:
    res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=90)
    print("STATUS:", res.status_code, "TIME:", round(time.time() - t0, 1), "s")
    print("RESPONSE:\n", res.json().get("response"))
except Exception as e:
    print("ERROR:", e)
