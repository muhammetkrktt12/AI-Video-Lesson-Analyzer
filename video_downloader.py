"""
EduFrame AI - Video Downloader (Root Compatibility Wrapper)
Bu dosya kök dizindeki çağrılar ve geriye dönük uyumluluk için
services.video_downloader modülünü sarmalar.
"""

from services.video_downloader import VideoDownloader, VideoDownloader as VideoMotoru

__all__ = ["VideoDownloader", "VideoMotoru"]

if __name__ == "__main__":
    motor = VideoDownloader()
    print("[VideoDownloader Wrapper] Hazır.")
