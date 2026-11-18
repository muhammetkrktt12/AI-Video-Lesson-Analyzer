"""
EduFrame AI - LangGraph Orkestratörü (Root Wrapper)
Bu dosya kök dizindeki çağrılar ve geriye dönük uyumluluk için
core.eduframe_agentic_core modülünü sarmalar.
"""

from core.eduframe_agentic_core import EduFrameAgenticCore, create_eduframe_graph
from core.state import EduFrameState

__all__ = ["EduFrameAgenticCore", "create_eduframe_graph", "EduFrameState"]

if __name__ == "__main__":
    core = EduFrameAgenticCore()
    print("[EduFrameAgenticCore Wrapper] Hazır.")
