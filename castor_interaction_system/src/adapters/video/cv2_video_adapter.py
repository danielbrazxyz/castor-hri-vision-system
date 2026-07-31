import cv2
from typing import Any, Tuple
from src.core.interfaces import IVideoSource

class OpenCVVideoAdapter(IVideoSource):
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"ERRO: Não foi possível abrir o vídeo no caminho: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self) -> Tuple[bool, Any]:
        return self.cap.read()

    def get_fps(self) -> float:
        return self.fps

    def get_total_frames(self) -> int:
        return self.total_frames

    def release(self) -> None:
        self.cap.release()