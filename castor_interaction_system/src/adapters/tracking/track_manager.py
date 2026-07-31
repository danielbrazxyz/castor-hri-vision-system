from typing import List, Any
from src.core.interfaces import IPersonDetector
from src.core.entities import PersonData
from src.core.tracking_logic import IDRecoveryManager

class RobustPersonTracker(IPersonDetector):
    """
    Decorator/Wrapper pattern. Envolve o detector básico com a lógica de tracking temporal.
    """
    def __init__(self, base_detector: IPersonDetector, max_lost_frames: int = 30):
        self.base_detector = base_detector
        self.recovery_manager = IDRecoveryManager(max_lost_frames=max_lost_frames)

    def detect_and_track(self, image: Any) -> List[PersonData]:
        # 1. Executa a detecção e o rastreamento primário (ex: YOLO + ByteTrack)
        raw_detections = self.base_detector.detect_and_track(image)
        
        # 2. Aplica a camada científica de recuperação de oclusões (Filtro temporal)
        refined_detections = self.recovery_manager.process_frame(raw_detections)
        
        return refined_detections