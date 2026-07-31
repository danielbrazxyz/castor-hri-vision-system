import sys
from typing import List, Optional, Any, Dict
from ultralytics import YOLO

from src.core.entities import BBox, PersonData, CastorData
from src.core.interfaces import IPersonDetector, ICastorDetector
from src.infrastructure.config_manager import HardwareConfig

class YoloModelRegistry:
    """
    Registry pattern para garantir que modelos YOLO sejam carregados apenas uma vez
    e compartilhem o mesmo dispositivo de hardware.
    """
    _instances: Dict[str, YOLO] = {}

    @classmethod
    def get_model(cls, model_path: str, hardware_cfg: HardwareConfig) -> YOLO:
        if model_path not in cls._instances:
            try:
                model = YOLO(model_path)
                # Configuração inicial do hardware (warmup não incluso aqui para brevidade)
                cls._instances[model_path] = model
            except Exception as e:
                print(f"Erro ao carregar o modelo YOLO: {model_path}", file=sys.stderr)
                print(str(e), file=sys.stderr)
                sys.exit(1)
        return cls._instances[model_path]

class YoloPersonDetectorAdapter(IPersonDetector):
    def __init__(self, model_path: str, hardware_cfg: HardwareConfig):
        self.hardware_cfg = hardware_cfg
        self.model = YoloModelRegistry.get_model(model_path, hardware_cfg)

    def detect_and_track(self, image: Any) -> List[PersonData]:
        # Executa inferência com rastreamento nativo
        # O ByteTrack ou Bot-SORT nativo do YOLO será usado provisoriamente
        # até implementarmos a Etapa 04.
        results = self.model.track(
            image,
            persist=True,
            classes=[0], # 0 = pessoa (COCO)
            device=self.hardware_cfg.device,
            half=self.hardware_cfg.half_precision,
            verbose=False
        )
        
        people_list: List[PersonData] = []
        
        if not results or len(results) == 0:
            return people_list
            
        boxes = results[0].boxes
        if boxes is None:
            return people_list
            
        for box in boxes:
            # Pessoas sem ID de rastreamento são ignoradas
            if box.id is None:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            track_id = int(box.id[0].item())
            
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            person = PersonData(id=track_id, bbox=bbox, confidence=conf)
            people_list.append(person)
            
        return people_list


class YoloCastorDetectorAdapter(ICastorDetector):
    def __init__(self, model_path: str, hardware_cfg: HardwareConfig):
        self.hardware_cfg = hardware_cfg
        self.model = YoloModelRegistry.get_model(model_path, hardware_cfg)

    def detect(self, image: Any) -> Optional[CastorData]:
        results = self.model(
            image,
            device=self.hardware_cfg.device,
            half=self.hardware_cfg.half_precision,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return None
            
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
            
        # Procura especificamente pela classe 'robo_castor'
        # Assumindo que no modelo treinado a classe pode ter um ID específico
        # Verificamos pelo nome mapeado no modelo
        best_castor = None
        highest_conf = -1.0
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = self.model.names[cls_id]
            
            if label == 'robo_castor':
                conf = float(box.conf[0].item())
                if conf > highest_conf: # Pega a detecção mais confiável se houver falsos positivos
                    highest_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    best_castor = CastorData(bbox=bbox, confidence=conf)
                    
        return best_castor