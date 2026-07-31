from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from src.core.entities import PersonData, BBox

@dataclass
class TrackHistory:
    last_known_bbox: BBox
    frames_since_seen: int = 0
    
class IDRecoveryManager:
    """
    Camada de inteligência para recuperação de IDs perdidos por oclusão.
    Prepara a estrutura para futura inserção de DeepSORT/ReID.
    """
    def __init__(self, max_lost_frames: int = 30, iou_threshold: float = 0.3):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.active_tracks: Dict[int, TrackHistory] = {}
        self.id_remap: Dict[int, int] = {} # Mapeia IDs novos (ruídos) para IDs reais

    def _calculate_iou(self, boxA: BBox, boxB: BBox) -> float:
        xA = max(boxA.x1, boxB.x1)
        yA = max(boxA.y1, boxB.y1)
        xB = min(boxA.x2, boxB.x2)
        yB = min(boxA.y2, boxB.y2)

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
        boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def process_frame(self, current_people: List[PersonData]) -> List[PersonData]:
        # Atualiza idade dos tracks armazenados
        for tid in list(self.active_tracks.keys()):
            self.active_tracks[tid].frames_since_seen += 1
            if self.active_tracks[tid].frames_since_seen > self.max_lost_frames:
                del self.active_tracks[tid]

        recovered_people = []

        for person in current_people:
            # Se o ID foi remapeado anteriormente, aplicamos o mapeamento
            original_id = person.id
            if original_id in self.id_remap:
                person.id = self.id_remap[original_id]
                
            # Verifica se é um ID potencialmente novo no nosso sistema
            if person.id not in self.active_tracks:
                best_iou = 0.0
                best_match_id = -1
                
                # Tenta casar o novo ID com algum ID recém-perdido
                for lost_id, history in self.active_tracks.items():
                    if history.frames_since_seen > 0:
                        iou = self._calculate_iou(person.bbox, history.last_known_bbox)
                        if iou > best_iou and iou > self.iou_threshold:
                            best_iou = iou
                            best_match_id = lost_id
                
                # Se encontrou um match plausível, recuperamos o ID antigo
                if best_match_id != -1:
                    self.id_remap[original_id] = best_match_id
                    person.id = best_match_id

            # Atualiza o histórico com a nova posição
            self.active_tracks[person.id] = TrackHistory(last_known_bbox=person.bbox)
            recovered_people.append(person)

        return recovered_people