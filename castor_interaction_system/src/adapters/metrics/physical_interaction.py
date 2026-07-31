import math
from typing import Optional
from src.core.entities import BBox, PoseData, PhysicalInteractionData, Keypoint
from src.core.interfaces import IPhysicalInteractionEngine

class PhysicalInteractionEngine(IPhysicalInteractionEngine):
    def __init__(self, intention_threshold_px: float = 50.0):
        """
        intention_threshold_px: Margem em pixels em torno do robô considerada como "área de alcance/intenção".
        """
        self.intention_threshold_px = intention_threshold_px

    def _point_to_bbox_distance(self, pt: Keypoint, bbox: BBox) -> float:
        """
        Calcula a menor distância euclidiana de um ponto (Keypoint) 
        até as bordas de um retângulo (BBox). Retorna 0.0 se o ponto estiver dentro.
        """
        dx = max(bbox.x1 - pt.x, 0, pt.x - bbox.x2)
        dy = max(bbox.y1 - pt.y, 0, pt.y - bbox.y2)
        
        # Fórmula Euclidiana: $d = \sqrt{dx^2 + dy^2}$
        return math.sqrt(dx**2 + dy**2)

    def evaluate_interaction(self, pose: Optional[PoseData], castor_bbox: BBox) -> PhysicalInteractionData:
        # Se não há dados de pose (pessoa ocluída ou longe), não há interação física computável
        if pose is None:
            return PhysicalInteractionData(False, False, float('inf'), 'none')

        left_wrist, right_wrist = pose.get_hands()
        
        dist_left = float('inf')
        dist_right = float('inf')

        if left_wrist and left_wrist.visibility > 0.5:
            dist_left = self._point_to_bbox_distance(left_wrist, castor_bbox)
            
        if right_wrist and right_wrist.visibility > 0.5:
            dist_right = self._point_to_bbox_distance(right_wrist, castor_bbox)

        min_dist = min(dist_left, dist_right)
        
        # Lógica de Classificação
        is_touching = min_dist == 0.0
        is_reaching = (0.0 < min_dist <= self.intention_threshold_px)
        
        # Determina qual mão está agindo
        active_hand = 'none'
        if min_dist < float('inf'):
            if dist_left == dist_right == min_dist:
                active_hand = 'both'
            elif dist_left == min_dist:
                active_hand = 'left'
            else:
                active_hand = 'right'

        return PhysicalInteractionData(
            is_touching=is_touching,
            is_reaching=is_reaching,
            closest_hand_distance_px=min_dist,
            active_hand=active_hand
        )