import cv2
import numpy as np
from typing import Optional, List
from src.core.entities import BBox, DistanceData, ProxemicZone
from src.core.interfaces import IDistanceEstimator
from src.infrastructure.config_manager import ProxemicsConfig

class HomographyDistanceEstimator(IDistanceEstimator):
    def __init__(self, config: ProxemicsConfig, src_pts: List[List[float]], dst_pts: List[List[float]]):
        """
        Inicializa o estimador calculando a Matriz de Homografia.
        src_pts: 4 pontos [x,y] na imagem (pixels) forming um polígono no chão.
        dst_pts: 4 pontos [x,y] reais correspondentes (em centímetros).
        """
        self.config = config
        
        # Garante o formato correto do Numpy para o OpenCV
        pts_src = np.array(src_pts, dtype=np.float32)
        pts_dst = np.array(dst_pts, dtype=np.float32)
        
        if len(pts_src) == 4 and len(pts_dst) == 4:
            self.homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)
        else:
            self.homography_matrix = None

    def _get_bottom_center(self, bbox: BBox) -> np.ndarray:
        """Retorna o ponto central da base do BBox (ancoragem no chão)."""
        x_center = (bbox.x1 + bbox.x2) / 2.0
        y_bottom = float(bbox.y2) # Y cresce para baixo na imagem
        return np.array([[[x_center, y_bottom]]], dtype=np.float32)

    def _classify_zone(self, distance_cm: float) -> ProxemicZone:
        if distance_cm <= self.config.intimate_cm:
            return ProxemicZone.INTIMATE
        elif distance_cm <= self.config.personal_cm:
            return ProxemicZone.PERSONAL
        elif distance_cm <= self.config.social_cm:
            return ProxemicZone.SOCIAL
        else:
            return ProxemicZone.PUBLIC

    def estimate_distance(self, person_bbox: BBox, castor_bbox: BBox) -> Optional[DistanceData]:
        if self.homography_matrix is None:
            # Fallback seguro caso não haja calibração de câmera
            return None
            
        # Pega a âncora no chão da pessoa e do robô
        person_anchor = self._get_bottom_center(person_bbox)
        castor_anchor = self._get_bottom_center(castor_bbox)
        
        # Transforma os pontos da perspectiva da imagem para o plano 2D real (Bird's-Eye View)
        person_real = cv2.perspectiveTransform(person_anchor, self.homography_matrix)
        castor_real = cv2.perspectiveTransform(castor_anchor, self.homography_matrix)
        
        px, py = person_real[0][0]
        cx, cy = castor_real[0][0]
        
        # Distância Euclidiana no plano real (em centímetros)
        distance_cm = float(np.sqrt((px - cx)**2 + (py - cy)**2))
        
        return DistanceData(
            distance_cm=distance_cm,
            distance_meters=distance_cm / 100.0,
            zone=self._classify_zone(distance_cm),
            real_world_coords_person=(px, py),
            real_world_coords_castor=(cx, cy)
        )