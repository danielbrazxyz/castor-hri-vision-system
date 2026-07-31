from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


# ==========================================
# ENUMS
# ==========================================

class ProxemicZone(Enum):
    INTIMATE = "intima"
    PERSONAL = "pessoal"
    SOCIAL = "social"
    PUBLIC = "publica"
    UNKNOWN = "desconhecida"


class InteractionLevel(Enum):
    LOW = "baixa_interacao"
    MILD = "interacao_leve"
    MODERATE = "interacao_moderada"
    HIGH = "interacao_alta"
    INTENSE = "interacao_intensa"


# ==========================================
# GEOMETRIA E DETECÇÃO BASE
# ==========================================

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class PersonData:
    id: int
    bbox: BBox
    confidence: float


@dataclass
class CastorData:
    bbox: BBox
    confidence: float


# ==========================================
# BIOMECÂNICA E POSE
# ==========================================

@dataclass
class Keypoint:
    x: float          # Coordenada absoluta X na imagem
    y: float          # Coordenada absoluta Y na imagem
    visibility: float # Confiança (0.0 a 1.0)


@dataclass
class PoseData:
    landmarks: Dict[str, Keypoint]              # Mapeamento semântico (ex: 'left_wrist')
    body_orientation_vector: Tuple[float, float] # Vetor 2D (x,y) do peito
    head_orientation_vector: Tuple[float, float] # Vetor 2D (x,y) da face

    def get_hands(self) -> Tuple[Optional[Keypoint], Optional[Keypoint]]:
        """Retorna os keypoints dos pulsos/mãos esquerda e direita."""
        return self.landmarks.get('left_wrist'), self.landmarks.get('right_wrist')


# ==========================================
# ATENÇÃO VISUAL (GAZE)
# ==========================================

@dataclass
class AttentionTarget:
    target_id: str          # Ex: 'CASTOR', 'TERAPEUTA_1'
    is_looking: bool        # Booleano se está olhando
    confidence_score: float # Grau de ativação no heatmap (0.0 a 1.0)


@dataclass
class GazeData:
    heatmap: Any                                  # Mapa 2D bruto de probabilidades
    targets_attention: Dict[str, AttentionTarget] # Mapeia ID do alvo para objeto AttentionTarget


# ==========================================
# DISTÂNCIA E INTERAÇÃO FÍSICA
# ==========================================

@dataclass
class DistanceData:
    distance_cm: float
    distance_meters: float
    zone: ProxemicZone
    real_world_coords_person: Tuple[float, float] # (X, Y) no chão
    real_world_coords_castor: Tuple[float, float] # (X, Y) no chão


@dataclass
class PhysicalInteractionData:
    is_touching: bool
    is_reaching: bool               # Intenção de toque (alcance)
    closest_hand_distance_px: float
    active_hand: str                # 'left', 'right', 'both' ou 'none'


# ==========================================
# MÉTRICAS E RESULTADOS DA SESSÃO
# ==========================================

@dataclass
class ScoreData:
    total_score: float
    level: InteractionLevel
    visual_component: float
    proxemic_component: float
    touch_component: float
    persistence_component: float


@dataclass
class InteractionResult:
    frame_idx: int
    person_id: int
    gaze_score: float
    distance_px: float
    zone: str
    touch: bool
    interaction_score: float


@dataclass
class InteractionEvent:
    event_type: str          # Ex: 'TOUCH', 'GAZE_CASTOR', 'ZONE_INTIMATE'
    person_id: int
    start_frame: int
    end_frame: int
    duration_sec: float
    metadata: Dict[str, Any] # Ex: média de score, distância média


@dataclass
class FrameContext:
    frame_idx: int
    image_data: Any
    people: List[PersonData]
    castor: Optional[CastorData]
    interactions: List[InteractionResult]