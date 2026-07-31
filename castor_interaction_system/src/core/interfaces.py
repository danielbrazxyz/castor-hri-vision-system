from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from src.core.entities import (
    BBox,
    PersonData,
    CastorData,
    PoseData,
    GazeData,
    DistanceData,
    PhysicalInteractionData,
    InteractionResult,
    InteractionEvent
)


# ==========================================
# INTERFACES DE ENTRADA E CAPTURA
# ==========================================

class IVideoSource(ABC):
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Any]: 
        pass

    @abstractmethod
    def get_fps(self) -> float: 
        pass

    @abstractmethod
    def get_total_frames(self) -> int: 
        pass

    @abstractmethod
    def release(self) -> None: 
        pass


# ==========================================
# INTERFACES DE DETECÇÃO E IA
# ==========================================

class IPersonDetector(ABC):
    @abstractmethod
    def detect_and_track(self, image: Any) -> List[PersonData]: 
        pass


class ICastorDetector(ABC):
    @abstractmethod
    def detect(self, image: Any) -> Optional[CastorData]: 
        pass


class IPoseEstimator(ABC):
    @abstractmethod
    def estimate_pose(self, image: Any, person_bbox: BBox) -> Optional[PoseData]:
        """Extrai a pose apenas da região da pessoa delimitada pelo BBox."""
        pass


class IGazeEstimator(ABC):
    @abstractmethod
    def estimate_attention(
        self, 
        image: Any, 
        face_bbox: BBox, 
        candidate_targets: Dict[str, BBox]
    ) -> Optional[GazeData]:
        """Estima a atenção visual partindo do rosto em direção a múltiplos alvos."""
        pass


# ==========================================
# INTERFACES DE MÉTRICAS E ANÁLISE
# ==========================================

class IDistanceEstimator(ABC):
    @abstractmethod
    def estimate_distance(self, person_bbox: BBox, castor_bbox: BBox) -> Optional[DistanceData]:
        """Estima a distância real no plano físico entre a pessoa e o CASTOR."""
        pass


class IPhysicalInteractionEngine(ABC):
    @abstractmethod
    def evaluate_interaction(self, pose: Optional[PoseData], castor_bbox: BBox) -> PhysicalInteractionData:
        """Avalia se a pessoa está tocando ou tentando tocar o robô com base na pose."""
        pass


class IInteractionEngine(ABC):
    @abstractmethod
    def evaluate(self, image: Any, person: PersonData, castor: CastorData) -> InteractionResult: 
        pass


# ==========================================
# INTERFACES DE SAÍDA E RELATÓRIOS
# ==========================================

class IOutputAdapter(ABC):
    @abstractmethod
    def save_metrics(self, results: List[InteractionResult]) -> None: 
        pass


class IDashboardGenerator(ABC):
    @abstractmethod
    def generate(
        self, 
        frame_results: List[InteractionResult], 
        events: List[InteractionEvent], 
        fps: float, 
        output_path: str
    ) -> None:
        """Gera e salva o dashboard científico em formato de imagem."""
        pass


class IDataExporter(ABC):
    @abstractmethod
    def export_csv(self, frame_results: List[InteractionResult], output_path: str) -> None:
        pass

    @abstractmethod
    def export_json(self, events: List[InteractionEvent], metadata: Dict[str, Any], output_path: str) -> None:
        pass


class IPDFGenerator(ABC):
    @abstractmethod
    def generate_report(
        self, 
        session_name: str, 
        metadata: Dict[str, Any], 
        dashboard_image_path: str, 
        output_path: str
    ) -> None:
        pass