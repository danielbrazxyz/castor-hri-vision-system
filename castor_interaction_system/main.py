import os
from src.infrastructure.config_manager import ConfigLoader
from src.adapters.video.cv2_video_adapter import OpenCVVideoAdapter
from src.adapters.cv_models.yolo_adapters import YoloPersonDetectorAdapter, YoloCastorDetectorAdapter
from src.adapters.cv_models.mediapipe_adapters import MediaPipePoseAdapter
from src.adapters.cv_models.gazelle_adapter import GazelleAdapter
from src.adapters.tracking.track_manager import RobustPersonTracker
from src.adapters.metrics.homography_distance import HomographyDistanceEstimator
from src.adapters.metrics.physical_interaction import PhysicalInteractionEngine
from src.adapters.io.dashboard_generator import MatplotlibDashboardGenerator
from src.adapters.io.data_exporter import DataExporterAdapter
from src.adapters.io.pdf_generator import PDFReportAdapter

from src.core.metrics_engine import InteractionMetricsEngine
from src.core.event_manager import TemporalEventManager
from src.use_cases.analyze_session import AnalyzeSessionUseCase
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ... restante do seu código (importações e função main) ...
def main():
    # 1. Carrega as configurações (Certifique-se de ter um arquivo configs/default.yaml válido)
    config = ConfigLoader.load_yaml("configs/default.yaml")
    
    # Cria pasta de outputs se não existir
    os.makedirs("outputs", exist_ok=True)

    # 2. Inicializa os Adaptadores (Infraestrutura)
    video_source = OpenCVVideoAdapter(config.video_path)
    
    # Detecção e Rastreamento
    base_person_detector = YoloPersonDetectorAdapter(config.models.yolo_person, config.hardware)
    robust_tracker = RobustPersonTracker(base_detector=base_person_detector, max_lost_frames=30)
    castor_detector = YoloCastorDetectorAdapter(config.models.yolo_castor, config.hardware)
    
    # Biomecânica e Visual
    pose_estimator = MediaPipePoseAdapter(config.hardware)
    gaze_estimator = GazelleAdapter(config.hardware, config.models)
    
    # Métricas Geométricas
    # Mocking de pontos de calibração do chão (Adapte para o seu yaml no futuro)
    src_pts = [[0, 0], [100, 0], [100, 100], [0, 100]] 
    dst_pts = [[0, 0], [100, 0], [100, 100], [0, 100]] 
    distance_estimator = HomographyDistanceEstimator(config.proxemics, src_pts, dst_pts)
    
    physical_engine = PhysicalInteractionEngine(intention_threshold_px=50.0)

    # Exportadores
    data_exporter = DataExporterAdapter()
    dashboard_generator = MatplotlibDashboardGenerator()
    pdf_generator = PDFReportAdapter()

    # 3. Inicializa o Core (Regras de Negócio)
    metrics_engine = InteractionMetricsEngine()
    event_manager = TemporalEventManager(fps=video_source.get_fps(), tolerance_frames=15)

    # 4. Injeta tudo no Caso de Uso e Executa
    use_case = AnalyzeSessionUseCase(
        session_name=config.session_name,
        video_source=video_source,
        person_detector=robust_tracker,
        castor_detector=castor_detector,
        pose_estimator=pose_estimator,
        gaze_estimator=gaze_estimator,
        distance_estimator=distance_estimator,
        physical_engine=physical_engine,
        metrics_engine=metrics_engine,
        event_manager=event_manager,
        data_exporter=data_exporter,
        dashboard_generator=dashboard_generator,
        pdf_generator=pdf_generator
    )

    use_case.execute()

if __name__ == "__main__":
    main()