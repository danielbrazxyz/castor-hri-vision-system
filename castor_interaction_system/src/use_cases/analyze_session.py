from typing import List, Dict, Any
from tqdm import tqdm

from src.core.interfaces import (
    IVideoSource, IPersonDetector, ICastorDetector, IPoseEstimator, 
    IGazeEstimator, IDistanceEstimator, IPhysicalInteractionEngine,
    IDataExporter, IDashboardGenerator, IPDFGenerator
)
from src.core.entities import InteractionResult
from src.core.event_manager import TemporalEventManager
from src.core.metrics_engine import InteractionMetricsEngine

class AnalyzeSessionUseCase:
    def __init__(
        self,
        session_name: str,
        video_source: IVideoSource,
        person_detector: IPersonDetector,
        castor_detector: ICastorDetector,
        pose_estimator: IPoseEstimator,
        gaze_estimator: IGazeEstimator,
        distance_estimator: IDistanceEstimator,
        physical_engine: IPhysicalInteractionEngine,
        metrics_engine: InteractionMetricsEngine,
        event_manager: TemporalEventManager,
        data_exporter: IDataExporter,
        dashboard_generator: IDashboardGenerator,
        pdf_generator: IPDFGenerator
    ):
        self.session_name = session_name
        self.video_source = video_source
        self.person_detector = person_detector
        self.castor_detector = castor_detector
        self.pose_estimator = pose_estimator
        self.gaze_estimator = gaze_estimator
        self.distance_estimator = distance_estimator
        self.physical_engine = physical_engine
        self.metrics_engine = metrics_engine
        self.event_manager = event_manager
        
        self.data_exporter = data_exporter
        self.dashboard_generator = dashboard_generator
        self.pdf_generator = pdf_generator

    def execute(self) -> None:
        all_interactions: List[InteractionResult] = []
        frame_idx = 0
        total_frames = self.video_source.get_total_frames()
        fps = self.video_source.get_fps()

        print(f"Iniciando processamento da sessão: {self.session_name}")
        
        with tqdm(total=total_frames, desc="Processando Frames") as pbar:
            while True:
                ret, frame = self.video_source.read_frame()
                if not ret:
                    break

                # 1. Detecção Base
                people = self.person_detector.detect_and_track(frame)
                castor = self.castor_detector.detect(frame)

                if castor and people:
                    for person in people:
                        # 2. Extração Biomecânica e Visual
                        pose = self.pose_estimator.estimate_pose(frame, person.bbox)
                        
                        targets = {"CASTOR": castor.bbox}
                        gaze = self.gaze_estimator.estimate_attention(frame, person.bbox, targets)
                        
                        distance = self.distance_estimator.estimate_distance(person.bbox, castor.bbox)
                        physical = self.physical_engine.evaluate_interaction(pose, castor.bbox)

                        # 3. Registro Temporal (Eventos)
                        is_looking = gaze.targets_attention["CASTOR"].is_looking if gaze and "CASTOR" in gaze.targets_attention else False
                        self.event_manager.update_state(frame_idx, person.id, "GAZE_CASTOR", is_looking)
                        self.event_manager.update_state(frame_idx, person.id, "TOUCH", physical.is_touching if physical else False)
                        
                        if distance:
                            self.event_manager.update_state(frame_idx, person.id, f"ZONE_{distance.zone.name}", True, distance.distance_cm)

                        # 4. Cálculo do Score
                        # Pega o evento mais longo aberto para essa pessoa
                        continuous_time = 0.0
                        if person.id in self.event_manager.active_events and self.event_manager.active_events[person.id]:
                            longest_frames = max([frame_idx - state.start_frame for state in self.event_manager.active_events[person.id].values()])
                            continuous_time = longest_frames / fps

                        score_data = self.metrics_engine.calculate_frame_score(gaze, distance, physical, continuous_time)

                        # 5. Salvar Resultado do Frame
                        interaction = InteractionResult(
                            frame_idx=frame_idx,
                            person_id=person.id,
                            gaze_score=gaze.targets_attention["CASTOR"].confidence_score if gaze and "CASTOR" in gaze.targets_attention else 0.0,
                            distance_px=distance.distance_cm if distance else 0.0,
                            zone=distance.zone.value if distance else "desconhecida",
                            touch=physical.is_touching if physical else False,
                            interaction_score=score_data.total_score
                        )
                        # Adiciona dinamicamente para o exportador CSV ler
                        interaction.interaction_level = score_data.level.value 
                        all_interactions.append(interaction)

                frame_idx += 1
                pbar.update(1)

        self.video_source.release()
        
        # 6. Finalização e Relatórios
        completed_events = self.event_manager.finalize_all_events()
        metadata = {
            "total_frames": total_frames,
            "fps": fps,
            "total_interacoes": len(all_interactions),
            "eventos_concluidos": len(completed_events)
        }

        print("\nGerando relatórios científicos...")
        self.dashboard_generator.generate(all_interactions, completed_events, fps, f"outputs/dashboard_{self.session_name}.png")
        self.data_exporter.export_csv(all_interactions, f"outputs/interactions_{self.session_name}.csv")
        self.data_exporter.export_json(completed_events, metadata, f"outputs/analysis_{self.session_name}.json")
        self.pdf_generator.generate_report(
            self.session_name, metadata, f"outputs/dashboard_{self.session_name}.png", f"outputs/report_{self.session_name}.pdf"
        )
        print("Processamento concluído com sucesso!")