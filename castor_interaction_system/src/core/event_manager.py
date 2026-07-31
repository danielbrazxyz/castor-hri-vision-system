from typing import List, Dict, Optional
from src.core.entities import InteractionEvent

class EventTrackerState:
    def __init__(self, start_frame: int):
        self.start_frame = start_frame
        self.last_seen_frame = start_frame
        self.accumulated_data: List[float] = [] # Para calcular médias (ex: distância média)

class TemporalEventManager:
    def __init__(self, fps: float, tolerance_frames: int = 15):
        """
        fps: Frames por segundo do vídeo, para converter frames em segundos.
        tolerance_frames: Quantos frames o estado pode ficar ausente antes de fechar o evento.
        """
        self.fps = fps if fps > 0 else 30.0
        self.tolerance_frames = tolerance_frames
        
        # Estrutura: active_events[person_id][event_type] = EventTrackerState
        self.active_events: Dict[int, Dict[str, EventTrackerState]] = {}
        self.completed_events: List[InteractionEvent] = []

    def update_state(
        self, 
        current_frame: int, 
        person_id: int, 
        event_type: str, 
        is_active: bool, 
        value_to_track: float = 0.0
    ) -> None:
        """Atualiza o estado de um evento específico para uma pessoa no frame atual."""
        if person_id not in self.active_events:
            self.active_events[person_id] = {}
            
        person_events = self.active_events[person_id]
        
        if is_active:
            if event_type not in person_events:
                # Inicia um novo evento
                person_events[event_type] = EventTrackerState(start_frame=current_frame)
            
            # Atualiza o último frame visto e acumula dados
            state = person_events[event_type]
            state.last_seen_frame = current_frame
            state.accumulated_data.append(value_to_track)
        else:
            # Se não está ativo, checamos se existe um evento aberto que deve ser encerrado
            if event_type in person_events:
                state = person_events[event_type]
                frames_absent = current_frame - state.last_seen_frame
                
                if frames_absent > self.tolerance_frames:
                    self._close_event(person_id, event_type, state)
                    del person_events[event_type]

    def _close_event(self, person_id: int, event_type: str, state: EventTrackerState) -> None:
        """Finaliza um evento e o salva na lista de completados."""
        duration_frames = state.last_seen_frame - state.start_frame
        
        # Ignora ruídos extremamente curtos (ex: toques de 1 frame)
        if duration_frames < (self.fps * 0.2): # Mínimo de 0.2 segundos para ser um evento válido
            return
            
        duration_sec = duration_frames / self.fps
        avg_value = sum(state.accumulated_data) / max(1, len(state.accumulated_data))
        
        event = InteractionEvent(
            event_type=event_type,
            person_id=person_id,
            start_frame=state.start_frame,
            end_frame=state.last_seen_frame,
            duration_sec=round(duration_sec, 2),
            metadata={"mean_value": round(avg_value, 2)}
        )
        self.completed_events.append(event)

    def finalize_all_events(self) -> List[InteractionEvent]:
        """Chamado no fim do vídeo para fechar qualquer evento que ainda esteja aberto."""
        for person_id, events in list(self.active_events.items()):
            for event_type, state in list(events.items()):
                self._close_event(person_id, event_type, state)
                
        self.active_events.clear()
        return self.completed_events