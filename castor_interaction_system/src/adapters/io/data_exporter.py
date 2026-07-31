import csv
import json
from dataclasses import asdict
from typing import List, Dict, Any

from src.core.interfaces import IDataExporter
from src.core.entities import InteractionResult, InteractionEvent

class DataExporterAdapter(IDataExporter):
    def export_csv(self, frame_results: List[InteractionResult], output_path: str) -> None:
        if not frame_results:
            return

        # Achatando (flattening) a estrutura de dados para o formato tabular do CSV
        fieldnames = [
            'frame_idx', 'person_id', 'gaze_score', 'distance_px', 
            'zone', 'touch', 'interaction_score', 'interaction_level'
        ]

        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in frame_results:
                writer.writerow({
                    'frame_idx': result.frame_idx,
                    'person_id': result.person_id,
                    'gaze_score': result.gaze_score,
                    'distance_px': round(result.distance_px, 2),
                    'zone': result.zone,
                    'touch': result.touch,
                    'interaction_score': result.interaction_score,
                    # O level é extraído da entidade ScoreData (adicionada na Etapa 10)
                    'interaction_level': getattr(result, 'interaction_level', 'UNKNOWN')
                })

    def export_json(self, events: List[InteractionEvent], metadata: Dict[str, Any], output_path: str) -> None:
        # Converte as entidades de domínio para dicionários nativos do Python
        serialized_events = [asdict(event) for event in events]
        
        output_data = {
            "session_metadata": metadata,
            "interaction_events": serialized_events
        }

        with open(output_path, mode='w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)