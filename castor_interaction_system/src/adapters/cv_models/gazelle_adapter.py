import torch
import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional
import sys

# Assume-se que a biblioteca externa 'gazelle' esteja acessível no PYTHONPATH
from gazelle.model import get_gazelle_model
from src.core.entities import BBox, GazeData, AttentionTarget
from src.core.interfaces import IGazeEstimator
from src.infrastructure.config_manager import HardwareConfig, ModelConfig

class GazelleAdapter(IGazeEstimator):
    def __init__(self, hardware_cfg: HardwareConfig, model_cfg: ModelConfig):
        self.device = hardware_cfg.device
        self.threshold = 0.25
        
        try:
            self.model, self.transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
            
            # Carregamento seguro dos pesos com tipagem correta
            state_dict = torch.load(
                model_cfg.gazelle_weights,
                map_location=self.device,
                weights_only=True
            )
            
            self.model.load_gazelle_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Otimização FP16 se solicitado
            if hardware_cfg.half_precision and self.device == "cuda":
                self.model.half()
                
        except Exception as e:
            print(f"Falha crítica ao inicializar modelo Gazelle: {e}", file=sys.stderr)
            sys.exit(1)

    @torch.no_grad()
    def _generate_heatmap(self, frame: np.ndarray, face_bbox: BBox) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Normalização do BBox para o modelo [0 a 1]
        norm_bbox = (
            max(0.0, face_bbox.x1 / w),
            max(0.0, face_bbox.y1 / h),
            min(1.0, face_bbox.x2 / w),
            min(1.0, face_bbox.y2 / h)
        )

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Otimização de tensor alinhada à configuração de hardware
        tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        if next(self.model.parameters()).dtype == torch.float16:
            tensor_image = tensor_image.half()

        input_data = {
            "images": tensor_image,
            "bboxes": [[norm_bbox]]
        }

        output = self.model(input_data)
        heatmap = output["heatmap"][0][0].cpu().numpy()
        return heatmap

    def estimate_attention(
        self, 
        image: Any, 
        face_bbox: BBox, 
        candidate_targets: Dict[str, BBox]
    ) -> Optional[GazeData]:
        
        try:
            heatmap = self._generate_heatmap(image, face_bbox)
        except Exception:
            return None # Proteção contra falhas de inferência isoladas

        heat_h, heat_w = heatmap.shape
        img_h, img_w = image.shape[:2]
        
        targets_result = {}

        for target_id, t_bbox in candidate_targets.items():
            # Mapeamento do BBox alvo (resolução da câmera) para o espaço do Heatmap (resolução Gazelle)
            hx1 = max(0, int((t_bbox.x1 / img_w) * heat_w))
            hy1 = max(0, int((t_bbox.y1 / img_h) * heat_h))
            hx2 = min(heat_w - 1, int((t_bbox.x2 / img_w) * heat_w))
            hy2 = min(heat_h - 1, int((t_bbox.y2 / img_h) * heat_h))

            region = heatmap[hy1:hy2, hx1:hx2]
            
            if region.size == 0:
                targets_result[target_id] = AttentionTarget(target_id, False, 0.0)
                continue

            score = float(region.max())
            is_looking = score >= self.threshold

            targets_result[target_id] = AttentionTarget(
                target_id=target_id,
                is_looking=is_looking,
                confidence_score=score
            )

        return GazeData(heatmap=heatmap, targets_attention=targets_result)