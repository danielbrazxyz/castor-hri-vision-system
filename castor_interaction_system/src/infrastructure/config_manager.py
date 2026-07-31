from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import yaml
from pathlib import Path
import sys

class HardwareConfig(BaseModel):
    device: str = Field(default="cpu", pattern="^(cpu|cuda)$", description="Dispositivo de processamento")
    half_precision: bool = Field(default=False, description="Usa FP16 na GPU para otimização")

class ModelConfig(BaseModel):
    yolo_person: str = Field(default="yolov8n.pt")
    yolo_castor: str = Field(default="models/castor.pt")
    yolo_hand: str = Field(default="models/hand_yolov8n.pt")
    gazelle_weights: str = Field(default="weights/gazelle_vitl.pt")

class ProxemicsConfig(BaseModel):
    intimate_cm: int = Field(default=45, ge=0)
    personal_cm: int = Field(default=120, gt=45)
    social_cm: int = Field(default=360, gt=120)

class AppConfig(BaseModel):
    session_name: str = Field(..., min_length=1, description="Nome único da sessão")
    video_path: str = Field(..., description="Caminho para o vídeo de entrada")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    proxemics: ProxemicsConfig = Field(default_factory=ProxemicsConfig)

class ConfigLoader:
    @staticmethod
    def load_yaml(yaml_path: str) -> AppConfig:
        """Carrega e valida a configuração a partir de um arquivo YAML."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {yaml_path}")

        with open(path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)

        try:
            # Pydantic faz toda a validação automaticamente aqui
            config = AppConfig(**raw_data)
            return config
        except ValidationError as e:
            print("Erro de validação no arquivo de configuração!", file=sys.stderr)
            print(e.json(indent=2), file=sys.stderr)
            sys.exit(1)