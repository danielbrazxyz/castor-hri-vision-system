from typing import Optional
from src.core.entities import GazeData, DistanceData, PhysicalInteractionData, ScoreData, InteractionLevel, ProxemicZone

class InteractionMetricsEngine:
    def __init__(self):
        # Pesos definidos cientificamente pela formulação do framework
        self.weight_visual = 0.35
        self.weight_proxemic = 0.30
        self.weight_touch = 0.25
        self.weight_persistence = 0.10
        
        # Mapeamento de score proxêmico (0 a 100) baseado na zona
        self.proxemic_scores = {
            ProxemicZone.INTIMATE: 100.0,
            ProxemicZone.PERSONAL: 70.0,
            ProxemicZone.SOCIAL: 30.0,
            ProxemicZone.PUBLIC: 0.0,
            ProxemicZone.UNKNOWN: 0.0
        }

    def _classify_level(self, score: float) -> InteractionLevel:
        """Classifica o score em categorias qualitativas."""
        if score < 20.0:
            return InteractionLevel.LOW
        elif score < 40.0:
            return InteractionLevel.MILD
        elif score < 60.0:
            return InteractionLevel.MODERATE
        elif score < 80.0:
            return InteractionLevel.HIGH
        else:
            return InteractionLevel.INTENSE

    def calculate_frame_score(
        self,
        gaze: Optional[GazeData],
        distance: Optional[DistanceData],
        physical: Optional[PhysicalInteractionData],
        continuous_interaction_seconds: float = 0.0
    ) -> ScoreData:
        """
        Calcula o índice global de interação no instante atual (frame).
        """
        
        # 1. Componente Visual (0 a 100)
        visual_score = 0.0
        if gaze is not None and 'CASTOR' in gaze.targets_attention:
            castor_gaze = gaze.targets_attention['CASTOR']
            if castor_gaze.is_looking:
                # Usa o confidence score do heatmap para graduar o foco (ex: 0.8 de confiança = 80 pontos)
                visual_score = min(castor_gaze.confidence_score * 100.0, 100.0)

        # 2. Componente Proxêmica (0 a 100)
        proxemic_score = 0.0
        if distance is not None:
            proxemic_score = self.proxemic_scores.get(distance.zone, 0.0)

        # 3. Componente Física/Touch (0 a 100)
        touch_score = 0.0
        if physical is not None:
            if physical.is_touching:
                touch_score = 100.0
            elif physical.is_reaching:
                # Intenção de toque recebe pontuação parcial
                touch_score = 50.0

        # 4. Componente de Persistência (0 a 100)
        # Bônus temporal: atinge o máximo após 10 segundos de engajamento contínuo
        persistence_score = min((continuous_interaction_seconds / 10.0) * 100.0, 100.0)

        # Cálculo do Score Multimodal Final
        total_score = (
            (visual_score * self.weight_visual) +
            (proxemic_score * self.weight_proxemic) +
            (touch_score * self.weight_touch) +
            (persistence_score * self.weight_persistence)
        )
        
        # Limita o score entre 0 e 100 e arredonda
        total_score = round(max(0.0, min(total_score, 100.0)), 2)

        return ScoreData(
            total_score=total_score,
            level=self._classify_level(total_score),
            visual_component=round(visual_score * self.weight_visual, 2),
            proxemic_component=round(proxemic_score * self.weight_proxemic, 2),
            touch_component=round(touch_score * self.weight_touch, 2),
            persistence_component=round(persistence_score * self.weight_persistence, 2)
        )