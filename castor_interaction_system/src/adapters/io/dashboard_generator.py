import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from typing import List
import numpy as np

from src.core.interfaces import IDashboardGenerator
from src.core.entities import InteractionResult, InteractionEvent

class MatplotlibDashboardGenerator(IDashboardGenerator):
    def generate(
        self, 
        frame_results: List[InteractionResult], 
        events: List[InteractionEvent], 
        fps: float, 
        output_path: str
    ) -> None:
        
        # Prevenção contra sessões vazias
        if not frame_results:
            return

        # Prepara a figura com 3 subplots empilhados (Timeline, Curva de Score, Distribuição Proxêmica)
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1])
        
        ax_timeline = fig.add_subplot(gs[0, :])
        ax_score = fig.add_subplot(gs[1, :])
        ax_proxemic = fig.add_subplot(gs[2, 0])
        ax_stats = fig.add_subplot(gs[2, 1])

        # ---------------------------------------------------------
        # 1. TIMELINE DE EVENTOS (Gantt Chart)
        # ---------------------------------------------------------
        y_labels = ["Gaze (Olhar)", "Toque", "Zona Íntima"]
        ax_timeline.set_yticks([10, 20, 30])
        ax_timeline.set_yticklabels(y_labels)
        ax_timeline.set_ylim(5, 35)
        
        for event in events:
            start_sec = event.start_frame / fps
            duration = event.duration_sec
            
            if "GAZE" in event.event_type:
                ax_timeline.broken_barh([(start_sec, duration)], (8, 4), facecolors='blue', alpha=0.7)
            elif "TOUCH" in event.event_type:
                ax_timeline.broken_barh([(start_sec, duration)], (18, 4), facecolors='red', alpha=0.7)
            elif "INTIMATE" in event.event_type:
                ax_timeline.broken_barh([(start_sec, duration)], (28, 4), facecolors='green', alpha=0.5)

        ax_timeline.set_title("Timeline de Eventos de Interação")
        ax_timeline.set_xlabel("Tempo (segundos)")
        ax_timeline.grid(True, axis='x', linestyle='--', alpha=0.6)

        # ---------------------------------------------------------
        # 2. CURVA DE SCORE MULTIMODAL
        # ---------------------------------------------------------
        # Assumindo que InteractionResult possui o campo interaction_score
        time_seconds = [r.frame_idx / fps for r in frame_results]
        scores = [r.interaction_score for r in frame_results]
        
        ax_score.plot(time_seconds, scores, label='Score de Engajamento', color='purple', linewidth=2)
        ax_score.fill_between(time_seconds, scores, color='purple', alpha=0.1)
        ax_score.set_ylim(0, 105)
        ax_score.set_title("Evolução do Score Multimodal")
        ax_score.set_ylabel("Score (0-100)")
        ax_score.set_xlabel("Tempo (segundos)")
        
        # Linhas de referência para os Níveis de Interação
        ax_score.axhline(20, color='gray', linestyle=':', alpha=0.5)
        ax_score.axhline(40, color='gray', linestyle=':', alpha=0.5)
        ax_score.axhline(60, color='gray', linestyle=':', alpha=0.5)
        ax_score.axhline(80, color='gray', linestyle=':', alpha=0.5)
        ax_score.legend(loc="upper right")

        # ---------------------------------------------------------
        # 3. DISTRIBUIÇÃO PROXÊMICA (Gráfico de Rosca / Donut Chart)
        # ---------------------------------------------------------
        zones = [r.zone for r in frame_results]
        zone_counts = Counter(zones)
        
        labels = list(zone_counts.keys())
        sizes = list(zone_counts.values())
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
        
        ax_proxemic.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
        # Transforma em Donut
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax_proxemic.add_artist(centre_circle)
        ax_proxemic.set_title("Distribuição do Tempo por Zona Proxêmica")

        # ---------------------------------------------------------
        # 4. ESTATÍSTICAS GERAIS (Texto)
        # ---------------------------------------------------------
        ax_stats.axis('off')
        avg_score = np.mean(scores) if scores else 0
        total_time = len(frame_results) / fps
        total_touches = sum(1 for e in events if "TOUCH" in e.event_type)
        
        stats_text = (
            f"Resumo da Sessão\n"
            f"-----------------\n"
            f"Duração Total: {total_time:.1f} s\n"
            f"Score Médio: {avg_score:.1f} / 100\n"
            f"Total de Toques: {total_touches}\n"
            f"Pico de Engajamento: {max(scores):.1f}\n"
        )
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center', fontfamily='monospace')

        # Ajuste de layout e exportação
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Libera memória RAM