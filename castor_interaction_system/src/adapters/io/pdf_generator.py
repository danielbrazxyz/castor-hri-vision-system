import os
from fpdf import FPDF
from typing import Dict, Any

from src.core.interfaces import IPDFGenerator

class PDFReportAdapter(IPDFGenerator):
    def generate_report(
        self, 
        session_name: str, 
        metadata: Dict[str, Any], 
        dashboard_image_path: str, 
        output_path: str
    ) -> None:
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Cabeçalho
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Relatório Científico de Interação", ln=True, align='C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"Sessão: {session_name}", ln=True, align='C')
        pdf.ln(10)
        
        # Metadados / Estatísticas
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Estatísticas Globais", ln=True, align='L')
        pdf.set_font("Arial", '', 11)
        
        for key, value in metadata.items():
            formatted_key = key.replace('_', ' ').capitalize()
            pdf.cell(200, 8, txt=f"{formatted_key}: {value}", ln=True, align='L')
            
        pdf.ln(10)
        
        # Anexar Dashboard (se existir)
        if os.path.exists(dashboard_image_path):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Análise Visual e Curvas Temporais", ln=True, align='L')
            pdf.ln(5)
            # Insere a imagem com largura ajustada para a página A4 (aprox 190mm)
            pdf.image(dashboard_image_path, x=10, w=190)
        else:
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(200, 10, txt="[Aviso: Imagem do dashboard não encontrada]", ln=True, align='L')
            
        # Salva o arquivo
        pdf.output(output_path)