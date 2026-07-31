# CASTOR Interaction Framework V4 🤖📹

![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![Architecture](https://img.shields.io/badge/Architecture-Clean%20Architecture-success)
![Status](https://img.shields.io/badge/Status-Research%20%26%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

O **CASTOR Interaction Framework V4** é um sistema computacional científico projetado para extrair métricas objetivas de engajamento físico e visual entre crianças (com foco no Transtorno do Espectro Autista - TEA) e o robô social CASTOR. 

Desenvolvido para apoiar pesquisas em Interação Humano-Robô Cognitiva (IHRc), o sistema elimina a subjetividade da observação manual, processando vídeos de sessões terapêuticas através de algoritmos de estado da arte em Inteligência Artificial.

---

## 🎯 Principais Funcionalidades

* **Rastreamento Multi-Alvo Resiliente:** Acompanha múltiplos participantes na cena, com sistema de recuperação automática de IDs via IoU para mitigar falhas durante oclusões.
* **Atenção Visual (Gaze Estimation):** Integra o modelo *Gazelle (DINOv2)* para gerar mapas de calor e estimar cientificamente se a criança está direcionando atenção visual ao robô CASTOR.
* **Estimativa de Proxêmica Real:** Utiliza Matrizes de Homografia (Bird's-Eye View) para converter pixels em distâncias reais (centímetros), classificando a interação em zonas (Íntima, Pessoal, Social e Pública).
* **Interação Física Biomecânica:** Extrai pontos-chave do corpo e das mãos (*MediaPipe*) para detectar contatos físicos exatos e intenções de toque (alcance).
* **Score Multimodal de Engajamento:** Calcula um índice global (0-100) unindo componentes visuais, proxêmicos, táteis e temporais.
* **Geração Automática de Relatórios:** Produz relatórios clínicos em PDF, dashboards visuais, e arquivos de dados brutos (CSV e JSON) prontos para análise estatística.

---

## 🏗️ Arquitetura do Sistema

O framework foi construído rigorosamente sob os princípios de **Clean Architecture** e **SOLID**, garantindo máxima escalabilidade e testabilidade para pesquisas de longo prazo.

* `src/core/`: O coração do sistema. Contém as regras de negócio, cálculo de métricas e entidades matemáticas puras. Independe totalmente de bibliotecas comerciais de IA.
* `src/adapters/`: Os músculos do sistema. Conectores que implementam redes neurais (YOLOv8, MediaPipe, Gazelle) e geração de arquivos (FPDF, Matplotlib).
* `src/use_cases/`: O orquestrador (`analyze_session.py`) que sincroniza o processamento de quadros do OpenCV com a extração de dados científicos.

---

## ⚙️ Pré-requisitos e Instalação

### Requisitos de Hardware
* CPU: Intel Core i5/i7 (8ª Gen+) ou equivalente AMD.
* RAM: 16 GB mínimo (32 GB recomendado).
* GPU: Placa NVIDIA com suporte a CUDA (mínimo 6GB VRAM) recomendada para processamento ágil.

### Instalação

1. Clone este repositório:
```bash
git clone [https://github.com/SEU_USUARIO/castor-interaction-framework.git](https://github.com/SEU_USUARIO/castor-interaction-framework.git)
cd castor-interaction-framework
