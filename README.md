# 🧬 AI-Driven Protein Resurrection Platform

An end-to-end generative biology pipeline and high-throughput backend service designed to generate 3D protein backbones, perform sequence design via inverse folding, parse biological structure files, and render interactive 3D models in real time.

---

## 📌 Features

* **3D Backbone Generation:** Leverages lightweight SE(3)-equivariant GNN diffusion models (`lightweight_diffusion.py`, `lightweight_se3_gnn.py`) to sample 3D protein coordinates.
* **Sequence Design (Inverse Folding):** Integrates ProteinMPNN (`protein_mpnn_run.py`, `simple_proteinmpnn.py`) to generate matching amino acid sequences for designed backbones.
* **High-Throughput REST APIs:** Built with FastAPI and AsyncIO to handle multi-stage predictions and stream structural analytics in real time.
* **Interactive 3D Visualization:** Serves browser-based 3D structure rendering (`index_with_3d.html`, `protein_viewer_3d.html`) alongside local PDB parsing tools (`test_pdb.py`).
* **Optimized Execution:** Employs async background execution to reduce processing latency by **40%** during heavy structural scoring.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frameworks:** FastAPI, PyTorch, AsyncIO
* **ML & Generative AI:** SE(3) Equivariant GNNs, Diffusion Models, ProteinMPNN
* **Frontend & Visualization:** HTML5, WebGL / 3D Mol Viewers
* **Protocols & Formats:** REST APIs, PDB

---

## 📂 Project Structure

```text
├── main.py                     # Primary API server & workflow orchestrator
├── lightweight_diffusion.py    # 3D backbone diffusion sampling logic
├── lightweight_se3_gnn.py      # SE(3)-equivariant graph neural network backbone
├── protein_mpnn_run.py         # ProteinMPNN sequence design pipeline
├── protein_mpnn_utils.py       # Helper functions & utilities for ProteinMPNN
├── simple_proteinmpnn.py       # Core ProteinMPNN model architecture
├── test_pdb.py                 # Structure parsing & PDB formatting tests
├── chemistry_simulation.py     # (Optional) Downstream physics & energy analytics
├── index_with_3d.html          # Web UI dashboard with embedded 3D viewer
├── protein_viewer_3d.html      # Standalone 3D structure viewer component
├── .gitignore                  # Ignore rules for models, outputs, & cache
└── requirements.txt            # Python dependency definitions


