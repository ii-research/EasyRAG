<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-red.svg" alt="PyTorch 2.0+"></a>
  <a href="https://nicegui.io/"><img src="https://img.shields.io/badge/UI-NiceGUI-green.svg" alt="NiceGUI"></a>
</p>

# EasyRAG: A Beginner-friendly and Interactive Framework for Retrieval-Augmented Generation

**EasyRAG** is an open-source framework providing faithful implementations of five RAG algorithms — from simple baselines to advanced methods — with an interactive web dashboard for training, evaluation, and inference. It includes the **first publicly available implementations** of [FiD-Light](https://dl.acm.org/doi/abs/10.1145/3539618.3591687) (encoder compression + Source Pointing) and [Stochastic RAG](https://dl.acm.org/doi/10.1145/3626772.3657923) (Gumbel-Top-k differentiable reranking), designed for accessibility, reproducibility, and education.

---

## Table of Contents

- [Key Features](#key-features)
- [Supported Algorithms](#supported-algorithms)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Training Pipeline](#full-training-pipeline)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Key Features

- **5 RAG Algorithms** — Closed-book (Direct), Naive RAG, FiD, FiD-Light, and Stochastic RAG in a unified codebase
- **First Open-Source FiD-Light & Stochastic RAG** — Faithful reproductions following original papers' algorithms and hyperparameters
- **Modular 7-Step Pipeline** — From data download to evaluation, each step independently runnable and resumable
- **Interactive Web Dashboard** — NiceGUI-based UI for pipeline management, real-time training monitoring, and inference
- **Live Demo** — Instant RAG demonstration with DuckDuckGo web search, no dataset download required
- **Multi-Backbone** — Supports T5-Base (220M) and T5Gemma2 (540M)
- **Hardware Accessible** — Runs on any CUDA-enabled NVIDIA GPU, from a single consumer GPU to multi-GPU clusters
- **KILT Benchmark** — Evaluation on NQ, TriviaQA, and HotpotQA with EM, F1, KILT-EM, and KILT-F1 metrics

---

## Supported Algorithms

| Algorithm | Description | Provenance | Key Technique |
|:---|:---|:---:|:---|
| **Closed-book (Direct)** | Answer from parametric knowledge only, no retrieval | — | Baseline lower bound |
| **Naive RAG** | Concatenate retrieved passages into a single input | — | Standard retrieve-then-read |
| **FiD** | Encode passages independently, fuse in decoder cross-attention | — | [Izacard & Grave, 2021](https://aclanthology.org/2021.eacl-main.74/) |
| **FiD-Light** | FiD + encoder compression (top-k vectors) + Source Pointing | Yes | [Hofstatter et al., 2023](https://dl.acm.org/doi/abs/10.1145/3539618.3591687) |
| **Stochastic RAG** | End-to-end differentiable passage reranking via Gumbel-Top-k | Yes | [Zamani & Bendersky, 2024](https://dl.acm.org/doi/10.1145/3626772.3657923) |

---

## Screenshots

<p align="center">
  <img src="figures/dashboard.png" width="80%" alt="Training Dashboard">
  <br><em>Training pipeline dashboard with real-time loss curves and step status indicators.</em>
</p>

<p align="center">
  <img src="figures/inference.png" width="80%" alt="Inference Interface">
  <br><em>Interactive inference with retrieved passages and Source Pointing visualization.</em>
</p>

<p align="center">
  <img src="figures/compare.png" width="80%" alt="Model Comparison">
  <br><em>Side-by-side comparison of multiple RAG algorithms on the same query.</em>
</p>

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable NVIDIA GPU (8GB+ VRAM for T5-Base)
- CUDA 12.1

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/anonymous/EasyRAG.git
cd EasyRAG

# 2. Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install Faiss GPU
pip install faiss-gpu-cu12==1.13.2

# 4. Install remaining dependencies
pip install -r requirements.txt
```

<details>
<summary><b>Key dependencies</b></summary>

| Package | Purpose |
|:---|:---|
| `transformers` | Model loading and tokenization |
| `sentence-transformers` | GTR-T5-Base retriever |
| `faiss-gpu` | Dense vector similarity search |
| `nicegui` | Web dashboard UI |
| `ddgs` | DuckDuckGo search (live demo) |
| `datasets` | KILT data loading |
| `tensorboard` | Training visualization |

</details>

---

## Quick Start

### Live Demo (No Data Download)

Try RAG instantly using web search — no Wikipedia download or pre-training required:

```bash
# Interactive mode
python web_rag_demo.py --interactive --model t5base

# Single query
python web_rag_demo.py --query "Who directed Parasite?" --mode naive_rag --model t5base

# Compare: Naive RAG vs. Closed-book (Direct)
python web_rag_demo.py --query "Who directed Parasite?" --mode direct --model t5base
```

### Web Dashboard

Launch the full web interface for pipeline management, training, inference, and model comparison:

```bash
python -m web_demo.app
# Open http://localhost:8080
```

The dashboard provides five pages:

| Page | Function |
|:---|:---|
| **Dashboard** | Pipeline overview with step status, real-time loss curves, log streaming |
| **Evaluate** | Run evaluation on checkpoints with configurable metrics |
| **Inference** | Interactive Q&A with any trained model, Source Pointing visualization |
| **Compare** | Side-by-side comparison of multiple algorithms on the same query |
| **Live Demo** | Closed-book vs. Naive RAG with DuckDuckGo web search |

---

## Full Training Pipeline

EasyRAG implements a modular 7-step pipeline. Each step is independent and can be skipped if pre-trained artifacts are available.

```
Download & Preprocess ──> Build Index ──> Train Retriever ──> Precompute ──> Train Model ──> Evaluate
     [Step 1]              [Step 2-3]       [Step 4]          [Step 5]       [Step 6]       [Step 7]
```

### Step 1: Download and Prepare Data

```bash
python download_kilt_data.py        # Download KILT Wikipedia + NQ, TriviaQA, HotpotQA
python fix_triviaqa.py              # Fix TriviaQA missing question text
python filter_kilt_data.py          # Filter samples without valid provenance
```

### Step 2–3: Build Retrieval Index

```bash
python build_wiki_index.py          # Convert Wikipedia to Arrow format
python build_gtr_index.py           # Build Faiss index with GTR-T5-Base embeddings
```

### Step 4: Train Retriever (optional)

```bash
python generate_retrieval_training_data.py    # Generate training triplets
python train_gtr_retriever.py                 # Fine-tune GTR retriever
```

### Step 5: Precompute Retrieval

```bash
# For FiD-Light / Stochastic RAG (top-40 passages)
python precompute_retrieval.py

# For FiD (top-100 passages)
python precompute_retrieval_for_fid.py
```

### Step 6: Train Models

| Algorithm | T5-Base | T5Gemma2 |
|:---|:---|:---|
| Closed-book (Direct) | `train_direct.py` | `train_direct_t5gemma.py` |
| Naive RAG | `train_naive_rag.py` | `train_naive_rag_t5gemma.py` |
| FiD | `train_fid_pure.py` | `train_fid_pure_t5gemma.py` |
| FiD-Light | `train_fidlight_paper.py` | `train_fidlight_t5gemma.py` |
| Stochastic RAG | `train_stochastic_rag.py` | `train_stochastic_rag_t5gemma.py` |

Common options:

```bash
python train_fidlight_paper.py \
    --precomputed_path data/precomputed/all_tasks_train.parquet \
    --output_dir checkpoints/fidlight \
    --steps 50000 \
    --multi_gpu           # Use all available GPUs
    --resume checkpoint/  # Resume from checkpoint
```

### Step 7: Evaluate

```bash
# Single checkpoint
python evaluate_fidlight.py --checkpoint checkpoints/fidlight/final --task nq

# All checkpoints in a directory
python evaluate_fidlight_t5base_all_checkpoints.py \
    --checkpoint_dir checkpoints/fidlight/
```

---

## Evaluation

EasyRAG evaluates on three KILT benchmark tasks with four metrics:

| Metric | Description |
|:---|:---|
| **EM** (Exact Match) | Whether the predicted answer exactly matches any gold answer |
| **F1** | Token-level F1 between prediction and gold answer |
| **KILT-EM** | EM conditioned on correct provenance (Source Pointing) |
| **KILT-F1** | F1 conditioned on correct provenance |

Evaluation scripts are provided for each algorithm and backbone combination. All scripts support `--task` (nq, triviaqa, hotpotqa, or all) and `--multi_gpu` options.

For detailed experimental results, please refer to the paper.

---

## Project Structure

```
EasyRAG/
├── web_demo/                              # Web dashboard (NiceGUI)
│   ├── app.py                             #   Entry point: python -m web_demo.app
│   ├── pipeline_orchestrator.py           #   Pipeline step management
│   ├── inference_demo.py                  #   Inference engine
│   ├── state_monitor.py                   #   Real-time state monitoring
│   ├── components/
│   │   ├── pipeline_overview.py           #   Pipeline visualization
│   │   ├── step_dialog.py                 #   Step configuration dialogs
│   │   ├── log_viewer.py                  #   Log viewer & loss charts
│   │   ├── inference_panel.py             #   Interactive Q&A
│   │   ├── compare_panel.py              #   Side-by-side model comparison
│   │   ├── evaluate_panel.py              #   Evaluation UI
│   │   ├── web_rag_panel.py               #   Live demo (web search RAG)
│   │   └── workspace_selector.py          #   Workspace management
│   └── utils/
│       ├── process_manager.py             #   Subprocess management
│       └── state_io.py                    #   Pipeline state persistence
│
├── web_rag_demo.py                        # Standalone live demo (CLI)
│
├── download_kilt_data.py                  # Step 1: Download KILT datasets
├── fix_triviaqa.py                        # Step 1: Fix TriviaQA format
├── filter_kilt_data.py                    # Step 1: Filter invalid samples
├── build_wiki_index.py                    # Step 2: Build Wikipedia Arrow index
├── build_gtr_index.py                     # Step 3: Build Faiss index
├── generate_retrieval_training_data.py    # Step 4: Generate retriever training data
├── train_gtr_retriever.py                 # Step 4: Fine-tune GTR retriever
├── precompute_retrieval.py                # Step 5: Precompute passages (top-40)
├── precompute_retrieval_for_fid.py        # Step 5: Precompute passages (top-100)
│
├── train_direct.py                        # Train: Closed-book (T5-Base)
├── train_naive_rag.py                     # Train: Naive RAG (T5-Base)
├── train_fid_pure.py                      # Train: FiD (T5-Base)
├── train_fidlight_paper.py                # Train: FiD-Light (T5-Base)
├── train_stochastic_rag.py                # Train: Stochastic RAG (T5-Base)
├── train_*_t5gemma.py                     # Train: T5Gemma2 variants
│
├── evaluate_*.py                          # Evaluation scripts (per algorithm)
├── gtr_retriever.py                       # GTR dense retriever module
├── kilt_loader.py                         # KILT dataset loader
├── multitask_loader.py                    # Multi-task training data sampler
│
├── requirements.txt
└── LICENSE                                # MIT License
```

---

## Citation

```bibtex
@inproceedings{easyrag2026,
  title     = {EasyRAG: A Beginner-friendly and Interactive Framework for Retrieval-Augmented Generation},
  author    = {Anonymous},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (Demo Track)},
  year      = {2026}
}
```

---

## Acknowledgments

EasyRAG builds upon the following works:

- **FiD** — Izacard & Grave. [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://aclanthology.org/2021.eacl-main.74/). EACL 2021.
- **FiD-Light** — Hofstatter et al. [FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation](https://dl.acm.org/doi/abs/10.1145/3539618.3591687). SIGIR 2023.
- **Stochastic RAG** — Zamani & Bendersky. [Stochastic RAG: End-to-End Retrieval-Augmented Generation through Expected Utility Maximization](https://dl.acm.org/doi/10.1145/3626772.3657923). SIGIR 2024.
- **KILT** — Petroni et al. [KILT: a Benchmark for Knowledge Intensive Language Tasks](https://aclanthology.org/2021.naacl-main.200/). NAACL 2021.
- **GTR** — Ni et al. [Large Dual Encoders Are Generalizable Retrievers](https://arxiv.org/abs/2112.07899). 2021.
- **T5Gemma2** — Zhang et al. [T5Gemma 2: Seeing, Reading, and Understanding Longer](https://arxiv.org/abs/2512.14856). 2025.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
