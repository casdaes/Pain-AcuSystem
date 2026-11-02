# Pain-AcuSystem

**Pain-AcuSystem** is a lightweight, locally deployable clinical decision-support system for acupuncture-based pain management. Built on Qwen3.5-4B, it combines LoRA fine-tuning with retrieval-augmented generation (RAG) to support clinical reasoning, syndrome differentiation, acupoint selection, acupuncture procedures, and safety guidance.

## Study configurations

- **A1:** Qwen3.5-4B base model
- **A2:** Qwen3.5-4B + RAG
- **A3:** Qwen3.5-4B + LoRA fine-tuning
- **A4:** Qwen3.5-4B + LoRA fine-tuning + RAG (**Pain-AcuSystem**)
- **A5:** Qwen3.5-397B-A17B + the same RAG pipeline (large-model scale reference)

The public repository contains source code for data construction, QLoRA training, RAG indexing/retrieval, multi-configuration inference, deterministic safety post-processing, blinded rating-sheet generation, statistical analysis, and automated validation.

## Repository structure

```text
Pain-AcuSystem/
├── configs/          # Public study configuration (local paths must be edited)
├── data/             # Data-availability notice only; no clinical data are included
├── painrag/          # RAG knowledge-base and retrieval implementation
├── scripts/          # Training, inference, evaluation, and validation entry points
├── tcm_pipeline/     # Shared model/training utilities
├── tcm_study/        # Core study, safety, schema, and analysis modules
├── R/                # Supporting sensitivity analyses
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `configs/study.yaml` before running the pipeline. The public configuration uses repository-relative placeholders such as `models/Qwen3.5-4B` and `models/bge-m3`; local model files are intentionally excluded from version control.

API credentials must be supplied through environment variables rather than committed to the repository:

```bash
export LLM_API_KEY="YOUR_API_KEY"
```

## Main workflow

```bash
# Build/prepare study data after placing authorized local source files in data/
python scripts/01b_enrich_sft_targets.py

# Fine-tune the 4B model
python scripts/02_train_sft.py

# Build the RAG index
python scripts/03_build_rag.py

# Generate study configurations
python scripts/05_generate_arms.py

# Create blinded rating sheets
python scripts/06_make_rating_sheets.py

# Collect and analyze ratings
python scripts/07_collect_ratings.py
python scripts/08_analyze_study.py

# Automated validation
python scripts/15_validate_outputs.py

# Pain-AcuSystem deployment inference
python scripts/30_a4_infer.py --input case.txt
```

## Data and privacy

The full knowledge base (about 20,000 units) and the reference-book texts are not redistributed because of third-party copyright; they are available from the corresponding author upon reasonable request. Copyright of the source material remains with the original guideline issuers, authors and publishers. The data are provided for non-commercial research use only and are not a substitute for clinical judgement.

## Important note

Pain-AcuSystem is a research decision-support system and is not intended to replace clinician judgment. Clinical deployment requires appropriate validation, governance, and safety oversight.
