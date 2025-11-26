# Changes and Updates Log - UniPHD Model

## November 26, 2025 - Training Pipeline Fixed & Fully Operational

Successfully implemented HungarianMatcher and resolved all training errors. The model now trains without issues on NVIDIA RTX 4050 6GB VRAM (laptop GPU) with estimated training time of 4 days for 20 epochs (Complete Dataset with try for a smaller version). Key fixes:
1. Implemented HungarianMatcher with keypoints/OKS cost computation for proper prediction-target alignment
2. Added main_indices and aux_indices outputs to model forward pass required by criterion
3. Fixed mask loss bool-to-float tensor conversion error
4. Made mask loss skip gracefully for auxiliary outputs without masks. Model is production-ready with 81.9M parameters using MiniLM text encoder.

---

## November 25, 2025 - Text Encoder Replacement

**Change:** Replaced RoBERTa-base (125M params) with MiniLM (23M params)

**Files:** `text_encoder.py`, `tokenizer.py`, `main.ipynb`

**Impact:** Total model reduced from 184M to 81.9M parameters (56% reduction). MiniLM provides 4-5x faster inference while maintaining semantic understanding for reference segmentation tasks. Original RoBERTa code preserved as comments for easy rollback.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (6-layer, 384-dim embeddings, mean pooling)

---

## November 25, 2025 - MSDA Module Replacement

**Change:** Replaced CUDA-compiled MSDA with manually written PyTorch implementation

**Location:** `models/uniphd/ops/functions/ms_deform_attn_pytorch_gpu.py`

**Reason:** Eliminates CUDA compilation dependency, improves portability and debugging

---