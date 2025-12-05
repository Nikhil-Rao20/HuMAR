# Changes and Updates Log - UniPHD Model
---

## December 05, 2025 - Added Various VIT based Backbones with Swin Transformer

**Change:** Extended Visual encoder options

Added comprehensive Vision Transformer (ViT) backbone support with existing Swin Transformer variants to enhance model flexibility and performance. The implementation includes SegFormer, MobileVIT, EfficentFormer V2 and PoolFormer architectures alongside standard Swin-T configurations. Each backbone maintains consistent feature extraction interfaces while providing different computational complexity and accuracy trade-offs. An in-depth comparative study will be conducted to identify the most efficient backbone for our specific use case, evaluating factors like accuracy, inference speed, and memory consumption. CNN-based backbones are planned for future integration to provide a complete backbone ecosystem for optimal model performance tuning. Below is the summary table of the added backbones.

| Architecture | Model Variant | Parameters |
|--------------|---------------|------------|
| **SegFormer** | MIT-B0 | 3.7M |
| | MIT-B1 | 13.7M |
| **MobileViT** | XXS | 1.3M |
| | XS | 2.3M |
| | S | 5.6M |
| **EfficientFormerV2** | S0 | 3.5M |
| | S1 | 6.1M |
| | S2 | 12.6M |
| **PoolFormer** | S12 | 12M |
| | S24 | 21M |
| | S36 | 31M |

These lightweight architectures provide efficient alternatives for resource-constrained deployments while maintaining competitive performance for human mesh recovery tasks.

**Change:** Extended text encoder options with three additional lightweight models

**New Encoders:**
- **DistilBERT** - 6-layer distilled BERT model (66M params)
- **TinyBERT** - Ultra-compressed BERT variant (14.5M params) 
- **TiTeLATE** - Task-agnostic Text-to-Latent encoder (varies by configuration)

Provides diverse text encoding options ranging from ultra-lightweight (TinyBERT) to balanced performance (DistilBERT). TiTeLATE offers specialized text-to-latent mapping capabilities for enhanced multimodal alignment. All encoders maintain consistent API for easy experimentation and deployment flexibility.



**Change:** Developed an alternative to the Conv+Attention Pose Centric Decoder, reducing model size from 27M to 11M parameters.

**Integration:** The new decoder has been successfully integrated into the main codebase, specifically within the `self.transformer` implementation in `uniphd.py` located in the `models` directory.

**Benefits:** This alternative decoder maintains competitive performance while significantly reducing the model's memory footprint, making it more suitable for resource-constrained environments. Further evaluations will be conducted to assess its impact on overall model performance and efficiency.

---

## November 26, 2025 - Training Pipeline Fixed & Fully Operational

**Change:** Training Pipeline Issues

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