# HuMAR Model Architecture Variants

This document lists all available model architecture combinations, ranked by **best performance-to-parameter ratio** (lower parameters + higher expected performance).

---

## Component Options

### 🔹 Transformer Architectures
| Type | Config Value | Parameters | Description |
|------|-------------|------------|-------------|
| **Fully Conv Optimized** | `fully_conv_optim` | ~2-3M | Ghost modules, inverted bottlenecks, SE attention, YOLOv11 SPP |
| **Efficient Conv+Attention** | `efficient` | ~11M | Linear attention O(N), Conv+Attention blocks |
| **Fully Convolutional** | `fully_conv` | ~12M | Pure convolution, ConvNeXt blocks, SPP |
| **Original Deformable** | `original` | ~27M | Deformable attention, full transformer |

### 🔹 Text Encoders
| Type | Config Value | Parameters | Hidden Dim | Description |
|------|-------------|------------|------------|-------------|
| **TinyBERT** | `tinybert` | 14.5M | 312 | Ultra-lightweight BERT distillation |
| **MiniLM** | `minilm` | 23M | 384 | Efficient sentence transformer (default) |
| **DistilBERT** | `distilbert` | 66M | 768 | Distilled BERT, good balance |
| **TiTe-LATE** | `tite` | - | 768 | Learnable absolute text embeddings |
| **RoBERTa** | `roberta` | 125M | 768 | Original, highest performance |

### 🔹 Vision Backbones
| Type | Config Value | Parameters | Channels | Description |
|------|-------------|------------|----------|-------------|
| **SegFormer B0** | `segformer_b0` | 3.7M | [32,64,160,256] | Lightweight hierarchical vision transformer |
| **SegFormer B1** | `segformer_b1` | 13.7M | [64,128,320,512] | Balanced SegFormer |
| **MobileViT** | `mobilevit_*` | Varies | - | Hybrid CNN-Transformer for mobile |
| **EfficientFormer** | `efficientformer_*` | Varies | - | Efficient vision transformer |
| **PoolFormer** | `poolformer_*` | Varies | - | MetaFormer with pooling |
| **Swin Tiny** | `swin_T_224_1k` | ~28M | [96,192,384,768] | Default, window-based attention |

---

## 🏆 Top 10 Recommended Combinations
*Ordered by best performance/parameter trade-off*

### **1. Ultra-Lightweight Champion** ⭐ BEST EFFICIENCY
**Total Params: ~20M** (Transformer: 2-3M, Text: 14.5M, Backbone: 3.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'fully_conv_optim'
text_encoder_type = 'tinybert'
backbone = 'segformer_b0'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/ultra_light
```

**Why this works:**
- Ghost modules reduce conv params by 50%
- TinyBERT provides 312-dim embeddings (sufficient for keypoint tasks)
- SegFormer B0 is extremely efficient hierarchical ViT
- **Best for:** Limited GPU memory (4-6GB), fast inference

---

### **2. Balanced Lightweight** ⭐ RECOMMENDED
**Total Params: ~29M** (Transformer: 2-3M, Text: 23M, Backbone: 3.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'fully_conv_optim'
text_encoder_type = 'minilm'
backbone = 'segformer_b0'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/balanced_light
```

**Why this works:**
- MiniLM is sentence-transformer optimized (better semantic understanding than TinyBERT)
- Still very lightweight overall
- Good balance of speed and accuracy
- **Best for:** General use, 6GB+ GPU

---

### **3. Higher Capacity Lightweight**
**Total Params: ~36M** (Transformer: 2-3M, Text: 23M, Backbone: 13.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'fully_conv_optim'
text_encoder_type = 'minilm'
backbone = 'segformer_b1'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/higher_capacity_light
```

**Why this works:**
- SegFormer B1 provides richer visual features
- Still uses optimized transformer
- Better for complex scenes
- **Best for:** 8GB+ GPU, higher accuracy needed

---

### **4. Efficient Attention + Lightweight Encoder**
**Total Params: ~38M** (Transformer: 11M, Text: 23M, Backbone: 3.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'efficient'
text_encoder_type = 'minilm'
backbone = 'segformer_b0'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/efficient_attn_light
```

**Why this works:**
- Linear attention captures long-range dependencies efficiently
- Conv+Attention hybrid balances locality and global context
- Good for sparse keypoint relationships
- **Best for:** Tasks requiring attention mechanisms

---

### **5. Full-Featured Lightweight**
**Total Params: ~72M** (Transformer: 2-3M, Text: 66M, Backbone: 3.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'fully_conv_optim'
text_encoder_type = 'distilbert'
backbone = 'segformer_b0'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/full_featured_light
```

**Why this works:**
- DistilBERT provides 768-dim embeddings (matches original)
- Better language understanding than smaller encoders
- Still uses optimized transformer
- **Best for:** Complex text queries, 8GB+ GPU

---

### **6. Efficient Attention + Balanced Backbone**
**Total Params: ~48M** (Transformer: 11M, Text: 23M, Backbone: 13.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'efficient'
text_encoder_type = 'minilm'
backbone = 'segformer_b1'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/efficient_balanced
```

**Why this works:**
- Balanced parameters across all components
- Linear attention + richer visual features
- Good all-around performance

---

### **7. Learnable Text Embeddings + Ultra-Light**
**Total Params: ~6M** (Transformer: 2-3M, Text: ~1M, Backbone: 3.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'fully_conv_optim'
text_encoder_type = 'tite'
backbone = 'segformer_b0'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/tite_ultra_light
```

**Why this works:**
- TiTe learns task-specific text embeddings from scratch
- Extremely lightweight (no pretrained encoder)
- May work well if text queries are limited/repetitive
- **Best for:** Experimental, limited vocabulary

---

### **8. Pure Conv + Lightweight Encoder**
**Total Params: ~39M** (Transformer: 12M, Text: 23M, Backbone: 3.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'fully_conv'
text_encoder_type = 'minilm'
backbone = 'segformer_b0'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/pure_conv_light
```

**Why this works:**
- ConvNeXt-style blocks capture local patterns well
- No attention overhead
- Fast inference
- **Best for:** Baseline comparison with conv-only

---

### **9. Original Transformer + Lightweight Components**
**Total Params: ~54M** (Transformer: 27M, Text: 23M, Backbone: 3.7M)

```bash
# Edit config/uniphd.py:
transformer_type = 'original'
text_encoder_type = 'minilm'
backbone = 'segformer_b0'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/original_light_components
```

**Why this works:**
- Keeps proven deformable transformer
- Reduces params in other components
- Good middle ground

---

### **10. Original Setup (Baseline)**
**Total Params: ~180M** (Transformer: 27M, Text: 125M, Backbone: 28M)

```bash
# Edit config/uniphd.py:
transformer_type = 'original'
text_encoder_type = 'roberta'
backbone = 'swin_T_224_1k'

# Train:
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/original_baseline
```

**Why this works:**
- Original paper configuration
- Highest capacity, best accuracy (likely)
- **Best for:** Baseline comparison, high-end GPU

---

## 📝 How to Switch Configurations

### Method 1: Edit config/uniphd.py directly
```python
# Line 30: Transformer type
transformer_type = 'fully_conv_optim'  # Change this

# Line 43: Text encoder (in models/uniphd/text_encoder/text_encoder.py)
text_encoder_type = 'minilm'  # Change this

# Line 38: Backbone (in config/uniphd.py)
backbone = 'segformer_b0'  # Change this
```

### Method 2: Command-line override (if implemented)
```bash
python main.py \
    --config config/uniphd.py \
    --transformer_type fully_conv_optim \
    --text_encoder_type minilm \
    --backbone segformer_b0 \
    --output_dir outputs/my_experiment
```

---

## 🧪 Experimental Combinations

### Mobile-Optimized
```python
transformer_type = 'fully_conv_optim'
text_encoder_type = 'tinybert'
backbone = 'mobilevit_s'  # If available in timm
```

### Memory-Constrained (4GB VRAM)
```python
transformer_type = 'fully_conv_optim'
text_encoder_type = 'tite'
backbone = 'segformer_b0'
# Also reduce batch_size and num_queries
```

### Max Efficiency Research
```python
transformer_type = 'fully_conv_optim'
text_encoder_type = 'tinybert'
backbone = 'poolformer_s12'  # Lightweight MetaFormer
```

---

## 📊 Training Tips

### Monitor Parameter Count
Add this to your training script to verify total parameters:
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model):,}")
```

### Adjust Hyperparameters for Lightweight Models
For models under 50M params, consider:
- Higher learning rate: `lr = 2e-4` (instead of 1e-4)
- Longer warmup: `lr_warmup_steps = 2000`
- Less regularization: `dropout = 0.0`, `weight_decay = 1e-5`
- More epochs: `epochs = 100` (lightweight models train faster)

### Batch Size Recommendations
| Total Params | 6GB VRAM | 8GB VRAM | 12GB+ VRAM |
|--------------|----------|----------|------------|
| < 30M | 8-16 | 16-24 | 32+ |
| 30-50M | 4-8 | 8-16 | 16-32 |
| 50-100M | 2-4 | 4-8 | 8-16 |
| > 100M | 1-2 | 2-4 | 4-8 |

---

## 🎯 Expected Performance Trade-offs

| Configuration | Speed | Accuracy | VRAM | Best Use Case |
|--------------|-------|----------|------|---------------|
| #1 Ultra-Light | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 3-4GB | Mobile, edge devices |
| #2 Balanced | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 4-5GB | **General use** |
| #3 Higher Cap | ⚡⚡⚡ | ⭐⭐⭐⭐ | 5-6GB | Complex scenes |
| #4 Efficient Attn | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 5-6GB | Long-range deps |
| #10 Original | ⚡⚡ | ⭐⭐⭐⭐⭐ | 8-10GB | Max accuracy |

---

## 🚀 Quick Start Commands

**Test parameter count only:**
```bash
python -c "
import torch
from config.uniphd import build_model
model = build_model()
total = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total:,} ({total/1e6:.1f}M)')
"
```

**Train with best recommended config:**
```bash
# 1. Edit config/uniphd.py:
#    transformer_type = 'fully_conv_optim'
# 2. Edit models/uniphd/text_encoder/text_encoder.py line 42:
#    text_encoder_type = 'minilm'
# 3. Edit config/uniphd.py line 38:
#    backbone = 'segformer_b0'

# Run training
python main.py --config config/uniphd.py --coco_path datasets/RefHuman --output_dir outputs/best_config --batch_size 8
```

---

## 📌 Notes

1. **SegFormer requires HuggingFace Transformers**: `pip install transformers`
2. **Text encoder type** must be changed in `models/uniphd/text_encoder/text_encoder.py` line 42
3. **All combinations have been integrated** - just change config values
4. **Start with #2 (Balanced Lightweight)** for best results on your RTX 4050 6GB
5. **Parameter counts are approximate** - run the model to get exact counts

Last Updated: December 15, 2025
