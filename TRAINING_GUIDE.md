# 🚀 Enhanced Multitask ReLA Training Guide
## Detection + Segmentation + Pose Estimation on HuMAR Dataset

This guide provides complete instructions for training the enhanced multitask ReLA model with sophisticated detection and pose heads that match segmentation head complexity.

## 📋 Prerequisites

### 1. Environment Setup
```powershell
# Make sure you're in the ReLA directory
cd "C:\Users\nikhi\Desktop\ReLA"

# Install required packages (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install transformers
pip install pycocotools
pip install opencv-python
pip install matplotlib
pip install seaborn
pip install wandb  # Optional: for experiment tracking
```

### 2. Dataset Preparation
```powershell
# Ensure HuMAR dataset is properly structured:
# datasets/
#   └── humar/
#       ├── train/  # Training images
#       ├── val/    # Validation images  
#       ├── gref_umd_train.json      # GREF format annotations
#       ├── gref_umd_val.json        # GREF format annotations
#       ├── instances_train.json     # COCO format with keypoints
#       └── instances_val.json       # COCO format with keypoints

# Verify dataset registration
python -c "from gres_model.data.datasets.register_humar import register_humar_datasets; register_humar_datasets(); print('✅ HuMAR datasets registered successfully!')"
```

## 🎯 Training Commands

### Option 1: Quick Start Training (Recommended)
```powershell
# Single GPU training with enhanced multitask model
python train_enhanced_multitask.py --config-file configs/enhanced_multitask_humar.yaml --num-gpus 1 OUTPUT_DIR ./output/enhanced_multitask_run1

# Multi-GPU training (if you have multiple GPUs)
python train_enhanced_multitask.py --config-file configs/enhanced_multitask_humar.yaml --num-gpus 2 OUTPUT_DIR ./output/enhanced_multitask_run1
```

### Option 2: Custom Training Parameters
```powershell
# Training with custom parameters
python train_enhanced_multitask.py \
    --config-file configs/enhanced_multitask_humar.yaml \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 100000 \
    MODEL.MULTITASK.DETECTION_LOSS_WEIGHT 1.0 \
    MODEL.MULTITASK.POSE_LOSS_WEIGHT 1.0 \
    OUTPUT_DIR ./output/custom_multitask_run
```

### Option 3: Resume Training
```powershell
# Resume from checkpoint
python train_enhanced_multitask.py \
    --config-file configs/enhanced_multitask_humar.yaml \
    --resume \
    --num-gpus 1 \
    OUTPUT_DIR ./output/enhanced_multitask_run1
```

## 📊 Monitoring Training

### 1. Real-time Monitoring
During training, you'll see output like:
```
📊 Iteration 100 - Multitask Metrics:
   🎯 Detection Loss: 0.8234
   🎭 Segmentation Loss: 1.2456
   🤸 Pose Loss: 0.6789
   📈 Total Loss: 2.7479
   ⚡ Learning Rate: 0.000100
   ⏱️  Iter Time: 0.856s
```

### 2. TensorBoard Monitoring
```powershell
# Launch TensorBoard to monitor training
tensorboard --logdir ./output/enhanced_multitask_run1

# Open http://localhost:6006 in your browser
```

### 3. Saved Metrics
Training metrics are automatically saved to:
- `./output/enhanced_multitask_run1/training_metrics.json`
- Contains detailed loss curves for all tasks

## 🎛️ Training Configuration Options

### Key Parameters in `enhanced_multitask_humar.yaml`:

```yaml
# Batch size (adjust based on GPU memory)
SOLVER.IMS_PER_BATCH: 16  # Reduce to 8 or 4 if out of memory

# Learning rate
SOLVER.BASE_LR: 0.0001  # Base learning rate

# Training duration  
SOLVER.MAX_ITER: 150000  # Total training iterations

# Loss weights (balance between tasks)
MODEL.MULTITASK.DETECTION_LOSS_WEIGHT: 1.0
MODEL.MULTITASK.SEGMENTATION_LOSS_WEIGHT: 1.0
MODEL.MULTITASK.POSE_LOSS_WEIGHT: 1.0

# Architecture complexity
MODEL.DETECTION_HEAD.NUM_TRANSFORMER_LAYERS: 3  # Detection transformer layers
MODEL.POSE_HEAD.NUM_TRANSFORMER_LAYERS: 4       # Pose transformer layers
```

## 🔧 Troubleshooting

### 1. Out of Memory Error
```powershell
# Reduce batch size
python train_enhanced_multitask.py \
    --config-file configs/enhanced_multitask_humar.yaml \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 4 \
    INPUT.IMAGE_SIZE 384 \
    OUTPUT_DIR ./output/reduced_memory_run
```

### 2. Slow Training
```powershell
# Reduce model complexity temporarily
python train_enhanced_multitask.py \
    --config-file configs/enhanced_multitask_humar.yaml \
    --num-gpus 1 \
    MODEL.DETECTION_HEAD.NUM_TRANSFORMER_LAYERS 2 \
    MODEL.POSE_HEAD.NUM_TRANSFORMER_LAYERS 2 \
    OUTPUT_DIR ./output/reduced_complexity_run
```

### 3. Dataset Issues
```powershell
# Test dataset loading
python test_humar_dataloader.py  # Verify dataloader works

# Check dataset registration
python -c "from detectron2.data import DatasetCatalog; print(list(DatasetCatalog.list()))"
```

## 📈 Expected Training Timeline

| Phase | Iterations | Expected Behavior |
|-------|------------|-------------------|
| **Warmup** | 0-1,000 | High losses, rapid decrease |
| **Initial Learning** | 1,000-20,000 | Steady loss decrease, all tasks improving |
| **Stabilization** | 20,000-80,000 | Slower loss decrease, balanced multitask learning |
| **Fine-tuning** | 80,000-150,000 | Fine loss adjustments, convergence |

## 🎯 Expected Loss Values

After convergence, you should see approximately:
- **Detection Loss**: ~0.3-0.6
- **Segmentation Loss**: ~0.8-1.2  
- **Pose Loss**: ~0.2-0.5
- **Total Loss**: ~1.5-2.5

## 💾 Model Checkpoints

Training automatically saves:
- `model_final.pth` - Final trained model
- `model_0009999.pth` - Checkpoint every 10K iterations
- `last_checkpoint` - Path to latest checkpoint for resuming

## 🧪 Testing Trained Model

```powershell
# Test the trained model
python train_enhanced_multitask.py \
    --config-file configs/enhanced_multitask_humar.yaml \
    --eval-only \
    MODEL.WEIGHTS ./output/enhanced_multitask_run1/model_final.pth \
    OUTPUT_DIR ./output/enhanced_multitask_run1/evaluation
```

## 🚀 Advanced Training Options

### 1. Distributed Training (Multiple GPUs)
```powershell
# 4 GPU training
python train_enhanced_multitask.py \
    --config-file configs/enhanced_multitask_humar.yaml \
    --num-gpus 4 \
    SOLVER.IMS_PER_BATCH 32 \
    OUTPUT_DIR ./output/distributed_run
```

### 2. Mixed Precision Training
```powershell
# Enable automatic mixed precision for faster training
python train_enhanced_multitask.py \
    --config-file configs/enhanced_multitask_humar.yaml \
    --num-gpus 1 \
    SOLVER.AMP.ENABLED True \
    OUTPUT_DIR ./output/mixed_precision_run
```

### 3. Custom Loss Weights
```powershell
# Emphasize pose estimation
python train_enhanced_multitask.py \
    --config-file configs/enhanced_multitask_humar.yaml \
    --num-gpus 1 \
    MODEL.MULTITASK.DETECTION_LOSS_WEIGHT 0.8 \
    MODEL.MULTITASK.SEGMENTATION_LOSS_WEIGHT 0.8 \
    MODEL.MULTITASK.POSE_LOSS_WEIGHT 1.5 \
    OUTPUT_DIR ./output/pose_focused_run
```

## ✅ Success Indicators

Training is successful when you see:
1. **All three losses decreasing** steadily
2. **No NaN values** in losses
3. **Balanced learning** across tasks
4. **GPU memory usage** stable
5. **Multitask metrics** improving consistently

## 📞 Need Help?

If you encounter issues:
1. Check the error messages in terminal
2. Review `./output/enhanced_multitask_run1/log.txt`
3. Verify dataset paths and registration
4. Try reduced batch size or model complexity
5. Ensure all dependencies are installed correctly

## 🎉 Next Steps After Training

1. **Evaluate** the trained model on test set
2. **Visualize** predictions using the visualization scripts
3. **Fine-tune** hyperparameters if needed
4. **Export** model for deployment
5. **Analyze** multitask learning performance

---

**🚀 Ready to train your enhanced multitask ReLA model!**

Start with the recommended quick start command and monitor the training progress. The enhanced model with sophisticated detection and pose heads will learn to perform all three tasks simultaneously! 🎯🎭🤸