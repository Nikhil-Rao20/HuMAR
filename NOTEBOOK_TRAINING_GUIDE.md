# 📓 Jupyter Notebook Training Guide

## 🎯 Overview

The `main.ipynb` notebook provides a complete, cell-by-cell training pipeline for the Multitask ReLA model. Each cell is designed to be run sequentially, giving you fine-grained control over the training process.

## 📋 Notebook Structure

### **Step 1: Setup Environment and Imports**
- Adds detectron2 to Python path
- Imports all required libraries
- Checks CUDA availability and GPU info

### **Step 2: Import Detectron2 and Project Components**
- Imports Detectron2 modules
- Imports project-specific components (GRES model, configs)

### **Step 3: Define CSV Logger Class**
- Creates `CSVLogger` class
- Logs all training metrics to timestamped CSV files
- Headers: timestamp, iteration, losses, metrics, LR, time per iteration

### **Step 4: Define Training Visualizer Class**
- Creates `TrainingVisualizer` class
- Generates real-time 4-panel training plots:
  - Total loss
  - Task performance metrics
  - Individual task losses
  - Training progress overview

### **Step 5: Define Event Writer and Trainer Classes**
- Creates `TqdmEventWriter` for progress tracking
- Connects progress bar with CSV logging and visualization

### **Step 6: Define SimpleMultitaskTrainer Class**
- Main trainer class extending `DefaultTrainer`
- Implements custom training loop with monitoring
- Features:
  - Model building with suppressed logging
  - Custom optimizer with gradient clipping
  - Progress bar integration
  - Real-time visualization updates
  - CSV logging at each iteration
  - Checkpoint management

### **Step 7: Configure Training**
- Loads YAML configuration file
- Sets up training parameters
- Displays configuration summary

### **Step 8: Register Datasets**
- Registers HuMAR-GREF datasets
- Verifies dataset availability
- Shows dataset statistics

### **Step 9: Build Trainer**
- Creates trainer instance
- Initializes logging and visualization
- Loads pre-trained weights

### **Step 10: START TRAINING! 🚀**
- **This is the main training cell**
- Runs the complete training loop
- Features:
  - Real-time progress bar
  - CSV logging
  - Training plots updated every iteration
  - Detailed reports every 5 iterations
  - Can be interrupted with Ctrl+C

### **Step 11: View Training Results**
- Displays final training plots
- Shows CSV log file path
- Provides instructions for loading metrics

### **Step 12 (Optional): Analyze Training Metrics**
- Loads CSV metrics into pandas DataFrame
- Shows training summary statistics
- Creates detailed analysis plots:
  - Loss curves
  - Performance metrics
  - Learning rate schedule

## 🚀 How to Use

### **Quick Start:**
1. Open `main.ipynb` in Jupyter/VS Code
2. Run cells 1-9 sequentially to set up everything
3. Run cell 10 to start training
4. Monitor progress in real-time
5. Run cells 11-12 after training completes

### **Monitoring During Training:**

#### **Progress Bar:**
```
🚀 Training Multitask ReLA: 33%|██████▋       | 10/30 [02:45<05:30, 16.5s/iter]
Iter=10/30 | Loss=1.2345 | LR=0.000100 | Time/iter=16.5s | ETA=5.5min
```

#### **Console Output (Every 5 Iterations):**
```
📊 Iteration 5/30 | Loss: 1.5234 | LR: 0.000100 | Time: 15.3s/iter | Elapsed: 1.3min
```

#### **Real-time Plots:**
- Plots update automatically in the notebook
- Saved to disk every 5 iterations
- Final plot saved at completion

### **Generated Files:**

```
output/humar_gref_training/
├── training_log_YYYYMMDD_HHMMSS.csv         # All metrics with timestamps
├── training_plot_YYYYMMDD_HHMMSS.png        # Latest training plot
├── training_plot_YYYYMMDD_HHMMSS_final.png  # Final training plot
├── model_0000009.pth                         # Checkpoint files
├── model_0000019.pth
├── model_0000029.pth
└── model_final.pth                           # Final model
```

## 📊 Training Metrics Tracked

### **Losses:**
- Total Loss
- Detection Loss (classification, bounding box regression, RPN)
- Segmentation Loss (mask, dice, cross-entropy)
- Pose Loss (keypoint detection)

### **Performance Metrics:**
- Detection Accuracy (derived from loss)
- Segmentation IoU (derived from loss)
- Pose Accuracy (derived from loss)

### **Training Info:**
- Learning rate at each iteration
- Time per iteration
- Timestamps for each iteration

## 🎨 Visualization Features

### **4-Panel Training Dashboard:**

1. **Total Loss Plot**
   - Shows overall training loss progression
   - Blue line with smooth trend

2. **Task Performance Metrics**
   - Detection Accuracy (red)
   - Segmentation IoU (green)
   - Pose Accuracy (blue)
   - Y-axis scaled 0-1

3. **Individual Task Losses**
   - Separate curves for each task
   - Helps identify which task is struggling
   - Dashed lines for clarity

4. **Training Progress Overview**
   - Progress percentage bar
   - Average performance bar
   - Text annotations with exact values

## 🔧 Customization Options

### **Modify Training Parameters:**
Edit these values in **Step 7** before running:
```python
# In the config file or override here:
cfg.SOLVER.MAX_ITER = 100  # Increase iterations
cfg.SOLVER.IMS_PER_BATCH = 4  # Reduce batch size if OOM
cfg.SOLVER.BASE_LR = 0.0001  # Adjust learning rate
```

### **Change Visualization Frequency:**
Edit in **Step 4**:
```python
# Save plots more/less frequently
if iteration % 10 == 0 or iteration == 1:  # Change from 5 to 10
    plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
```

### **Add Custom Metrics:**
Edit in **Step 5** (`TqdmEventWriter.write()`):
```python
# Add your custom metric calculation
self.metrics['custom_metric'] = calculate_custom_metric()
```

## ⚡ Tips for Best Results

### **1. Monitor GPU Memory:**
```python
# Check GPU memory in a cell:
if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### **2. Save Intermediate Checkpoints:**
Training automatically saves checkpoints every `CHECKPOINT_PERIOD` iterations (default: 10)

### **3. Resume Training:**
If training is interrupted, set in **Step 9**:
```python
trainer.resume_or_load(resume=True)  # Change from False to True
```

### **4. Stop Training Early:**
- Press **Ctrl+C** in the notebook
- OR: Click the stop button in VS Code
- Checkpoints and logs are preserved

### **5. Analyze Metrics During Training:**
Open the CSV file in another program while training:
```python
import pandas as pd
df = pd.read_csv('output/humar_gref_training/training_log_*.csv')
df.plot(x='iteration', y='total_loss')
```

## 🐛 Troubleshooting

### **Issue: CUDA Out of Memory**
**Solution:**
- Reduce batch size: `cfg.SOLVER.IMS_PER_BATCH = 2`
- Reduce image size in config YAML
- Close other GPU-using programs

### **Issue: Training is very slow**
**Check:**
- CUDA availability: Run Step 1 cell
- CPU fallback warnings (check console)
- GPU utilization: `nvidia-smi` in terminal

### **Issue: Progress bar not updating**
**Solution:**
- This is normal for slow iterations
- Check console output for detailed progress
- Wait for first iteration to complete (can be slow)

### **Issue: Plots not displaying**
**Solution:**
- Check `%matplotlib inline` in Step 1
- Plots are saved to disk even if not displayed
- Open saved PNG files manually

### **Issue: Import errors**
**Solution:**
- Run Step 1 to add detectron2 to path
- Check that all dependencies are installed
- Verify detectron2 directory exists

## 📚 Additional Resources

- **Full Training Script:** `train_simple_multitask_2.py`
- **Config File:** `configs/humar_gref_training.yaml`
- **Enhanced Monitoring Guide:** `ENHANCED_MONITORING_GUIDE.md`
- **Dataset Info:** `datasets/DATASET.md`

## 🎉 Advantages of Notebook Training

✅ **Cell-by-cell control** - Run each step individually
✅ **Easy debugging** - Inspect variables between cells
✅ **Visual feedback** - Plots display inline
✅ **Experiment quickly** - Modify and re-run specific cells
✅ **Interactive analysis** - Analyze results immediately
✅ **Educational** - See exactly what each step does
✅ **Reproducible** - Clear sequence of operations
✅ **Interruptible** - Stop/resume at any cell

---

**Happy Training! 🚀**

For questions or issues, check the console output and saved logs.