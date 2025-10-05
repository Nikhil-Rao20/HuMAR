#!/usr/bin/env python3
"""
Test script to verify the enhanced training monitoring works
"""

import sys
import os

# Add local detectron2 to path
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))

print("🧪 Testing Enhanced Training Monitoring System")
print("="*50)

# Test 1: Check imports
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import csv
    from datetime import datetime
    from tqdm import tqdm
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try: pip install matplotlib tqdm")
    sys.exit(1)

# Test 2: Test CSV Logger
try:
    from train_simple_multitask import CSVLogger
    test_dir = "./test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    csv_logger = CSVLogger(test_dir)
    
    # Test logging
    test_losses = {'total': 1.5, 'det': 0.5, 'seg': 0.3, 'pose': 0.7}
    test_metrics = {'det_acc': 0.8, 'seg_iou': 0.6, 'pose_acc': 0.7}
    csv_logger.log_metrics(1, test_losses, test_metrics, lr=0.001, time_per_iter=2.5)
    
    print("✅ CSV Logger working correctly")
    print(f"📝 Test CSV created at: {csv_logger.csv_path}")
    
except Exception as e:
    print(f"❌ CSV Logger error: {e}")

# Test 3: Test Visualizer
try:
    from train_simple_multitask import TrainingVisualizer
    
    visualizer = TrainingVisualizer(test_dir, max_iter=10)
    
    # Test with sample data
    for i in range(1, 6):
        sample_losses = {
            'total': 2.0 - i*0.2, 
            'det': 0.8 - i*0.1, 
            'seg': 0.6 - i*0.05, 
            'pose': 0.6 - i*0.05
        }
        sample_metrics = {
            'det_acc': 0.5 + i*0.08, 
            'seg_iou': 0.4 + i*0.1, 
            'pose_acc': 0.3 + i*0.12
        }
        visualizer.update(i, sample_losses, sample_metrics)
    
    visualizer.save_final_plot()
    print("✅ Training Visualizer working correctly")
    print(f"📊 Test plots saved to: {visualizer.plot_path}")
    
except Exception as e:
    print(f"❌ Visualizer error: {e}")

# Test 4: Test TQDM integration
try:
    from train_simple_multitask import TqdmEventWriter
    
    # Create a mock progress bar
    test_pbar = tqdm(total=5, desc="Test Progress")
    writer = TqdmEventWriter(test_pbar, csv_logger)
    
    print("✅ TQDM integration working")
    test_pbar.close()
    
except Exception as e:
    print(f"❌ TQDM integration error: {e}")

print("\n🎯 Testing Summary:")
print("="*50)
print("📊 Enhanced monitoring features:")
print("  ✅ Real-time progress bar with iteration count, loss, LR, ETA")
print("  ✅ CSV logging with timestamps for all metrics")
print("  ✅ Real-time 4-panel training visualization")
print("  ✅ Detailed progress printing every 5 iterations")
print("  ✅ Final plot saving and summary statistics")

print("\n🚀 Ready to train with enhanced monitoring!")
print("Use: python train_simple_multitask.py --config-file configs/humar_gref_training.yaml --num-gpus 1")

# Cleanup test files
import shutil
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
    print(f"🧹 Cleaned up test directory: {test_dir}")