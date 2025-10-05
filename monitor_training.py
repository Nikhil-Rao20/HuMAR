#!/usr/bin/env python3
"""
Training Monitor Script
Run this in a separate terminal to monitor training progress
"""

import os
import time
import glob
import json
from datetime import datetime

def monitor_training(output_dir="./output/humar_gref_training", check_interval=10):
    """Monitor training progress by checking logs and checkpoints"""
    
    print("🔍 Training Monitor Started")
    print(f"📁 Monitoring directory: {output_dir}")
    print(f"🔄 Check interval: {check_interval} seconds")
    print("="*60)
    
    last_checkpoint_count = 0
    start_time = time.time()
    
    while True:
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            
            # Check for checkpoints
            checkpoint_pattern = os.path.join(output_dir, "model_*.pth")
            checkpoints = glob.glob(checkpoint_pattern)
            checkpoint_count = len(checkpoints)
            
            # Check for log files
            log_pattern = os.path.join(output_dir, "*.log")
            log_files = glob.glob(log_pattern)
            
            # Check for metrics files
            metrics_pattern = os.path.join(output_dir, "metrics.json")
            metrics_exist = os.path.exists(metrics_pattern)
            
            print(f"[{current_time}] 📊 Training Status:")
            print(f"  ⏱️  Elapsed: {elapsed/60:.1f} minutes")
            print(f"  💾 Checkpoints: {checkpoint_count}")
            print(f"  📋 Log files: {len(log_files)}")
            print(f"  📈 Metrics file: {'✅' if metrics_exist else '❌'}")
            
            # If new checkpoint appeared
            if checkpoint_count > last_checkpoint_count:
                print(f"  🎉 NEW CHECKPOINT SAVED!")
                last_checkpoint_count = checkpoint_count
            
            # Check if training is still running (look for recent file modifications)
            recent_files = []
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        mod_time = os.path.getmtime(file_path)
                        if time.time() - mod_time < 120:  # Modified in last 2 minutes
                            recent_files.append(file)
            
            if recent_files:
                print(f"  🟢 Training appears ACTIVE (recent files: {len(recent_files)})")
            else:
                print(f"  🟡 No recent file activity (training may be finished or stuck)")
            
            print("-"*40)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(check_interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--output-dir", default="./output/humar_gref_training", 
                       help="Training output directory to monitor")
    parser.add_argument("--interval", type=int, default=10,
                       help="Check interval in seconds")
    
    args = parser.parse_args()
    
    try:
        monitor_training(args.output_dir, args.interval)
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")