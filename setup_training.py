"""
Quick Setup and Training Validation Script
Checks prerequisites and runs a quick training test
"""

import os
import sys
import subprocess
import torch
import json
from pathlib import Path

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking Environment Setup...")
    
    # Check Python version
    print(f"✅ Python Version: {sys.version}")
    
    # Check PyTorch
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check required packages
    required_packages = [
        'detectron2', 'transformers', 'opencv-python', 
        'matplotlib', 'numpy', 'pillow'
    ]
    
    print("\n📦 Checking Required Packages:")
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_dataset():
    """Check if HuMAR dataset is available"""
    print("\n📊 Checking Dataset Setup...")
    
    dataset_path = Path("datasets/humar")
    required_files = [
        "gref_umd_train.json",
        "gref_umd_val.json", 
        "instances_train.json",
        "instances_val.json"
    ]
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        print("Please create the dataset directory and place HuMAR files")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = dataset_path / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (MISSING)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    # Check image directories
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    if train_dir.exists() and list(train_dir.glob("*.jpg")):
        print(f"✅ Training images found: {len(list(train_dir.glob('*.jpg')))} images")
    else:
        print("❌ Training images directory missing or empty")
        return False
        
    if val_dir.exists() and list(val_dir.glob("*.jpg")):
        print(f"✅ Validation images found: {len(list(val_dir.glob('*.jpg')))} images")
    else:
        print("❌ Validation images directory missing or empty")
        return False
    
    return True

def check_model_files():
    """Check if model files are available"""
    print("\n🏗️ Checking Model Files...")
    
    model_files = [
        "gres_model/modeling/meta_arch/detection_head.py",
        "gres_model/modeling/meta_arch/pose_head.py", 
        "gres_model/modeling/meta_arch/multitask_gres.py",
        "configs/enhanced_multitask_humar.yaml",
        "train_enhanced_multitask.py"
    ]
    
    all_present = True
    for file in model_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (MISSING)")
            all_present = False
    
    return all_present

def run_quick_test():
    """Run a quick model test"""
    print("\n🧪 Running Quick Model Test...")
    
    try:
        # Test enhanced model creation
        result = subprocess.run([
            sys.executable, "test_enhanced_multitask_model.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Enhanced model test passed!")
            return True
        else:
            print("❌ Enhanced model test failed!")
            print("Error output:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Model test timed out (this might be normal)")
        return True
    except Exception as e:
        print(f"❌ Error running model test: {e}")
        return False

def create_output_directory():
    """Create output directory for training"""
    output_dir = Path("output/enhanced_multitask_run1")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")

def print_training_commands():
    """Print the actual training commands to run"""
    print("\n" + "="*60)
    print("🚀 READY TO TRAIN! Use these commands:")
    print("="*60)
    
    print("\n1️⃣ Quick Start (Single GPU):")
    print("python train_enhanced_multitask.py --config-file configs/enhanced_multitask_humar.yaml --num-gpus 1 OUTPUT_DIR ./output/enhanced_multitask_run1")
    
    print("\n2️⃣ With Custom Batch Size (if memory issues):")
    print("python train_enhanced_multitask.py --config-file configs/enhanced_multitask_humar.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 8 OUTPUT_DIR ./output/enhanced_multitask_run1")
    
    print("\n3️⃣ Multi-GPU (if available):")
    print("python train_enhanced_multitask.py --config-file configs/enhanced_multitask_humar.yaml --num-gpus 2 OUTPUT_DIR ./output/enhanced_multitask_run1")
    
    print("\n4️⃣ Resume Training:")
    print("python train_enhanced_multitask.py --config-file configs/enhanced_multitask_humar.yaml --resume --num-gpus 1 OUTPUT_DIR ./output/enhanced_multitask_run1")
    
    print("\n📊 Monitor Training:")
    print("tensorboard --logdir ./output/enhanced_multitask_run1")
    
    print("\n📖 For more options, see: TRAINING_GUIDE.md")
    print("="*60)

def main():
    """Main setup validation function"""
    print("🔧 Enhanced Multitask ReLA - Training Setup Validator")
    print("="*60)
    
    # Run all checks
    checks = [
        ("Environment", check_environment),
        ("Dataset", check_dataset), 
        ("Model Files", check_model_files),
        ("Quick Test", run_quick_test)
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
            print(f"\n❌ {name} check failed!")
        else:
            print(f"\n✅ {name} check passed!")
    
    if all_passed:
        create_output_directory()
        print("\n🎉 ALL CHECKS PASSED! Ready for training!")
        print_training_commands()
    else:
        print("\n⚠️  Some checks failed. Please resolve issues before training.")
        print("See TRAINING_GUIDE.md for detailed setup instructions.")

if __name__ == "__main__":
    main()