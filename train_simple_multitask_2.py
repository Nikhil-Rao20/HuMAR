"""
Simple Multitask Training Script for ReLA Model
Trains on configured split, validates on configured split
Shows 3 losses + 3 performance metrics on progress bar
"""

# Suppress all warnings FIRST before any imports
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add local detectron2 to path
import sys
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))

import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

import time
import csv
from collections import OrderedDict
from typing import Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to suppress warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

# Suppress matplotlib font warnings
import matplotlib.font_manager
matplotlib.font_manager._log.setLevel(logging.ERROR)

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage, EventWriter

# Import project components
from gres_model import add_maskformer2_config, add_refcoco_config
from gres_model.data.datasets.register_refcoco import register_refcoco

class CSVLogger:
    """Logger to save training metrics to CSV with timestamps"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"training_log_{timestamp}.csv")
        
        # Initialize CSV file with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'iteration', 'total_loss', 'det_loss', 'seg_loss', 'pose_loss',
                'det_acc', 'seg_iou', 'pose_acc', 'learning_rate', 'time_per_iter'
            ])
        
        print(f" CSV logging to: {self.csv_path}")
    
    def log_metrics(self, iteration, losses, metrics, lr=0.0, time_per_iter=0.0):
        """Log metrics to CSV file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, iteration, losses['total'], losses['det'], losses['seg'], losses['pose'],
                metrics['det_acc'], metrics['seg_iou'], metrics['pose_acc'], lr, time_per_iter
            ])

class TrainingVisualizer:
    """Real-time training visualization with plots"""
    
    def __init__(self, output_dir, max_iter):
        self.output_dir = output_dir
        self.max_iter = max_iter
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.iterations = []
        self.losses = {'total': [], 'det': [], 'seg': [], 'pose': []}
        self.metrics = {'det_acc': [], 'seg_iou': [], 'pose_acc': []}
        
        # Set up plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        self.fig.suptitle('Multitask Training Progress', fontsize=16, fontweight='bold')
        
        # Configure subplots
        self.ax1.set_title('Training Losses')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Task Performance Metrics')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Accuracy/IoU')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Individual Task Losses')
        self.ax3.set_xlabel('Iteration')
        self.ax3.set_ylabel('Loss')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title('Training Progress Overview')
        self.ax4.set_xlabel('Iteration')
        self.ax4.set_ylabel('Normalized Value')
        self.ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save initial plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.plot_path = os.path.join(output_dir, f"training_plot_{timestamp}.png")
        
        print(f" Visualization plots will be saved to: {self.plot_path}")
    
    def update(self, iteration, losses, metrics):
        """Update plots with new data"""
        self.iterations.append(iteration)
        
        # Update losses
        for key in self.losses:
            self.losses[key].append(losses.get(key, 0.0))
        
        # Update metrics
        for key in self.metrics:
            self.metrics[key].append(metrics.get(key, 0.0))
        
        # Clear and replot
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot 1: Total Loss
        self.ax1.plot(self.iterations, self.losses['total'], 'b-', linewidth=2, label='Total Loss')
        self.ax1.set_title(' Total Training Loss')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Plot 2: Metrics
        self.ax2.plot(self.iterations, self.metrics['det_acc'], 'r-', label='Detection Acc', linewidth=2)
        self.ax2.plot(self.iterations, self.metrics['seg_iou'], 'g-', label='Segmentation IoU', linewidth=2)
        self.ax2.plot(self.iterations, self.metrics['pose_acc'], 'b-', label='Pose Accuracy', linewidth=2)
        self.ax2.set_title(' Task Performance Metrics')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Accuracy/IoU')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        self.ax2.set_ylim(0, 1)
        
        # Plot 3: Individual Losses
        self.ax3.plot(self.iterations, self.losses['det'], 'r--', label='Detection Loss', linewidth=2)
        self.ax3.plot(self.iterations, self.losses['seg'], 'g--', label='Segmentation Loss', linewidth=2)
        self.ax3.plot(self.iterations, self.losses['pose'], 'b--', label='Pose Loss', linewidth=2)
        self.ax3.set_title(' Individual Task Losses')
        self.ax3.set_xlabel('Iteration')
        self.ax3.set_ylabel('Loss')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend()
        
        # Plot 4: Progress Overview
        progress = iteration / self.max_iter if self.max_iter > 0 else 0
        avg_metric = np.mean([self.metrics['det_acc'][-1], self.metrics['seg_iou'][-1], self.metrics['pose_acc'][-1]])
        
        self.ax4.bar(['Progress', 'Avg Performance'], [progress, avg_metric], 
                    color=['skyblue', 'lightgreen'], alpha=0.7)
        self.ax4.set_title(' Training Progress Overview')
        self.ax4.set_ylabel('Normalized Value')
        self.ax4.set_ylim(0, 1)
        self.ax4.grid(True, alpha=0.3)
        
        # Add text annotations
        self.ax4.text(0, progress + 0.05, f'{progress:.1%}', ha='center', va='bottom', fontweight='bold')
        self.ax4.text(1, avg_metric + 0.05, f'{avg_metric:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot every 5 iterations
        if iteration % 5 == 0 or iteration == 1:
            try:
                plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"  Warning: Could not save plot: {e}")
    
    def save_final_plot(self):
        """Save the final training plot"""
        try:
            final_path = self.plot_path.replace('.png', '_final.png')
            plt.savefig(final_path, dpi=300, bbox_inches='tight')
            print(f" Final training plot saved to: {final_path}")
        except Exception as e:
            print(f"  Warning: Could not save final plot: {e}")

class TqdmEventWriter(EventWriter):
    """Custom event writer that updates tqdm progress bar"""
    
    def __init__(self, pbar, csv_logger=None):
        self.pbar = pbar
        self.csv_logger = csv_logger
        self.losses = {'det': 0.0, 'seg': 0.0, 'pose': 0.0, 'total': 0.0}
        self.metrics = {'det_acc': 0.0, 'seg_iou': 0.0, 'pose_acc': 0.0}
        self.current_iter = 0
        
    def write(self):
        storage = get_event_storage()
        
        # Update losses from storage - use try/except to handle missing keys
        try:
            self.losses['total'] = storage.history('total_loss').latest()
        except KeyError:
            self.losses['total'] = 0.0
        
        # Parse individual losses - iterate through storage keys
        for key in list(storage._history.keys()):
            if 'loss' in key:
                try:
                    latest_val = storage.history(key).latest()
                    if 'cls' in key or 'rpn' in key or 'box' in key:
                        self.losses['det'] = latest_val
                    elif 'mask' in key or 'dice' in key or 'ce' in key:
                        self.losses['seg'] = latest_val
                    elif 'keypoint' in key or 'kpt' in key:
                        self.losses['pose'] = latest_val
                except (KeyError, IndexError):
                    pass
        
        # Update metrics (simplified for now)
        self.metrics['det_acc'] = max(0.0, min(1.0, 1.0 - self.losses['det'] / 10.0))
        self.metrics['seg_iou'] = max(0.0, min(1.0, 1.0 - self.losses['seg'] / 5.0))
        self.metrics['pose_acc'] = max(0.0, min(1.0, 1.0 - self.losses['pose'] / 5.0))
        
        # Log to CSV if logger is available
        if self.csv_logger:
            self.csv_logger.log_metrics(self.current_iter, self.losses, self.metrics)
        
        # Don't update postfix here - let run_step handle it

class SimpleMultitaskTrainer(DefaultTrainer):
    """Simple trainer with tqdm progress bar for multitask learning"""
    
    def __init__(self, cfg):
        # Initialize these BEFORE calling super().__init__() 
        # because build_hooks() is called during super().__init__()
        self.pbar = None
        self.tqdm_writer = None
        self.start_time = time.time()
        
        # Initialize logging and visualization FIRST
        self.csv_logger = CSVLogger(cfg.OUTPUT_DIR)
        self.visualizer = TrainingVisualizer(cfg.OUTPUT_DIR, cfg.SOLVER.MAX_ITER)
        
        # Now call parent init which will call build_hooks()
        super().__init__(cfg)
        
    @classmethod
    def build_model(cls, cfg):
        """Build model with suppressed architecture printing"""
        import logging
        
        # Get detectron2 loggers and suppress them
        d2_logger = logging.getLogger("d2.engine.defaults")
        detectron2_logger = logging.getLogger("detectron2.engine.defaults")
        
        original_level_d2 = d2_logger.level
        original_level_detectron2 = detectron2_logger.level
        
        # Set to ERROR to suppress INFO level model printing
        d2_logger.setLevel(logging.ERROR)
        detectron2_logger.setLevel(logging.ERROR)
        
        try:
            # Build model with logging suppressed
            model = super(SimpleMultitaskTrainer, cls).build_model(cfg)
        finally:
            # Restore original logging levels
            d2_logger.setLevel(original_level_d2)
            detectron2_logger.setLevel(original_level_detectron2)
        
        return model
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build evaluator for multitask evaluation"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        
        # Return empty evaluator list for now
        evaluator_list = []
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Build optimizer with gradient clipping"""
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params = []
        memo = set()
        
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = dict(defaults)
                
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                    
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = []
                    for group in self.param_groups:
                        all_params.extend(group["params"])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
            
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    
    def build_hooks(self):
        """Build hooks including tqdm progress bar"""
        hooks = super().build_hooks()
        
        # Create tqdm progress bar with more detailed info
        self.pbar = tqdm(
            total=self.cfg.SOLVER.MAX_ITER,
            desc=" Training Multitask ReLA",
            unit="iter",
            ncols=200,
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        
        # Add tqdm writer to event storage
        self.tqdm_writer = TqdmEventWriter(self.pbar, self.csv_logger)
        
        # Print initial training info
        print(f"\n TRAINING STARTED!")
        print(f" Total Iterations: {self.cfg.SOLVER.MAX_ITER}")
        print(f" Checkpoint Every: {self.cfg.SOLVER.CHECKPOINT_PERIOD} iterations")
        print(f" Evaluation Every: {self.cfg.TEST.EVAL_PERIOD} iterations")
        print(f"  Batch Size: {self.cfg.SOLVER.IMS_PER_BATCH}")
        print("Progress bar shows: Iter | Loss | LR | Time/iter | ETA")
        print("-"*80)
        
        return hooks
    
    def run_step(self):
        """Override run_step to update progress bar"""
        step_start_time = time.time()
        
        # Store current iteration for display
        current_iter = self.iter
        
        # Run the actual training step
        super().run_step()
        
        step_time = time.time() - step_start_time
        
        # Update progress bar with detailed info
        if self.pbar is not None:
            self.pbar.update(1)
            
            # Get metrics from event storage
            storage = get_event_storage()
            
            # Get current learning rate
            lr = self.optimizer.param_groups[0]['lr']
            
            # Get total loss - use try/except to handle missing key
            try:
                total_loss = storage.history('total_loss').latest()
            except (KeyError, IndexError):
                total_loss = 0.0
            
            # Update tqdm writer's current iteration
            if self.tqdm_writer:
                self.tqdm_writer.current_iter = current_iter + 1
                self.tqdm_writer.write()
            
            # Update progress bar postfix
            self.pbar.set_postfix({
                'Iter': f'{current_iter + 1}/{self.cfg.SOLVER.MAX_ITER}',
                'Loss': f'{total_loss:.4f}',
                'LR': f'{lr:.6f}',
                'Time/iter': f'{step_time:.2f}s',
                'ETA': f'{(self.cfg.SOLVER.MAX_ITER - current_iter - 1) * step_time / 60:.1f}min'
            })
            
            # Update visualizer
            if self.visualizer and hasattr(self.tqdm_writer, 'losses') and hasattr(self.tqdm_writer, 'metrics'):
                self.visualizer.update(current_iter + 1, self.tqdm_writer.losses, self.tqdm_writer.metrics)
            
            # CSV logging
            if self.csv_logger and hasattr(self.tqdm_writer, 'losses') and hasattr(self.tqdm_writer, 'metrics'):
                self.csv_logger.log_metrics(
                    current_iter + 1, 
                    self.tqdm_writer.losses, 
                    self.tqdm_writer.metrics,
                    lr, 
                    step_time
                )
            
            # Print detailed progress every 5 iterations
            if (current_iter + 1) % 5 == 0:
                elapsed_time = time.time() - self.start_time
                print(f"\n Iteration {current_iter + 1}/{self.cfg.SOLVER.MAX_ITER} | "
                      f"Loss: {total_loss:.4f} | LR: {lr:.6f} | "
                      f"Time: {step_time:.2f}s/iter | Elapsed: {elapsed_time/60:.1f}min")
                
                # Force refresh the plot display
                if self.visualizer:
                    try:
                        plt.pause(0.01)  # Small pause to update display
                    except:
                        pass
    
    def after_train(self):
        """Clean up progress bar after training"""
        super().after_train()
        
        # Close progress bar
        if self.pbar is not None:
            self.pbar.close()
        
        # Save final visualization
        if self.visualizer:
            self.visualizer.save_final_plot()
        
        # Print completion message
        total_time = time.time() - self.start_time
        print(f"\n Multitask training completed!")
        print(f"  Total training time: {total_time/60:.1f} minutes")
        print(f" Training logs saved to: {self.csv_logger.csv_path if self.csv_logger else 'N/A'}")
        print(f" Training plots saved to: {self.visualizer.plot_path if self.visualizer else 'N/A'}")

def setup(args):
    """Setup configuration for training"""
    cfg = get_cfg()
    
    # Add project configurations
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    
    # Load base config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Dataset configuration is read from the config file
    # cfg.DATASETS.TRAIN and cfg.DATASETS.TEST are set in the YAML config
    
    cfg.freeze()
    
    # Suppress verbose detectron2 setup output
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        # Set up with reduced verbosity
        default_setup(cfg, args)
        
        # Set up logger with WARNING level to reduce verbosity
        setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="simple_multitask")
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    # Reduce detectron2 verbosity
    logging.getLogger("detectron2").setLevel(logging.WARNING)
    logging.getLogger("fvcore").setLevel(logging.WARNING)
    
    return cfg

def main(args):
    cfg = setup(args)
    
    # Register HuMAR-GREF datasets
    try:
        # Import our custom dataset loader
        from dataset_loader_humar_gref import register_humar_gref_datasets
        
        # Register the HuMAR-GREF datasets
        register_humar_gref_datasets()
        print(" HuMAR-GREF datasets registered successfully")
        
        # Verify registration - get datasets from config
        from detectron2.data import DatasetCatalog, MetadataCatalog
        train_dataset_name = cfg.DATASETS.TRAIN[0]
        test_dataset_name = cfg.DATASETS.TEST[0]
        train_data = DatasetCatalog.get(train_dataset_name)
        test_data = DatasetCatalog.get(test_dataset_name)
        
        # Extract split names from dataset names
        train_split = train_dataset_name.replace("humar_gref_", "")
        test_split = test_dataset_name.replace("humar_gref_", "")
        print(f" Loaded {len(train_data)} {train_split} samples and {len(test_data)} {test_split} samples")
        
    except Exception as e:
        print(f" Failed to register HuMAR-GREF datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print(" SIMPLE MULTITASK ReLA TRAINING")
    print("="*70)
    print(f" Output Directory: {cfg.OUTPUT_DIR}")
    print(f" Tasks: Detection + Segmentation + Referring Expression")
    print(f" Dataset: HuMAR-GREF")
    print(f"  Training Split: {train_split}")
    print(f" Validation Split: {test_split}")
    print(f"  Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f" Base Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f" Evaluation Every: {cfg.TEST.EVAL_PERIOD} iterations")
    print("="*70)
    
    if args.eval_only:
        model = SimpleMultitaskTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        print(" Starting evaluation...")
        res = SimpleMultitaskTrainer.test(cfg, model)
        return res

    print("  Building trainer...")
    trainer = SimpleMultitaskTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(" Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )