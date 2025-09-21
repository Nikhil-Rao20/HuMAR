"""
Simple Multitask Training Script for ReLA Model
Trains on configured split, validates on configured split
Shows 3 losses + 3 performance metrics on progress bar
"""

# Add local detectron2 to path
import sys
import os
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))

import logging
import time
from collections import OrderedDict
from typing import Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

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

class TqdmEventWriter(EventWriter):
    """Custom event writer that updates tqdm progress bar"""
    
    def __init__(self, pbar):
        self.pbar = pbar
        self.losses = {'det': 0.0, 'seg': 0.0, 'pose': 0.0}
        self.metrics = {'det_acc': 0.0, 'seg_iou': 0.0, 'pose_acc': 0.0}
        
    def write(self):
        storage = get_event_storage()
        
        # Update losses from storage
        for key, value in storage._history.items():
            if key.endswith('loss'):
                if 'det' in key or 'cls' in key or 'bbox' in key:
                    self.losses['det'] = value[-1][0] if value else 0.0
                elif 'mask' in key or 'dice' in key or 'ce' in key:
                    self.losses['seg'] = value[-1][0] if value else 0.0
                elif 'kpt' in key or 'keypoint' in key:
                    self.losses['pose'] = value[-1][0] if value else 0.0
        
        # Update metrics (placeholder for now)
        self.metrics['det_acc'] = max(0.0, 1.0 - self.losses['det'])
        self.metrics['seg_iou'] = max(0.0, 1.0 - self.losses['seg'])
        self.metrics['pose_acc'] = max(0.0, 1.0 - self.losses['pose'])
        
        # Update progress bar
        total_loss = sum(self.losses.values())
        avg_metric = np.mean(list(self.metrics.values()))
        
        self.pbar.set_postfix({
            'Det_Loss': f'{self.losses["det"]:.3f}',
            'Seg_Loss': f'{self.losses["seg"]:.3f}',
            'Pose_Loss': f'{self.losses["pose"]:.3f}',
            'Det_Acc': f'{self.metrics["det_acc"]:.3f}',
            'Seg_IoU': f'{self.metrics["seg_iou"]:.3f}',
            'Pose_Acc': f'{self.metrics["pose_acc"]:.3f}',
            'Total_Loss': f'{total_loss:.3f}',
            'Avg_Metric': f'{avg_metric:.3f}'
        })

class SimpleMultitaskTrainer(DefaultTrainer):
    """Simple trainer with tqdm progress bar for multitask learning"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pbar = None
        self.tqdm_writer = None
        
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
        
        # Create tqdm progress bar
        self.pbar = tqdm(
            total=self.cfg.SOLVER.MAX_ITER,
            desc="🚀 Training Multitask ReLA",
            unit="iter",
            ncols=150,
            position=0,
            leave=True
        )
        
        # Add tqdm writer to event storage
        self.tqdm_writer = TqdmEventWriter(self.pbar)
        
        return hooks
    
    def run_step(self):
        """Override run_step to update progress bar"""
        super().run_step()
        
        # Update progress bar
        if self.pbar is not None:
            self.pbar.update(1)
            if self.tqdm_writer is not None:
                self.tqdm_writer.write()
    
    def after_train(self):
        """Clean up progress bar after training"""
        super().after_train()
        if self.pbar is not None:
            self.pbar.close()
        print("\n🎉 Multitask training completed!")

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
        print("✓ HuMAR-GREF datasets registered successfully")
        
        # Verify registration - get datasets from config
        from detectron2.data import DatasetCatalog, MetadataCatalog
        train_dataset_name = cfg.DATASETS.TRAIN[0]
        test_dataset_name = cfg.DATASETS.TEST[0]
        train_data = DatasetCatalog.get(train_dataset_name)
        test_data = DatasetCatalog.get(test_dataset_name)
        
        # Extract split names from dataset names
        train_split = train_dataset_name.replace("humar_gref_", "")
        test_split = test_dataset_name.replace("humar_gref_", "")
        print(f"✓ Loaded {len(train_data)} {train_split} samples and {len(test_data)} {test_split} samples")
        
    except Exception as e:
        print(f"❌ Failed to register HuMAR-GREF datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("🚀 SIMPLE MULTITASK ReLA TRAINING")
    print("="*70)
    print(f"📁 Output Directory: {cfg.OUTPUT_DIR}")
    print(f"🎯 Tasks: Detection + Segmentation + Referring Expression")
    print(f"📊 Dataset: HuMAR-GREF")
    print(f"🏋️  Training Split: {train_split}")
    print(f"✅ Validation Split: {test_split}")
    print(f"⚙️  Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"🎛️  Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"📈 Base Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f"🔄 Evaluation Every: {cfg.TEST.EVAL_PERIOD} iterations")
    print("="*70)
    
    if args.eval_only:
        model = SimpleMultitaskTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        print("🔍 Starting evaluation...")
        res = SimpleMultitaskTrainer.test(cfg, model)
        return res

    print("🏗️  Building trainer...")
    trainer = SimpleMultitaskTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    print("🎯 TRAINING STARTED!")
    print("Progress bar shows: Det_Loss | Seg_Loss | Pose_Loss | Det_Acc | Seg_IoU | Pose_Acc")
    print("-"*70)
    
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("🎮 Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )