"""
Enhanced Multitask Training Script for ReLA Model
Supports Detection + Segmentation + Pose Estimation
"""

# Add local detectron2 to path
import sys
import os
if os.path.join(os.path.dirname(__file__), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))
if os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2"))

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import json
import time
from datetime import datetime
from functools import reduce
import operator

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.utils.data as torchdata
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import DatasetEvaluators, verify_results
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage

# Import our multitask components
from gres_model import (
    add_maskformer2_config,
    add_refcoco_config
)

# Import HuMAR dataset components
from gres_model.data.datasets.register_humar import register_humar_datasets
from gres_model.data.dataset_mappers.multitask_mapper import MultitaskDatasetMapper

class MultitaskMetrics:
    """Track metrics for all three tasks"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.detection_losses = []
        self.segmentation_losses = []
        self.pose_losses = []
        self.total_losses = []
        self.learning_rates = []
        self.iteration_times = []
    
    def update(self, losses_dict, lr, iter_time):
        """Update metrics from loss dictionary"""
        # Extract task-specific losses
        det_loss = sum(v for k, v in losses_dict.items() if 'det_' in k)
        seg_loss = sum(v for k, v in losses_dict.items() if ('loss_mask' in k or 'loss_dice' in k or 'loss_ce' in k))
        pose_loss = sum(v for k, v in losses_dict.items() if 'kpt_' in k)
        total_loss = losses_dict.get('loss', 0)
        
        self.detection_losses.append(det_loss)
        self.segmentation_losses.append(seg_loss)
        self.pose_losses.append(pose_loss)
        self.total_losses.append(total_loss)
        self.learning_rates.append(lr)
        self.iteration_times.append(iter_time)
    
    def get_recent_averages(self, window=100):
        """Get recent average metrics"""
        def avg_recent(lst):
            return np.mean(lst[-window:]) if lst else 0
        
        return {
            'detection_loss': avg_recent(self.detection_losses),
            'segmentation_loss': avg_recent(self.segmentation_losses),
            'pose_loss': avg_recent(self.pose_losses),
            'total_loss': avg_recent(self.total_losses),
            'learning_rate': avg_recent(self.learning_rates),
            'iter_time': avg_recent(self.iteration_times)
        }
    
    def save_metrics(self, output_dir):
        """Save all metrics to JSON file"""
        metrics_data = {
            'detection_losses': self.detection_losses,
            'segmentation_losses': self.segmentation_losses,
            'pose_losses': self.pose_losses,
            'total_losses': self.total_losses,
            'learning_rates': self.learning_rates,
            'iteration_times': self.iteration_times,
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_file = os.path.join(output_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"📊 Metrics saved to: {metrics_file}")

class EnhancedMultitaskTrainer(DefaultTrainer):
    """Enhanced trainer for multitask learning"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.multitask_metrics = MultitaskMetrics()
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build evaluator for multitask evaluation"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        
        # TODO: Implement multitask evaluator
        # For now, return empty evaluator list
        evaluator_list = []
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        """Build training loader for HuMAR multitask dataset"""
        mapper = MultitaskDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test loader for HuMAR multitask dataset"""
        mapper = MultitaskDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Build optimizer with different learning rates for different components"""
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

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if "text_encoder" in module_name:
                    continue
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                
                # Different learning rates for different heads
                if "detection_head" in module_name:
                    hyperparams["lr"] = cfg.SOLVER.BASE_LR * 1.0  # Same as base
                elif "pose_head" in module_name:
                    hyperparams["lr"] = cfg.SOLVER.BASE_LR * 1.0  # Same as base
                elif "sem_seg_head" in module_name:
                    hyperparams["lr"] = cfg.SOLVER.BASE_LR * 0.5  # Lower for pre-trained seg head

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

        # Add text encoder parameters if they exist
        try:
            hyperparams = copy.copy(defaults)
            text_encoder_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
            if text_encoder_params:
                params.append({"params": text_encoder_params, **hyperparams})
        except AttributeError:
            pass  # No text encoder

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
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
    
    def run_step(self):
        """Override run_step to track multitask metrics"""
        start_time = time.time()
        
        # Standard training step
        super().run_step()
        
        # Track metrics
        iter_time = time.time() - start_time
        storage = get_event_storage()
        losses_dict = {k: v.item() for k, v in storage._history.items() if 'loss' in k}
        lr = self.optimizer.param_groups[0]['lr']
        
        self.multitask_metrics.update(losses_dict, lr, iter_time)
        
        # Show progress every 10 iterations for better visibility
        if self.iter % 10 == 0:
            progress = (self.iter / self.cfg.SOLVER.MAX_ITER) * 100
            print(f"🔄 Iter {self.iter:5d}/{self.cfg.SOLVER.MAX_ITER} ({progress:5.1f}%) - Loss: {losses_dict.get('loss', 0):.4f} - LR: {lr:.6f}")
        
        # Log detailed multitask metrics every 100 iterations
        if self.iter % 100 == 0:
            recent_metrics = self.multitask_metrics.get_recent_averages()
            
            print(f"\n📊 Iteration {self.iter} - Detailed Multitask Metrics:")
            print(f"   🎯 Detection Loss: {recent_metrics['detection_loss']:.4f}")
            print(f"   🎭 Segmentation Loss: {recent_metrics['segmentation_loss']:.4f}")
            print(f"   🤸 Pose Loss: {recent_metrics['pose_loss']:.4f}")
            print(f"   📈 Total Loss: {recent_metrics['total_loss']:.4f}")
            print(f"   ⚡ Learning Rate: {recent_metrics['learning_rate']:.6f}")
            print(f"   ⏱️  Iter Time: {recent_metrics['iter_time']:.3f}s")
            print("-" * 50)
    
    def after_train(self):
        """Save metrics after training"""
        super().after_train()
        self.multitask_metrics.save_metrics(self.cfg.OUTPUT_DIR)
        print("🎉 Enhanced multitask training completed!")

def setup(args):
    """Create configs and perform basic setups"""
    cfg = get_cfg()
    
    # Add configurations
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    
    # Add custom multitask config nodes BEFORE merging YAML file
    from detectron2.config import CfgNode as CN
    
    # Add DETECTION_HEAD config node
    cfg.MODEL.DETECTION_HEAD = CN()
    cfg.MODEL.DETECTION_HEAD.NAME = "AdvancedDetectionHead"
    cfg.MODEL.DETECTION_HEAD.HIDDEN_DIM = 256
    cfg.MODEL.DETECTION_HEAD.NUM_CLASSES = 80
    cfg.MODEL.DETECTION_HEAD.NUM_TRANSFORMER_LAYERS = 3
    cfg.MODEL.DETECTION_HEAD.NUM_ATTENTION_HEADS = 8
    cfg.MODEL.DETECTION_HEAD.DROPOUT = 0.1
    cfg.MODEL.DETECTION_HEAD.CONFIDENCE_THRESHOLD = 0.5
    cfg.MODEL.DETECTION_HEAD.NMS_THRESHOLD = 0.6
    
    # Add POSE_HEAD config node
    cfg.MODEL.POSE_HEAD = CN()
    cfg.MODEL.POSE_HEAD.NAME = "AdvancedPoseHead"
    cfg.MODEL.POSE_HEAD.HIDDEN_DIM = 256
    cfg.MODEL.POSE_HEAD.NUM_KEYPOINTS = 17
    cfg.MODEL.POSE_HEAD.NUM_TRANSFORMER_LAYERS = 4
    cfg.MODEL.POSE_HEAD.NUM_ATTENTION_HEADS = 8
    cfg.MODEL.POSE_HEAD.DROPOUT = 0.1
    cfg.MODEL.POSE_HEAD.KEYPOINT_THRESHOLD = 0.3
    cfg.MODEL.POSE_HEAD.POSE_QUALITY_THRESHOLD = 0.5
    
    # Add MULTITASK config node
    cfg.MODEL.MULTITASK = CN()
    cfg.MODEL.MULTITASK.DETECTION_LOSS_WEIGHT = 1.0
    cfg.MODEL.MULTITASK.SEGMENTATION_LOSS_WEIGHT = 1.0
    cfg.MODEL.MULTITASK.POSE_LOSS_WEIGHT = 1.0
    cfg.MODEL.MULTITASK.QUALITY_LOSS_WEIGHT = 0.1
    cfg.MODEL.MULTITASK.CONSISTENCY_LOSS_WEIGHT = 0.1
    
    # NOW merge from file (YAML config can override these defaults)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="multitask_rela")
    
    # Register HuMAR datasets
    register_humar_datasets()
    
    return cfg

def main(args):
    cfg = setup(args)
    
    # Suppress verbose keypoint validation warnings
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    print("\n" + "="*60)
    print("🚀 ENHANCED MULTITASK ReLA TRAINING")
    print("="*60)
    print(f"📁 Output Directory: {cfg.OUTPUT_DIR}")
    print(f"🎯 Tasks: Detection + Segmentation + Pose Estimation")
    print(f"📊 Dataset: HuMAR Multitask")
    print(f"⚙️  Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"🎛️  Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"📈 Base Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f"🔄 Evaluation Every: {cfg.TEST.EVAL_PERIOD} iterations")
    print("="*60)
    
    # Suppress keypoint validation warnings by temporarily redirecting stderr
    import sys
    import io
    
    # Create a filter for unwanted messages
    class KeypointWarningFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
            self.buffer = ""
            
        def write(self, s):
            # Filter out keypoint validation messages
            if "Expected 51 keypoint values" not in s:
                self.original_stderr.write(s)
            
        def flush(self):
            self.original_stderr.flush()
    
    # Apply filter during dataset loading
    original_stderr = sys.stderr
    sys.stderr = KeypointWarningFilter(original_stderr)
    
    print("📋 Loading and validating dataset...")
    
    try:
        if args.eval_only:
            model = EnhancedMultitaskTrainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            print("🔍 Starting evaluation...")
            res = EnhancedMultitaskTrainer.test(cfg, model)
            if comm.is_main_process():
                verify_results(cfg, res)
            return res

        print("🏗️  Building trainer...")
        trainer = EnhancedMultitaskTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        
        print("🎯 TRAINING STARTED!")
        print("="*60)
        return trainer.train()
        
    finally:
        # Restore original stderr
        sys.stderr = original_stderr

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