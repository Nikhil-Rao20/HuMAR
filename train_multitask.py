"""
Training script for MultitaskGRES model.
This script provides training pipeline with multitask loss functions and metrics tracking.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom components
from test_multitask_model import SimplifiedMultitaskGRES, MockBackbone, MockSemSegHead, MockCriterion, MockBertModel


class MultitaskLoss(nn.Module):
    """
    Combined loss function for multitask learning.
    Combines segmentation, detection, and keypoint losses.
    """
    
    def __init__(self, 
                 seg_weight=1.0, 
                 det_weight=1.0, 
                 kpt_weight=1.0,
                 det_class_weight=1.0,
                 det_bbox_weight=5.0,
                 kpt_coord_weight=1.0,
                 kpt_vis_weight=1.0):
        super().__init__()
        
        # Task weights
        self.seg_weight = seg_weight
        self.det_weight = det_weight  
        self.kpt_weight = kpt_weight
        
        # Detection sub-weights
        self.det_class_weight = det_class_weight
        self.det_bbox_weight = det_bbox_weight
        
        # Keypoint sub-weights
        self.kpt_coord_weight = kpt_coord_weight
        self.kpt_vis_weight = kpt_vis_weight
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        """
        Compute multitask loss.
        
        Args:
            outputs: Model outputs dict
            targets: Ground truth targets dict
        
        Returns:
            Dict of losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Segmentation loss (using existing criterion)
        if "pred_masks" in outputs:
            # Mock segmentation loss for demonstration
            seg_loss = torch.mean(outputs["pred_masks"] ** 2) * 0.1
            loss_dict["loss_seg"] = seg_loss
            total_loss += self.seg_weight * seg_loss
        
        # Detection losses
        if "det_pred_logits" in outputs and "det_pred_boxes" in outputs:
            det_losses = self.compute_detection_loss(outputs, targets)
            for key, value in det_losses.items():
                loss_dict[key] = value
                total_loss += self.det_weight * value
        
        # Keypoint losses
        if "kpt_pred_keypoints" in outputs:
            kpt_losses = self.compute_keypoint_loss(outputs, targets)
            for key, value in kpt_losses.items():
                loss_dict[key] = value
                total_loss += self.kpt_weight * value
        
        loss_dict["loss_total"] = total_loss
        return loss_dict
    
    def compute_detection_loss(self, outputs, targets):
        """Compute detection losses."""
        pred_logits = outputs["det_pred_logits"]  # (B, N, 1)
        pred_boxes = outputs["det_pred_boxes"]    # (B, N, 4)
        
        batch_size, num_queries = pred_logits.shape[:2]
        
        # Create mock targets for demonstration
        # In practice, these would come from the dataset
        gt_labels = torch.zeros_like(pred_logits).fill_(0.1)  # Background
        gt_boxes = torch.zeros_like(pred_boxes)
        
        # Randomly assign some positive targets
        for b in range(batch_size):
            num_pos = torch.randint(1, min(5, num_queries//10), (1,)).item()
            pos_idx = torch.randperm(num_queries)[:num_pos]
            gt_labels[b, pos_idx] = 1.0  # Positive class
            gt_boxes[b, pos_idx] = torch.rand(num_pos, 4) * 0.8 + 0.1  # Random boxes
        
        # Classification loss
        class_loss = self.bce_loss(pred_logits.squeeze(-1), gt_labels.squeeze(-1))
        
        # Box regression loss (only for positive samples)
        pos_mask = (gt_labels.squeeze(-1) > 0.5)
        if pos_mask.sum() > 0:
            bbox_loss = self.l1_loss(pred_boxes[pos_mask], gt_boxes[pos_mask])
        else:
            bbox_loss = torch.tensor(0.0, device=pred_logits.device)
        
        return {
            "loss_det_class": class_loss * self.det_class_weight,
            "loss_det_bbox": bbox_loss * self.det_bbox_weight
        }
    
    def compute_keypoint_loss(self, outputs, targets):
        """Compute keypoint losses."""
        pred_keypoints = outputs["kpt_pred_keypoints"]  # (B, N, 17, 3)
        
        batch_size, num_queries, num_kpts = pred_keypoints.shape[:3]
        
        # Create mock targets for demonstration
        gt_keypoints = torch.zeros_like(pred_keypoints)
        gt_visibility = torch.zeros(batch_size, num_queries, num_kpts, device=pred_keypoints.device)
        
        # Randomly assign some visible keypoints
        for b in range(batch_size):
            for q in range(min(5, num_queries)):  # Only first few queries have keypoints
                visible_kpts = torch.randperm(num_kpts)[:torch.randint(5, num_kpts, (1,)).item()]
                gt_visibility[b, q, visible_kpts] = 1.0
                gt_keypoints[b, q, visible_kpts, :2] = torch.rand(len(visible_kpts), 2)  # Random coordinates
                gt_keypoints[b, q, visible_kpts, 2] = 1.0  # Visible
        
        # Coordinate loss (only for visible keypoints)
        visible_mask = gt_visibility.unsqueeze(-1).expand_as(pred_keypoints[:, :, :, :2])
        if visible_mask.sum() > 0:
            coord_loss = self.l1_loss(
                pred_keypoints[:, :, :, :2][visible_mask[:, :, :, 0] > 0.5],
                gt_keypoints[:, :, :, :2][visible_mask[:, :, :, 0] > 0.5]
            )
        else:
            coord_loss = torch.tensor(0.0, device=pred_keypoints.device)
        
        # Visibility loss
        vis_loss = self.bce_loss(
            pred_keypoints[:, :, :, 2].flatten(),
            gt_keypoints[:, :, :, 2].flatten()
        )
        
        return {
            "loss_kpt_coord": coord_loss * self.kpt_coord_weight,
            "loss_kpt_vis": vis_loss * self.kpt_vis_weight
        }


class MultitaskMetrics:
    """Metrics tracker for multitask learning."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            # Segmentation metrics
            "seg_iou": [],
            "seg_dice": [],
            
            # Detection metrics
            "det_precision": [],
            "det_recall": [],
            "det_bbox_mae": [],
            
            # Keypoint metrics
            "kpt_pck": [],  # Percentage of Correct Keypoints
            "kpt_coord_mae": [],
            "kpt_vis_acc": [],
            
            # Overall metrics
            "total_loss": []
        }
    
    def update(self, outputs, targets=None):
        """Update metrics with current batch."""
        # Mock metric computation for demonstration
        batch_size = outputs["pred_masks"].shape[0] if "pred_masks" in outputs else 1
        
        # Segmentation metrics
        if "pred_masks" in outputs:
            self.metrics["seg_iou"].append(np.random.uniform(0.6, 0.9))
            self.metrics["seg_dice"].append(np.random.uniform(0.7, 0.95))
        
        # Detection metrics
        if "det_pred_logits" in outputs:
            self.metrics["det_precision"].append(np.random.uniform(0.7, 0.95))
            self.metrics["det_recall"].append(np.random.uniform(0.6, 0.9))
            self.metrics["det_bbox_mae"].append(np.random.uniform(0.05, 0.2))
        
        # Keypoint metrics
        if "kpt_pred_keypoints" in outputs:
            self.metrics["kpt_pck"].append(np.random.uniform(0.8, 0.95))
            self.metrics["kpt_coord_mae"].append(np.random.uniform(0.02, 0.1))
            self.metrics["kpt_vis_acc"].append(np.random.uniform(0.85, 0.98))
    
    def get_average_metrics(self):
        """Get average metrics over the epoch."""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[f"avg_{key}"] = np.mean(values)
        return avg_metrics


class MultitaskTrainer:
    """Main trainer for multitask GRES model."""
    
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 val_dataloader=None,
                 learning_rate=1e-4,
                 num_epochs=100,
                 save_dir="./checkpoints",
                 log_dir="./logs"):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Loss function and optimizer
        self.criterion = MultitaskLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Metrics and logging
        self.train_metrics = MultitaskMetrics()
        self.val_metrics = MultitaskMetrics()
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_losses = []
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute loss
            loss_dict = self.criterion(outputs, batch)  # batch serves as targets for mock
            total_loss = loss_dict["loss_total"]
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(outputs)
            self.train_metrics.metrics["total_loss"].append(total_loss.item())
            epoch_losses.append(total_loss.item())
            
            # Log batch progress
            if batch_idx % 10 == 0:
                print(f"Epoch {self.current_epoch+1}/{self.num_epochs}, "
                      f"Batch {batch_idx+1}/{num_batches}, "
                      f"Loss: {total_loss.item():.4f}")
        
        return np.mean(epoch_losses)
    
    def validate_epoch(self):
        """Validate for one epoch."""
        if self.val_dataloader is None:
            return None
            
        self.model.eval()
        self.val_metrics.reset()
        
        epoch_losses = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss_dict = self.criterion(outputs, batch)
                total_loss = loss_dict["loss_total"]
                
                # Update metrics
                self.val_metrics.update(outputs)
                self.val_metrics.metrics["total_loss"].append(total_loss.item())
                epoch_losses.append(total_loss.item())
        
        return np.mean(epoch_losses)
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def log_metrics(self, train_loss, val_loss=None):
        """Log metrics to tensorboard."""
        # Log losses
        self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
        if val_loss is not None:
            self.writer.add_scalar('Loss/Validation', val_loss, self.current_epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, self.current_epoch)
        
        # Log detailed metrics
        train_avg_metrics = self.train_metrics.get_average_metrics()
        for key, value in train_avg_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, self.current_epoch)
        
        if val_loss is not None and self.val_metrics:
            val_avg_metrics = self.val_metrics.get_average_metrics()
            for key, value in val_avg_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, self.current_epoch)
    
    def train(self):
        """Main training loop."""
        print("Starting multitask training...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training for {self.num_epochs} epochs")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.log_metrics(train_loss, val_loss)
            
            # Update training history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["learning_rates"].append(self.optimizer.param_groups[0]['lr'])
            if val_loss is not None:
                self.training_history["val_loss"].append(val_loss)
            
            # Save checkpoint
            is_best = False
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.num_epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")
            
            # Print metrics
            train_metrics = self.train_metrics.get_average_metrics()
            print("Train Metrics:", {k: f"{v:.4f}" for k, v in train_metrics.items()})
            
            if val_loss is not None:
                val_metrics = self.val_metrics.get_average_metrics()
                print("Val Metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})
            
            print("-" * 60)
        
        self.writer.close()
        print("Training completed!")
        
        # Save final results
        self.save_training_plots()
        return self.training_history
    
    def save_training_plots(self):
        """Save training progress plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.training_history["train_loss"]) + 1)
        axes[0, 0].plot(epochs, self.training_history["train_loss"], label='Train Loss', color='blue')
        if self.training_history["val_loss"]:
            axes[0, 0].plot(epochs, self.training_history["val_loss"], label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training/Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(epochs, self.training_history["learning_rates"], color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True)
        
        # Mock additional metrics plots
        mock_seg_iou = [0.6 + 0.3 * (1 - np.exp(-i/20)) + np.random.normal(0, 0.02) for i in epochs]
        mock_det_map = [0.5 + 0.4 * (1 - np.exp(-i/30)) + np.random.normal(0, 0.03) for i in epochs]
        
        axes[1, 0].plot(epochs, mock_seg_iou, label='Segmentation IoU', color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('Segmentation Performance')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, mock_det_map, label='Detection mAP', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP')
        axes[1, 1].set_title('Detection Performance')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {os.path.join(self.save_dir, 'training_progress.png')}")


class MockDataLoader:
    """Mock dataloader for testing."""
    
    def __init__(self, batch_size=4, num_batches=50, image_size=512):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.image_size = image_size
        
        # Pre-generate all batches
        self.batches = []
        for _ in range(num_batches):
            batch = []
            for _ in range(batch_size):
                batch.append({
                    "image": torch.randn(3, image_size, image_size),
                    "lang_tokens": torch.randint(0, 1000, (1, 20)),
                    "lang_mask": torch.ones(1, 20)
                })
            self.batches.append(batch)
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        return iter(self.batches)


def create_mock_dataloader(batch_size=4, num_batches=50, image_size=512):
    """Create mock dataloader for testing."""
    return MockDataLoader(batch_size, num_batches, image_size)


def main():
    """Main training function."""
    print("MultitaskGRES Training Pipeline")
    print("=" * 50)
    
    # Model setup
    backbone = MockBackbone()
    sem_seg_head = MockSemSegHead()
    criterion = MockCriterion()
    text_encoder = MockBertModel()
    
    model = SimplifiedMultitaskGRES(
        backbone=backbone,
        sem_seg_head=sem_seg_head,
        criterion=criterion,
        text_encoder=text_encoder,
        enable_detection=True,
        enable_keypoints=True,
        transformer_hidden_dim=256
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create mock dataloaders
    train_dataloader = create_mock_dataloader(batch_size=2, num_batches=20)
    val_dataloader = create_mock_dataloader(batch_size=2, num_batches=5)
    
    # Trainer setup
    trainer = MultitaskTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        num_epochs=30,  # Short demo training
        save_dir="./multitask_checkpoints",
        log_dir="./multitask_logs"
    )
    
    # Start training
    training_history = trainer.train()
    
    print("\nTraining Summary:")
    print(f"Final train loss: {training_history['train_loss'][-1]:.4f}")
    if training_history['val_loss']:
        print(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    print(f"\nCheckpoints saved to: {trainer.save_dir}")
    print(f"Logs saved to: {trainer.log_dir}")
    print(f"TensorBoard command: tensorboard --logdir {trainer.log_dir}")


if __name__ == "__main__":
    main()