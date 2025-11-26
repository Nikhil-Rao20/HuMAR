# ------------------------------------------------------------------------
# Modified from GroupPose (https://github.com/Michel-liu/GroupPose)
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from util import box_ops
from util.keypoint_loss import OKSLoss


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_sigmoid_ce_loss = nn.BCEWithLogitsLoss(reduction='none')


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum('nc,mc->nm', focal_pos, targets) + torch.einsum('nc,mc->nm', focal_neg, (1 - targets))

    return loss / hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, args, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 cost_keypoints: float = 1, cost_keypoints_visi: float = 1, cost_oks: float = 1,
                 cost_mask: float = 1, cost_dice: float = 1, focal_alpha=0.25, num_body_points=17):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_keypoints = cost_keypoints
        self.cost_keypoints_visi = cost_keypoints_visi
        self.cost_oks = cost_oks
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.focal_alpha = focal_alpha
        self.num_body_points = num_body_points
        self.args = args
        
        self.oks = OKSLoss(
            linear=True,
            num_keypoints=num_body_points,
            eps=1e-6,
            reduction='none',
            loss_weight=1.0
        )
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_keypoints != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_keypoints = outputs["pred_keypoints"].flatten(0, 1)  # [batch_size * num_queries, num_keypoints * 3]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_keypoints = torch.cat([v["keypoints"] for v in targets])
        tgt_area = torch.cat([v["area"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                  box_ops.box_cxcywh_to_xyxy(tgt_bbox))

        # Compute keypoints cost
        Z_pred = out_keypoints[:, :(self.num_body_points * 2)]
        Z_gt = tgt_keypoints[:, :(self.num_body_points * 2)]
        V_gt = tgt_keypoints[:, (self.num_body_points * 2):]
        
        # Compute L1 distance for each keypoint coordinate
        # V_gt needs to be expanded to match coordinate pairs
        V_gt_expanded = V_gt.repeat_interleave(2, dim=1)  # [num_targets, num_body_points * 2]
        
        # Compute distance per coordinate, weighted by visibility
        Z_pred_expanded = Z_pred.unsqueeze(1)  # [bs*nq, 1, num_body_points * 2]
        Z_gt_expanded = Z_gt.unsqueeze(0)  # [1, num_targets, num_body_points * 2]
        V_gt_expanded = V_gt_expanded.unsqueeze(0)  # [1, num_targets, num_body_points * 2]
        
        cost_keypoints = torch.abs(Z_pred_expanded - Z_gt_expanded) * V_gt_expanded
        cost_keypoints = cost_keypoints.sum(-1)  # [bs*nq, num_targets]
        
        # Compute OKS cost - need to compute pairwise for all predictions vs all targets
        num_preds = Z_pred.shape[0]
        num_targets = Z_gt.shape[0]
        cost_oks = torch.zeros((num_preds, num_targets), device=Z_pred.device)
        
        # Compute OKS for each prediction-target pair
        for i in range(num_preds):
            for j in range(num_targets):
                oks_val = self.oks(
                    Z_pred[i:i+1], 
                    Z_gt[j:j+1], 
                    V_gt[j:j+1], 
                    tgt_area[j:j+1],
                    weight=None, 
                    avg_factor=None, 
                    reduction_override='none'
                )
                cost_oks[i, j] = oks_val.item() if oks_val.numel() == 1 else oks_val.mean().item()
        
        # Compute keypoints visibility cost
        if self.args.kps_visi_trigger and 'pred_keypoints_visi' in outputs:
            out_keypoints_visi = outputs["pred_keypoints_visi"].flatten(0, 1)
            cost_keypoints_visi = F.binary_cross_entropy_with_logits(
                out_keypoints_visi.unsqueeze(1).expand(-1, V_gt.shape[0], -1),
                V_gt.unsqueeze(0).expand(out_keypoints_visi.shape[0], -1, -1),
                reduction='none'
            ).sum(-1)
        else:
            cost_keypoints_visi = torch.zeros_like(cost_keypoints)

        # Compute mask cost
        if 'pred_masks' in outputs and not self.args.no_mask:
            out_mask = outputs["pred_masks"]  # [batch_size, num_queries, H, W]
            tgt_mask = torch.cat([v["masks"] for v in targets])  # [num_targets, H, W]
            
            out_mask = out_mask.flatten(0, 1)  # [batch_size * num_queries, H, W]
            
            # Downsample target masks to match prediction size if needed
            if out_mask.shape[-2:] != tgt_mask.shape[-2:]:
                tgt_mask = F.interpolate(tgt_mask.unsqueeze(1).float(), size=out_mask.shape[-2:], 
                                        mode='nearest').squeeze(1)
            
            out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask.flatten(1).float()  # [num_targets, H*W]
            
            # Focal loss
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask, alpha=self.focal_alpha, gamma=2.0)
            # Dice loss
            cost_dice = batch_dice_loss(out_mask, tgt_mask)
        else:
            cost_mask = torch.zeros((bs * num_queries, len(targets[0]['labels'])), device=out_bbox.device)
            cost_dice = torch.zeros_like(cost_mask)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + \
            self.cost_class * cost_class + \
            self.cost_giou * cost_giou + \
            self.cost_keypoints * cost_keypoints + \
            self.cost_keypoints_visi * cost_keypoints_visi + \
            self.cost_oks * cost_oks + \
            self.cost_mask * cost_mask + \
            self.cost_dice * cost_dice
            
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        # Compute mean costs for logging
        matched_costs = {}
        for b, (i, j) in enumerate(indices):
            if len(i) > 0:
                start_idx = sum(sizes[:b])
                end_idx = start_idx + sizes[b]
                matched_costs.setdefault('cost_class', []).append(cost_class.view(bs, num_queries, -1)[b, i, j].mean().item())
                matched_costs.setdefault('cost_bbox', []).append(cost_bbox.view(bs, num_queries, -1)[b, i, j].mean().item())
                matched_costs.setdefault('cost_giou', []).append(cost_giou.view(bs, num_queries, -1)[b, i, j].mean().item())
                matched_costs.setdefault('cost_keypoints', []).append(cost_keypoints.view(bs, num_queries, -1)[b, i, j].mean().item())
                matched_costs.setdefault('cost_oks', []).append(cost_oks.view(bs, num_queries, -1)[b, i, j].mean().item())
        
        cost_mean_dict = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in matched_costs.items()}
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], cost_mean_dict


def build_matcher(args):
    return HungarianMatcher(
        args,
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_keypoints=args.set_cost_keypoints,
        cost_keypoints_visi=args.set_cost_keypoints_visi,
        cost_oks=args.set_cost_oks,
        cost_mask=args.set_cost_mask,
        cost_dice=args.set_cost_dice,
        focal_alpha=args.focal_alpha,
        num_body_points=args.num_body_points
    )
