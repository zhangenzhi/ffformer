"""Semantic segmentation loss with deep supervision."""
import torch
import torch.nn.functional as F


def semantic_loss(pred_masks, gt_sem_masks, loss_weight=0.2):
    """Semantic loss: BCE with logits (per-class binary), matching original S3DISSemanticCriterion.

    Original uses mmdet.CrossEntropyLoss(use_sigmoid=True) which is BCE, not softmax CE.
    """
    losses = []
    for pred, gt in zip(pred_masks, gt_sem_masks):
        # pred: (n_sem, M) logits, gt: (n_classes, M) one-hot float
        target = gt.float().argmax(0)  # (M,) class indices
        # BCE per class (same as CrossEntropyLoss with use_sigmoid=True)
        loss = F.binary_cross_entropy_with_logits(pred.T, F.one_hot(target, pred.shape[0]).float())
        losses.append(loss)
    return loss_weight * torch.mean(torch.stack(losses))


def semantic_loss_with_aux(pred, gt_sem_masks, loss_weight=0.2, num_sem_classes=3):
    """Semantic loss with deep supervision on aux outputs.

    Args:
        pred: Dict with 'masks' and 'aux_outputs'.
        gt_sem_masks: List of GT semantic masks.

    Returns:
        Dict of losses: 'last_layer_seg_loss', 'layer_0_seg_loss', etc.
    """
    # Extract semantic masks from predictions (last num_sem_classes rows)
    n = num_sem_classes
    sem_preds = [m[-n:, :] for m in pred['masks']]
    loss = {'last_layer_seg_loss': semantic_loss(sem_preds, gt_sem_masks, loss_weight)}

    if 'aux_outputs' in pred:
        for i, aux in enumerate(pred['aux_outputs']):
            aux_sem = [m[-n:, :] for m in aux['masks']]
            loss[f'layer_{i}_seg_loss'] = semantic_loss(aux_sem, gt_sem_masks, loss_weight)

    return loss
