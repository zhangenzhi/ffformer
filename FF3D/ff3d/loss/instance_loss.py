"""Instance segmentation loss: mask BCE + dice + score."""
import torch
import torch.nn.functional as F


def dice_loss(pred, target):
    """Dice loss for binary masks."""
    pred = pred.sigmoid()
    intersection = (pred * target).sum(dim=-1)
    union = pred.sum(dim=-1) + target.sum(dim=-1)
    return (1 - 2 * intersection / (union + 1e-6)).mean()


def get_iou(pred_mask, gt_mask, ratio_inspoint=None):
    """Compute IoU between predicted and GT masks."""
    pred = pred_mask.sigmoid() > 0.5
    inter = (pred & gt_mask.bool()).sum(dim=-1).float()
    union = (pred | gt_mask.bool()).sum(dim=-1).float()
    iou = inter / (union + 1e-6)
    if ratio_inspoint is not None:
        iou = iou * ratio_inspoint
    return iou


def one2many_match(query_inslabel, n_gts):
    """One-to-many matching: each query matches its GT instance by position.

    Args:
        query_inslabel: (Q,) GT instance ID for each query (-1 = no match).
        n_gts: number of GT instances.

    Returns:
        idx_q: (Q,) query indices.
        idx_gt: (Q,) GT indices (-1 for unmatched).
    """
    idx_q = torch.arange(len(query_inslabel), device=query_inslabel.device)
    idx_gt = query_inslabel.clone()
    return idx_q, idx_gt


def instance_loss_layer(pred_masks, pred_scores, gt_masks, gt_labels,
                         query_inslabel, ratio_inspoint,
                         loss_weight=(1.0, 1.0, 0.5)):
    """Compute instance loss for one decoder layer.

    Args:
        pred_masks: List[(Q_i, M_i)] mask logits per sample.
        pred_scores: List[(Q_i, 1)] scores per sample.
        gt_masks: List[(K_i, M_i)] GT instance masks per sample.
        gt_labels: List[(K_i,)] GT labels per sample.
        query_inslabel: List[(Q_i,)] GT instance ID per query.
        ratio_inspoint: List[dict] ratio of instance points kept.
        loss_weight: (w_bce, w_dice, w_score).

    Returns:
        Scalar loss.
    """
    mask_bce_losses, mask_dice_losses, score_losses = [], [], []

    for i in range(len(pred_masks)):
        n_gts = gt_masks[i].shape[0]
        if n_gts == 0:
            continue

        idx_q, idx_gt = one2many_match(query_inslabel[i], n_gts)
        pred_mask = pred_masks[i][idx_q]

        # Build target masks
        tgt_mask = torch.zeros_like(pred_mask)
        valid = idx_gt != -1
        if valid.any():
            tgt_mask[valid] = gt_masks[i][idx_gt[valid]].float()

        mask_bce_losses.append(F.binary_cross_entropy_with_logits(pred_mask, tgt_mask))
        mask_dice_losses.append(dice_loss(pred_mask, tgt_mask))

        # Score loss
        if pred_scores[i] is not None:
            pred_score = pred_scores[i][idx_q]
            tgt_score = torch.zeros_like(pred_score)
            if valid.any():
                with torch.no_grad():
                    ratios = torch.tensor(
                        [ratio_inspoint[i].get(int(g), 1.0) for g in idx_gt[valid]],
                        device=pred_mask.device)
                    tgt_score[valid] = get_iou(
                        pred_mask[valid], tgt_mask[valid], ratios).unsqueeze(1).float()
            score_losses.append(F.mse_loss(pred_score, tgt_score))

    if not mask_bce_losses:
        return torch.tensor(0.0, device=pred_masks[0].device)

    n = len(pred_masks)  # batch_size
    n_valid = len(mask_bce_losses)  # samples with GT instances

    # Score loss
    score_loss = torch.stack(score_losses).sum() / n if score_losses else 0

    # Mask losses with fix_dice_loss_weight and fix_mean_loss (matching original)
    mask_bce_loss = torch.stack(mask_bce_losses).sum() / n
    mask_dice_loss = torch.stack(mask_dice_losses).sum() / n

    # fix_dice_loss_weight: normalize dice by batch_size, scale by 4
    mask_dice_loss = mask_dice_loss / n * 4

    # fix_mean_loss: correct for samples without GT
    mask_bce_loss = mask_bce_loss * n / n_valid
    mask_dice_loss = mask_dice_loss * n / n_valid

    return loss_weight[0] * mask_bce_loss + loss_weight[1] * mask_dice_loss + loss_weight[2] * score_loss
