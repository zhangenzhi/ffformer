"""Matrix NMS for instance mask suppression — matches mmcv/mmdet implementation."""
import torch


def mask_matrix_nms(masks, labels, scores, kernel='linear', sigma=2.0,
                    filter_thr=-1, nms_pre=-1, max_num=-1, mask_area=None):
    """Matrix NMS with compensate-IoU decay (matches mmdet).

    Args:
        masks: (N, M) float — sigmoid mask scores (0..1).
        labels: (N,) long.
        scores: (N,) float.
        kernel: 'linear' or 'gaussian'.
        sigma: gaussian std.
        filter_thr: score threshold after NMS (-1 to disable).
        nms_pre: limit before NMS (-1 to disable).
        max_num: limit after NMS (-1 to disable).
        mask_area: optional precomputed mask areas.

    Returns:
        scores, labels, masks, keep_inds
    """
    n = len(scores)
    if n == 0:
        return (scores.new_zeros(0), labels.new_zeros(0),
                masks.new_zeros(0, *masks.shape[-1:]), labels.new_zeros(0))

    if mask_area is None:
        mask_area = masks.sum(1).float()

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num = len(labels)
    flat = masks.reshape(num, -1).float()
    inter = torch.mm(flat, flat.t())
    expanded_area = mask_area.expand(num, num)
    iou = (inter / (expanded_area + expanded_area.t() - inter + 1e-6)).triu(diagonal=1)

    expanded_labels = labels.expand(num, num)
    label_matrix = (expanded_labels == expanded_labels.t()).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num, num).t()
    decay_iou = iou * label_matrix

    if kernel == 'gaussian':
        dm = torch.exp(-sigma * decay_iou ** 2)
        cm = torch.exp(-sigma * compensate_iou ** 2)
        decay, _ = (dm / cm).min(0)
    elif kernel == 'linear':
        dm = (1 - decay_iou) / (1 - compensate_iou + 1e-6)
        decay, _ = dm.min(0)
    else:
        raise NotImplementedError(f'{kernel} kernel not supported')

    scores = scores * decay

    if filter_thr > 0:
        keep = scores >= filter_thr
        if not keep.any():
            return (scores.new_zeros(0), labels.new_zeros(0),
                    masks.new_zeros(0, *masks.shape[-1:]), labels.new_zeros(0))
        masks = masks[keep]; scores = scores[keep]; labels = labels[keep]
        keep_inds = keep_inds[keep]

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds
