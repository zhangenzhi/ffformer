"""Evaluation metrics for forest point cloud segmentation.

Computes F1, Precision, Recall, mIoU, PQ, SQ, RQ, MUCov, MWCov
using the same protocol as the original ForestFormer3D UnifiedSegMetric.
"""
import numpy as np
from scipy import stats


def evaluate_scene(sem_pred, ins_pred, sem_gt, ins_gt,
                   stuff_class_inds=(0,), thing_class_inds=(1, 2)):
    """Evaluate one scene. Returns raw counts for global aggregation."""
    NUM_SEM = 4    # 0:unclassified, 1:ground, 2:wood, 3:leaf
    NUM_BIN = 3    # 0:unclassified, 1:stuff, 2:thing

    # Shift by 1 (0 = unclassified)
    sem_gt_i = sem_gt + 1
    sem_pre_i = sem_pred + 1

    # Semantic confusion
    true_pos_sem = np.zeros(NUM_SEM)
    pos_sem = np.zeros(NUM_SEM)
    gt_sem = np.zeros(NUM_SEM)
    for j in range(len(sem_gt_i)):
        gl, pl = int(sem_gt_i[j]), int(sem_pre_i[j])
        gt_sem[gl] += 1
        pos_sem[pl] += 1
        true_pos_sem[gl] += int(gl == pl)

    # Binary mapping: stuff → 1, thing → 2
    sem_gt_bi = np.copy(sem_gt_i)
    sem_pre_bi = np.copy(sem_pre_i)
    for sc in stuff_class_inds:
        sem_gt_bi[sem_gt_i == sc + 1] = 1
        sem_pre_bi[sem_pre_i == sc + 1] = 1
    for tc in thing_class_inds:
        sem_gt_bi[sem_gt_i == tc + 1] = 2
        sem_pre_bi[sem_pre_i == tc + 1] = 2

    true_pos_bi = np.zeros(NUM_BIN)
    pos_bi = np.zeros(NUM_BIN)
    gt_bi = np.zeros(NUM_BIN)
    for j in range(len(sem_gt_bi)):
        gl, pl = int(sem_gt_bi[j]), int(sem_pre_bi[j])
        gt_bi[gl] += 1
        pos_bi[pl] += 1
        true_pos_bi[gl] += int(gl == pl)

    # Instance evaluation (filter ground)
    idxc = (sem_gt_bi != 1) | (sem_pre_bi != 1)
    pred_ins_f = ins_pred[idxc]
    gt_ins_f = ins_gt[idxc]
    pred_sem_f = sem_pre_bi[idxc]
    gt_sem_f = sem_gt_bi[idxc]

    INS_CLASS = 2  # tree

    # Predicted instances grouped by class
    pts_in_pred = []
    for g in np.unique(pred_ins_f):
        if g == -1:
            continue
        tmp = pred_ins_f == g
        seg_i = int(stats.mode(pred_sem_f[tmp], keepdims=True)[0][0])
        if seg_i == INS_CLASS:
            pts_in_pred.append(tmp)

    # GT instances grouped by class
    pts_in_gt = []
    for g in np.unique(gt_ins_f):
        if g == 0:
            continue
        tmp = gt_ins_f == g
        seg_i = int(stats.mode(gt_sem_f[tmp], keepdims=True)[0][0])
        if seg_i == INS_CLASS:
            pts_in_gt.append(tmp)

    # PQ metrics
    n_gt = len(pts_in_gt)
    tp_list = []
    fp_list = []
    iou_tp = 0.0

    for ip, ins_pred_mask in enumerate(pts_in_pred):
        ovmax = -1.0
        if not pts_in_gt:
            fp_list.append(1.0)
            continue
        for ins_gt_mask in pts_in_gt:
            inter = (ins_pred_mask & ins_gt_mask).sum()
            union = (ins_pred_mask | ins_gt_mask).sum()
            iou = float(inter) / union
            if iou > ovmax:
                ovmax = iou
        if ovmax >= 0.5:
            tp_list.append(1.0)
            iou_tp += ovmax
        else:
            fp_list.append(1.0)

    # Coverage
    sum_cov = 0.0
    num_gt_pt = 0
    mean_weighted_cov = 0.0
    if pts_in_gt and pts_in_pred:
        for ins_gt_mask in pts_in_gt:
            ovmax = 0.0
            ngt = ins_gt_mask.sum()
            num_gt_pt += ngt
            for ins_pred_mask in pts_in_pred:
                inter = (ins_pred_mask & ins_gt_mask).sum()
                union = (ins_pred_mask | ins_gt_mask).sum()
                iou = float(inter) / union
                if iou > ovmax:
                    ovmax = iou
            sum_cov += ovmax
            mean_weighted_cov += ovmax * ngt

    return {
        'true_pos_sem': true_pos_sem,
        'pos_sem': pos_sem,
        'gt_sem': gt_sem,
        'true_pos_bi': true_pos_bi,
        'pos_bi': pos_bi,
        'gt_bi': gt_bi,
        'n_gt_ins': n_gt,
        'tp': tp_list,
        'fp': fp_list,
        'iou_tp': iou_tp,
        'sum_cov': sum_cov / max(len(pts_in_gt), 1) if pts_in_gt else 0,
        'mean_wcov': mean_weighted_cov / max(num_gt_pt, 1) if pts_in_gt else 0,
        'has_gt': len(pts_in_gt) > 0,
        'has_pred': len(pts_in_pred) > 0,
    }


def aggregate_metrics(scene_results):
    """Aggregate per-scene results into global metrics.

    Args:
        scene_results: List of dicts from evaluate_scene().

    Returns:
        Dict with mIoU, mIoU_binary, F1, Precision, Recall, mPQ, mSQ, mRQ, MUCov, MWCov.
    """
    NUM_SEM = 4
    NUM_BIN = 3

    true_pos_sem = np.zeros(NUM_SEM)
    pos_sem = np.zeros(NUM_SEM)
    gt_sem = np.zeros(NUM_SEM)
    true_pos_bi = np.zeros(NUM_BIN)
    pos_bi = np.zeros(NUM_BIN)
    gt_bi = np.zeros(NUM_BIN)
    total_gt_ins = 0
    all_tp = []
    all_fp = []
    iou_tp_total = 0.0
    all_cov = []
    all_wcov = []

    for r in scene_results:
        true_pos_sem += r['true_pos_sem']
        pos_sem += r['pos_sem']
        gt_sem += r['gt_sem']
        true_pos_bi += r['true_pos_bi']
        pos_bi += r['pos_bi']
        gt_bi += r['gt_bi']
        total_gt_ins += r['n_gt_ins']
        all_tp.extend(r['tp'])
        all_fp.extend(r['fp'])
        iou_tp_total += r['iou_tp']
        if r['has_gt'] and r['has_pred']:
            all_cov.append(r['sum_cov'])
            all_wcov.append(r['mean_wcov'])

    # Semantic mIoU
    valid_sem = [i for i in range(1, NUM_SEM) if gt_sem[i] > 0]
    iou_list = []
    for i in range(1, NUM_SEM):
        iou = true_pos_sem[i] / (gt_sem[i] + pos_sem[i] - true_pos_sem[i] + 1e-8)
        iou_list.append(iou)
    mIoU = np.mean([iou_list[i - 1] for i in valid_sem]) if valid_sem else 0

    # Binary mIoU
    valid_bi = [i for i in range(1, NUM_BIN) if gt_bi[i] > 0]
    iou_bi = []
    for i in range(1, NUM_BIN):
        iou = true_pos_bi[i] / (gt_bi[i] + pos_bi[i] - true_pos_bi[i] + 1e-8)
        iou_bi.append(iou)
    mIoU_bi = np.mean([iou_bi[i - 1] for i in valid_bi]) if valid_bi else 0

    # Instance metrics
    tp_sum = sum(all_tp)
    fp_sum = sum(all_fp)
    prec = tp_sum / (tp_sum + fp_sum + 1e-8)
    rec = tp_sum / (total_gt_ins + 1e-8)
    F1 = 2 * prec * rec / (prec + rec + 1e-8)
    RQ = F1
    SQ = iou_tp_total / (tp_sum + 1e-8)
    PQ = SQ * RQ

    MUCov = np.mean(all_cov) if all_cov else 0
    MWCov = np.mean(all_wcov) if all_wcov else 0

    return {
        'mIoU': round(mIoU, 4),
        'mIoU_binary': round(mIoU_bi, 4),
        'mPQ': round(PQ, 4),
        'mSQ': round(SQ, 4),
        'mRQ': round(RQ, 4),
        'mPrecision': round(prec, 4),
        'mRecall': round(rec, 4),
        'F1': round(F1, 4),
        'mMUCov': round(MUCov, 4),
        'mMWCov': round(MWCov, 4),
    }
