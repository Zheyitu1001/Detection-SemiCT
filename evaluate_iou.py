import numpy as np
import math

from numba import jit


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.
    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()
        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if dy < 0:
        return 0.0
    overlap_area = dx * dy
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )
    return overlap_area / union_area

@jit(nopython=True)
def calculate_giou(gt, pr, form='pascal_voc') -> float:
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()
        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if dy < 0:
        return 0.0
    overlap_area = dx * dy
    area_1 = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    area_2 = (pr[2] - pr[0] + 1) * (pr[2] - pr[0] + 1)
    union_area = area_1 + area_2 - overlap_area
    iou = overlap_area / union_area

    minimum_bound_rectangle = (max(gt[0], gt[2], pr[0], pr[2]) - min(gt[0], gt[2], pr[0], pr[2]) + 1) * \
                              (max(gt[1], gt[3], pr[1], pr[3]) - min(gt[1], gt[3], pr[1], pr[3]) + 1)

    g_factor = (minimum_bound_rectangle - union_area) / minimum_bound_rectangle
    return iou - g_factor

@jit(nopython=True)
def calculate_diou(gt, pr, form='pascal_voc') -> float:
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()
        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if dy < 0:
        return 0.0
    overlap_area = dx * dy
    area_1 = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    area_2 = (pr[2] - pr[0] + 1) * (pr[2] - pr[0] + 1)
    union_area = area_1 + area_2 - overlap_area
    iou = overlap_area / union_area

    x_gt_center = (gt[2] + gt[0]) / 2.0
    y_gt_center = (gt[3] + gt[1]) / 2.0
    x_pr_center = (pr[2] + pr[0]) / 2.0
    y_pr_center = (pr[3] + pr[1]) / 2.0
    center_distance = pow((x_gt_center - x_pr_center), 2) + pow((y_gt_center - y_pr_center), 2)
    x_min_mbc = min(gt[0], gt[2], pr[0], pr[2])
    y_min_mbc = min(gt[1], gt[3], pr[1], pr[3])
    x_max_mbc = max(gt[0], gt[2], pr[0], pr[2])
    y_max_mbc = max(gt[1], gt[3], pr[1], pr[3])
    diagonal = pow((x_min_mbc - x_max_mbc), 2) + pow((y_min_mbc - y_max_mbc), 2)
    d_factor = center_distance / diagonal
    return iou - d_factor

@jit(nopython=True)
def calculate_ciou(gt, pr, alpha=None, form='pascal_voc') -> float:
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()
        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if dy < 0:
        return 0.0
    overlap_area = dx * dy
    area_1 = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    area_2 = (pr[2] - pr[0] + 1) * (pr[2] - pr[0] + 1)
    union_area = area_1 + area_2 - overlap_area
    iou = overlap_area / union_area

    x_gt_center = (gt[2] + gt[0]) / 2.0
    y_gt_center = (gt[3] + gt[1]) / 2.0
    x_pr_center = (pr[2] + pr[0]) / 2.0
    y_pr_center = (pr[3] + pr[1]) / 2.0
    center_distance = pow((x_gt_center - x_pr_center), 2) + pow((y_gt_center - y_pr_center), 2)
    x_min_mbc = min(gt[0], gt[2], pr[0], pr[2])
    y_min_mbc = min(gt[1], gt[3], pr[1], pr[3])
    x_max_mbc = max(gt[0], gt[2], pr[0], pr[2])
    y_max_mbc = max(gt[1], gt[3], pr[1], pr[3])
    diagonal = pow((x_min_mbc - x_max_mbc), 2) + pow((y_min_mbc - y_max_mbc), 2)
    d_factor = center_distance / diagonal

    w_gt = gt[2] - gt[0]
    h_gt = gt[3] - gt[1]
    w_pr = pr[2] - pr[0]
    h_pr = pr[3] - pr[1]

    d1 = math.atan((w_gt / h_gt)) - math.atan((w_pr / h_pr))
    mu = (4 / math.pi ** 2) * (d1 ** 2)
    c_factor = d_factor + alpha * mu
    return iou - c_factor

@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, iou_type='iou', threshold=0.5, form='pascal_voc', ious=None) -> tuple(float, int):
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).
    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1
    for gt_idx in range(len(gts)):
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            if iou_type == 'iou':
                iou = calculate_iou(gts[gt_idx], pred, form=form)
            elif iou_type == 'giou':
                iou = calculate_giou(gts[gt_idx], pred, form=form)
            elif iou_type == 'diou':
                iou = calculate_diou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return (best_match_iou, best_match_idx)



@jit(nopython=True)
def calculate_precision(gts, preds, threshold=0.5, form='coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)
