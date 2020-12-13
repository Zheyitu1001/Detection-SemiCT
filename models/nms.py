import numpy as np

import torch
from .iou import bbox_iou, bbox_giou, bbox_diou, bbox_ciou

def nms(boxes, scores, max_output_size=300, iou_threshold=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # Sort the detections by maximum objectness confidence
    _, conf_sort_index = torch.sort(scores, descending=True)
    boxes = boxes[conf_sort_index]
    # Perform non-maximum suppression
    max_indexes = []
    count = 0
    while boxes.shape[0] > 0:
        # Get detection with highest confidence and save as max detection
        max_detections = boxes[0].unsqueeze(0)  # expand 1 dim
        max_indexes.append(conf_sort_index[0])
        # Stop if we're at the last detection
        if boxes.shape[0] == 1:
            break
        # Get the IOUs for all boxes with lower confidence
        ious = bbox_iou(max_detections, boxes[1:])
        # Remove detections with IoU >= NMS threshold
        boxes = boxes[1:][ious < iou_threshold]
        conf_sort_index = conf_sort_index[1:][ious < iou_threshold]
        # break when get enough bboxes
        count += 1
        if count >= max_output_size:
            break

    # max_detections = torch.cat(max_detections).data
    max_indexes = torch.stack(max_indexes).data
    return max_indexes

def soft_nms():
    return None

def soft_nms_with_uncertainty_vote():
    return None