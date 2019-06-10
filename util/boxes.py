import numpy as np


def bbox2feat_grid(bbox, stride_H, stride_W, feat_H, feat_W):
    x1, y1, w, h = bbox
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    # map the bbox coordinates to feature grid
    x1 = x1 * 1. / stride_W - 0.5
    y1 = y1 * 1. / stride_H - 0.5
    x2 = x2 * 1. / stride_W - 0.5
    y2 = y2 * 1. / stride_H - 0.5
    xc = min(max(int(round((x1 + x2) / 2.)), 0), feat_W - 1)
    yc = min(max(int(round((y1 + y2) / 2.)), 0), feat_H - 1)
    ind = yc * feat_W + xc
    offset = x1 - xc, y1 - yc, x2 - xc, y2 - yc
    return ind, offset


def feat_grid2bbox(ind, offset, stride_H, stride_W, feat_H, feat_W):
    xc = ind % feat_W
    yc = ind // feat_W
    x1 = (xc + offset[0] + 0.5) * stride_W
    y1 = (yc + offset[1] + 0.5) * stride_H
    x2 = (xc + offset[2] + 0.5) * stride_W
    y2 = (yc + offset[3] + 0.5) * stride_H
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    bbox = x1, y1, w, h
    return bbox


def bbox_iou(bbox_1, bbox_2):
    x1_1, y1_1, w_1, h_1 = bbox_1
    x2_1 = x1_1 + w_1 - 1
    y2_1 = y1_1 + h_1 - 1
    A_1 = w_1 * h_1
    x1_2, y1_2, w_2, h_2 = bbox_2
    x2_2 = x1_2 + w_2 - 1
    y2_2 = y1_2 + h_2 - 1
    A_2 = w_2 * h_2
    w_i = max(min(x2_1, x2_2) - max(x1_1, x1_2) + 1, 0)
    h_i = max(min(y2_1, y2_2) - max(y1_1, y1_2) + 1, 0)
    A_i = w_i * h_i
    IoU = A_i / (A_1 + A_2 - A_i)
    return IoU


def batch_bbox2feat_grid(bbox, stride_H, stride_W, feat_H, feat_W):
    x1, y1, w, h = bbox.T
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    # map the bbox coordinates to feature grid
    x1 = x1 * 1. / stride_W - 0.5
    y1 = y1 * 1. / stride_H - 0.5
    x2 = x2 * 1. / stride_W - 0.5
    y2 = y2 * 1. / stride_H - 0.5
    xc = np.minimum(
        np.maximum(np.int32(np.round((x1 + x2) / 2.)), 0), feat_W - 1)
    yc = np.minimum(
        np.maximum(np.int32(np.round((y1 + y2) / 2.)), 0), feat_H - 1)
    ind = yc * feat_W + xc
    offset = x1 - xc, y1 - yc, x2 - xc, y2 - yc
    return ind, offset


def batch_feat_grid2bbox(ind, offset, stride_H, stride_W, feat_H, feat_W):
    xc = ind % feat_W
    yc = ind // feat_W
    x1 = (xc + offset[:, 0] + 0.5) * stride_W
    y1 = (yc + offset[:, 1] + 0.5) * stride_H
    x2 = (xc + offset[:, 2] + 0.5) * stride_W
    y2 = (yc + offset[:, 3] + 0.5) * stride_H
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    bbox = np.stack((x1, y1, w, h), axis=1)
    return bbox


def batch_bbox_iou(bbox_1, bbox_2):
    x1_1, y1_1, w_1, h_1 = bbox_1.T
    x2_1 = x1_1 + w_1 - 1
    y2_1 = y1_1 + h_1 - 1
    A_1 = w_1 * h_1
    x1_2, y1_2, w_2, h_2 = bbox_2.T
    x2_2 = x1_2 + w_2 - 1
    y2_2 = y1_2 + h_2 - 1
    A_2 = w_2 * h_2
    w_i = np.maximum(np.minimum(x2_1, x2_2) - np.maximum(x1_1, x1_2) + 1, 0)
    h_i = np.maximum(np.minimum(y2_1, y2_2) - np.maximum(y1_1, y1_2) + 1, 0)
    A_i = w_i * h_i
    IoU = A_i / (A_1 + A_2 - A_i)
    return IoU
