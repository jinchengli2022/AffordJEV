"""Explicit full-image UMD metrics; ignore void pixels, not background."""
import numpy as np

from ..datasets.umd import CLASS_NAMES


def confusion_matrix(gt, prediction):
    gt, prediction = np.asarray(gt), np.asarray(prediction)
    if gt.ndim != 2 or gt.shape != prediction.shape:
        raise ValueError("Prediction and GT must use the same original 2D pixel grid")
    if not np.issubdtype(prediction.dtype, np.integer) or not np.issubdtype(gt.dtype, np.integer):
        raise ValueError("Class maps must contain integer IDs")
    if not set(np.unique(gt)).issubset(set(range(8)) | {-1}):
        raise ValueError("Unknown GT IDs")
    if not set(np.unique(prediction)).issubset(set(range(8))):
        raise ValueError("Prediction must contain only native IDs 0..7")
    valid = gt != -1
    indices = 8 * gt[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(indices, minlength=64).reshape(8, 8)


def scores_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (8, 8) or (matrix < 0).any():
        raise ValueError("Expected an 8 x 8 nonnegative confusion matrix")
    gt_count = matrix.sum(1)
    pred_count = matrix.sum(0)
    tp = np.diag(matrix)
    union = gt_count + pred_count - tp
    per_class = {}
    for class_id, name in CLASS_NAMES.items():
        per_class[name] = {
            "class_id": class_id, "gt_pixels": int(gt_count[class_id]),
            "pred_pixels": int(pred_count[class_id]), "intersection": int(tp[class_id]),
            "union": int(union[class_id]),
            "iou": float(tp[class_id] / union[class_id]) if union[class_id] else None,
            "dice": float(2 * tp[class_id] / (gt_count[class_id] + pred_count[class_id]))
                    if gt_count[class_id] + pred_count[class_id] else None,
        }
    fg_ious = [per_class[CLASS_NAMES[i]]["iou"] for i in range(1, 8) if union[i] > 0]
    return {
        "foreground_mIoU": float(np.mean(fg_ious)) if fg_ious else None,
        "foreground_pixel_cIoU": float(tp[1:].sum() / union[1:].sum()) if union[1:].sum() else None,
        "pixel_accuracy_with_background": float(tp.sum() / matrix.sum()) if matrix.sum() else None,
        "per_class": per_class, "confusion_matrix_gt_rows_pred_columns": matrix.tolist(),
    }


class Metrics:
    def __init__(self):
        self.matrix = np.zeros((8, 8), dtype=np.int64)
        self.query_ious = []
        self.query_dice = []
        self.images = 0

    def add(self, gt, prediction):
        matrix = confusion_matrix(gt, prediction)
        self.matrix += matrix
        self.images += 1
        scores = scores_from_confusion(matrix)
        rows = []
        # Queries are GT-present (image, affordance) pairs, evaluated only after inference.
        for class_id in range(1, 8):
            row = scores["per_class"][CLASS_NAMES[class_id]]
            if row["gt_pixels"]:
                self.query_ious.append(row["iou"])
                self.query_dice.append(row["dice"])
                rows.append({"class": CLASS_NAMES[class_id], **row})
        return rows

    def summary(self):
        return {
            **scores_from_confusion(self.matrix), "images": self.images,
            "gt_present_queries": len(self.query_ious),
            "gIoU_present_queries": float(np.mean(self.query_ious)) if self.query_ious else None,
            "mean_Dice_present_queries": float(np.mean(self.query_dice)) if self.query_dice else None,
            "P_at_50_present_queries": float(np.mean(np.asarray(self.query_ious) >= .5)) if self.query_ious else None,
        }
