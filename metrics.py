import numpy as np
import torch
from skimage import measure
from test import cal_tp_pos_fp_neg

class ROCMetric():
    """Computes ROC curve metrics for detection performance evaluation."""

    def __init__(self, nclass, bins):
        # bins determines the number of discrete threshold values on the ROC curve
        # nclass: number of classes (infrared small target detection typically has 1 class)
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        # Arrays to store metrics for each threshold bin
        self.true_positives = np.zeros(self.bins + 1)
        self.positive_samples = np.zeros(self.bins + 1)  # Ground truth positive samples
        self.false_positives = np.zeros(self.bins + 1)
        self.negative_samples = np.zeros(self.bins + 1)
        self.predicted_positives = np.zeros(self.bins + 1)  # Total predicted positive samples

    def update(self, preds, labels):
        """Update metrics with new predictions and labels."""
        for threshold_idx in range(self.bins + 1):
            score_thresh = -30 + threshold_idx * (255 / self.bins)
            tp, pos, fp, neg, pred_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.true_positives[threshold_idx] += tp
            self.positive_samples[threshold_idx] += pos
            self.false_positives[threshold_idx] += fp
            self.negative_samples[threshold_idx] += neg
            self.predicted_positives[threshold_idx] += pred_pos

    def get(self):
        """Get ROC curve metrics."""
        # True Positive Rate = Recall = TP/(TP+FN)
        tpr = self.true_positives / (self.positive_samples + 0.001)
        # False Positive Rate = FP/(FP+TN)
        fpr = self.false_positives / (self.negative_samples + 0.001)
        # False Positive rate (alternative calculation)
        fp_rate = self.false_positives / (self.negative_samples + self.positive_samples)
        # Recall = TP/(TP+FN)
        recall = self.true_positives / (self.positive_samples + 0.001)
        # Precision = TP/(TP+FP)
        precision = self.true_positives / (self.predicted_positives + 0.001)

        return tpr, fpr, recall, precision, fp_rate

    def reset(self):
        """Reset all metric arrays."""
        self.true_positives = np.zeros([11])
        self.positive_samples = np.zeros([11])
        self.false_positives = np.zeros([11])
        self.negative_samples = np.zeros([11])
        self.predicted_positives = np.zeros([11])


class mIoU():

    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct  # Number of correctly predicted pixels
        self.total_label += labeled  # Number of ground-truth target pixels
        self.total_inter += inter  # Intersection
        self.total_union += union  # Union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class PDFA():
    """Computes Probability of Detection (PD) and False Alarm (FA) metrics."""

    def __init__(self):
        super(PDFA, self).__init__()
        self.predicted_areas = []  # All predicted object areas
        self.matched_areas = []    # Matched predicted object areas
        self.false_alarm_pixels = 0  # False alarm pixel count
        self.total_pixels = 0      # Total image pixels
        self.detected_targets = 0  # Number of detected targets
        self.ground_truth_targets = 0  # Number of ground truth targets

    def update(self, preds, labels, size):
        """Update metrics with new predictions and labels."""
        predictions = np.array((preds).cpu()).astype('int64')
        ground_truth = np.array((labels).cpu()).astype('int64')

        # Find connected components in predictions and ground truth
        pred_components = measure.label(predictions, connectivity=2)
        gt_components = measure.label(ground_truth, connectivity=2)

        pred_regions = measure.regionprops(pred_components)
        gt_regions = measure.regionprops(gt_components)

        # Count ground truth targets
        self.ground_truth_targets += len(gt_regions)
        self.predicted_areas = []
        self.matched_areas = []
        self.matched_distances = []
        self.unmatched_predictions = []

        # Extract areas of predicted regions
        for region in pred_regions:
            self.predicted_areas.append(region.area)

        # Match predictions to ground truth based on centroid distance
        for gt_region in gt_regions:
            gt_centroid = np.array(list(gt_region.centroid))
            for pred_idx, pred_region in enumerate(pred_regions):
                pred_centroid = np.array(list(pred_region.centroid))
                distance = np.linalg.norm(pred_centroid - gt_centroid)
                if distance < 3:  # Match if centroids are close
                    self.matched_distances.append(distance)
                    self.matched_areas.append(pred_region.area)
                    # Remove matched prediction from list
                    pred_regions = [r for i, r in enumerate(pred_regions) if i != pred_idx]
                    break

        # Unmatched predictions are false alarms
        self.unmatched_predictions = [r.area for r in pred_regions]
        self.false_alarm_pixels += np.sum(self.unmatched_predictions)
        self.total_pixels += size[0] * size[1]
        self.detected_targets += len(self.matched_distances)  # Count of successfully matched targets

    def get(self):
        """Get final PD and FA metrics."""
        false_alarm_rate = self.false_alarm_pixels / self.total_pixels
        probability_detection = self.detected_targets / self.ground_truth_targets
        return probability_detection, float(false_alarm_rate)

    def reset(self):
        """Reset all metrics."""
        self.false_alarm_pixels = 0
        self.total_pixels = 0
        self.detected_targets = 0
        self.ground_truth_targets = 0


def _normalize_target_shape(target):
    """Normalize target tensor shape to 4D for consistent processing.

    Args:
        target: Target tensor (3D or 4D)

    Returns:
        Normalized target tensor
    """
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    return target


def batch_pix_accuracy(output, target):
    target = _normalize_target_shape(target)
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()  # Convert output from True/False to 1/0
    pixel_labeled = (target > 0).float().sum()  # Count of 1s in ground truth
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()  # Count of correct predictions
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    target = _normalize_target_shape(target)
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
