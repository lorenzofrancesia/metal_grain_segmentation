import torch
import torchmetrics


class BinaryMetrics2():
    """
    A class to calculate various binary classification metrics using TorchMetrics.

    Args:
        eps (float, optional): A small value to avoid division by zero.  TorchMetrics
            handles this internally, but we keep it for consistency with the original
            class's interface. Default is 1e-5.
        num_classes (int): Number of classes. Should be 1 for binary classification.

    Attributes:
        eps (float): A small value to avoid division by zero.
        metrics (dict): A dictionary to store the calculated metrics.  The keys are
            the names of the metrics, and the values are the *TorchMetrics metric objects*
            (not the functions themselves, as in the original).
    """
    def __init__(self, eps=1e-5, num_classes=1):
        self.eps = eps  # Kept for API consistency, but not directly used
        self.metrics = {
            "Precision": torchmetrics.classification.BinaryPrecision(),
            "Recall": torchmetrics.classification.BinaryRecall(),
            "F1": torchmetrics.classification.BinaryF1Score(),
            "Accuracy": torchmetrics.classification.BinaryAccuracy(),
            "Dice": torchmetrics.classification.BinaryF1Score(),  # Dice is equivalent to F1
            "IoU": torchmetrics.classification.BinaryJaccardIndex(),
            "F2": torchmetrics.classification.BinaryFBetaScore(beta=2.0)
        }
    
    def _check_and_invert(self, outputs, targets):
        """
        Checks if the mean of the target tensor is closer to 1 than to 0.
        If so, inverts both the predictions (after thresholding) and the targets.

        Args:
            outputs (torch.Tensor): Model predictions (probabilities or logits).
            targets (torch.Tensor): Ground truth labels (0 or 1).

        Returns:
            tuple: (inverted_preds, inverted_targets) if inversion is needed,
                   otherwise (preds, targets).
        """
        targets = targets.float() # prevent integer overflow in mean calculation.
        if torch.mean(targets) > 0.5:
            # Invert both predictions and targets
            return 1 - outputs, 1 - targets
        return outputs, targets
    
    def calculate_metrics(self, outputs, targets, threshold=0.5):
        """
        Calculates all the defined metrics.

        Args:
            outputs (torch.Tensor): Model predictions (probabilities or logits).
            targets (torch.Tensor): Ground truth labels (0 or 1).
            threshold (float, optional):  Threshold to convert probabilities to
                binary predictions. Default: 0.5.

        Returns:
            dict: A dictionary where keys are metric names and values are the
                calculated metric values (as floats).
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        
        # outputs, targets = self._check_and_invert(outputs, targets)
        
        # Apply threshold to outputs
        preds = (outputs >= threshold).int()

        results = {}
        for metric_name, metric_object in self.metrics.items():
            # Use the TorchMetrics object directly
            results[metric_name] = metric_object(preds, targets).item()

        return results

    def reset(self):
        """
        Resets the internal state of all metrics.  This is useful when you want
        to calculate metrics over multiple batches/epochs and need to clear the
        previous calculations.
        """
        for metric_object in self.metrics.values():
            metric_object.reset()


# ----- DEPRECATED --------

class BinaryMetrics():
    """
    A class to calculate various binary classification metrics.

    Args:
        eps (float, optional): A small value to avoid division by zero. Default is 1e-5.

    Attributes:
        eps (float): A small value to avoid division by zero.
        metrics (dict): A dictionary to store the calculated metrics.
    """
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.metrics = {
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "Accuracy": self.accuracy,
            "Dice": self.dice,
            "IoU": self.iou,
            "F2": self.fbeta
        }
        
    def _apply_threshold(self, outputs, threshold=0.5): # Helper function for thresholding
        return torch.where(outputs >= threshold, 1, 0).long()
        
    def precision(self, outputs, targets, threshold=0.5):
        thresh_outputs = self._apply_threshold(outputs, threshold)
        tp = torch.sum((thresh_outputs == 1) & (targets == 1))
        fp = torch.sum((thresh_outputs == 1) & (targets == 0))
        return tp / (tp + fp + self.eps)

    def recall(self, outputs, targets, threshold=0.5):  # Add threshold parameter
        thresh_outputs = self._apply_threshold(outputs, threshold)
        tp = torch.sum((thresh_outputs == 1) & (targets == 1))
        fn = torch.sum((thresh_outputs == 0) & (targets == 1))
        return tp / (tp + fn + self.eps)

    def f1(self, outputs, targets, threshold=0.5): # Add threshold parameter
        p = self.precision(outputs, targets, threshold)
        r = self.recall(outputs, targets, threshold)
        return 2 * p * r / (p + r + self.eps)

    def fbeta(self, outputs, targets, beta=2, threshold=0.5):  # Add threshold parameter
        p = self.precision(outputs, targets, threshold)
        r = self.recall(outputs, targets, threshold)
        return (1 + beta**2) * p * r / (beta**2 * p + r + self.eps)

    def accuracy(self, outputs, targets, threshold=0.5): # Add threshold parameter
        thresh_outputs = self._apply_threshold(outputs, threshold)
        correct = torch.sum(thresh_outputs == targets)
        total = targets.numel()
        return correct / total

    def iou(self, outputs, targets, threshold=0.5): # Add threshold parameter
        thresh_outputs = self._apply_threshold(outputs, threshold)
        intersection = torch.sum((thresh_outputs == 1) & (targets == 1))
        union = torch.sum((thresh_outputs == 1) | (targets == 1))
        return intersection / (union + self.eps)

    def dice(self, outputs, targets, threshold=0.5): # Add threshold parameter
        thresh_outputs = self._apply_threshold(outputs, threshold)
        intersection = torch.sum((thresh_outputs == 1) & (targets == 1))
        return (2. * intersection) / (torch.sum(thresh_outputs) + torch.sum(targets) + self.eps)
    
    def calculate_metrics(self, outputs, targets, threshold=0.5):  # Single threshold
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        
        results = {}
        for metric_name, metric_func in self.metrics.items():
            if metric_name == "F2":
                results[metric_name] = metric_func(outputs, targets, beta=2, threshold=threshold).item()
            else:
                results[metric_name] = metric_func(outputs, targets, threshold=threshold).item()
        return results