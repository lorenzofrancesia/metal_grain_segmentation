import torch
import torchmetrics

class BinaryMetrics():
    def __init__(self, eps=1e-5, num_classes=1, device=None):
        self.eps = eps
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create metrics on the correct device
        self.metrics = {
            "Precision": torchmetrics.classification.BinaryPrecision().to(self.device),
            "Recall": torchmetrics.classification.BinaryRecall().to(self.device),
            "F1": torchmetrics.classification.BinaryF1Score().to(self.device),
            "Accuracy": torchmetrics.classification.BinaryAccuracy().to(self.device),
            "Dice": torchmetrics.classification.BinaryF1Score().to(self.device),
            "IoU": torchmetrics.classification.BinaryJaccardIndex().to(self.device),
            "F2": torchmetrics.classification.BinaryFBetaScore(beta=2.0).to(self.device)
        }
    
    def _check_and_invert(self, outputs, targets):
        targets = targets.float()
        if torch.mean(targets) > 0.5:
            return 1 - outputs, 1 - targets
        return outputs, targets
    
    def calculate_metrics(self, outputs, targets, threshold=0.5):
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        
        # outputs, targets = self._check_and_invert(outputs, targets)
        
        # Apply threshold to outputs
        preds = (outputs >= threshold).int()
        
        results = {}
        for metric_name, metric_object in self.metrics.items():
            # Make sure all tensors are on the same device as the metric
            results[metric_name] = metric_object(preds, targets).item()
        
        return results
    
    def reset(self):
        for metric_object in self.metrics.values():
            metric_object.reset()

    def to(self, device):
        """Move all metrics to the specified device"""
        self.device = device
        for metric_name in self.metrics:
            self.metrics[metric_name] = self.metrics[metric_name].to(device)
        return self
    

# ----- DEPRECATED --------

class BinaryMetricsDeprecated():
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