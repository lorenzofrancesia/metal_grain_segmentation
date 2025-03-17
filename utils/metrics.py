import torch
import torchmetrics
 
 
class BinaryMetrics():
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


    def to(self, device):
        """Move all metrics to the specified device"""
        self.device = device
        for metric_name in self.metrics:
            self.metrics[metric_name] = self.metrics[metric_name].to(device)
        return self