import torch

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
            "IoU": self.iou
        }
        
    def precision(self, outputs, targets):
        tp = torch.sum((outputs==1) & (targets==1))
        fp = torch.sum((outputs==1) & (targets==0))
        return tp / (tp + fp + self.eps)
    
    def recall(self, outputs, targets):
        tp = torch.sum((outputs==1) & (targets==1))
        fn = torch.sum((outputs==0) & (targets==1))
        return tp / (tp + fn + self.eps)
    
    def f1(self, outputs, targets):
        p = self.precision(outputs, targets)
        r = self.recall(outputs, targets)
        return 2 * p * r / (p + r + self.eps)
    
    def accuracy(self, outputs, targets):
        correct = torch.sum((outputs==targets))
        total = targets.numel()
        return correct / total
    
    def iou(self, outputs, targets):
        intersection = torch.sum((outputs==1) & (targets==1))
        union = torch.sum((outputs==1) | (targets==1))
        return intersection/ (union + self.eps)
    
    def dice(self, outputs, targets):
        intersection = torch.sum((outputs==1) & (targets==1))
        return (2. * intersection) / (torch.sum(outputs) + torch.sum(targets) + self.eps)
    
    def calculate_metrics(self, outputs, targets):
        
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(outputs, targets).item()
        return results
    


class BinaryMetrics2():
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
            "Precision": 0,
            "Recall": 0,
            "F1": 0,
            "Accuracy": 0,
            "Dice": 0,
            "IoU": 0
        }
        
    def calculate_metrics(self, prediction_list, groundtruth_list):
        assert len(groundtruth_list) == len(prediction_list), "The number of groundtruths and predictions does not match."
        
        self.metrics = {
            "Precision": 0,
            "Recall": 0,
            "F1": 0,
            "Accuracy": 0,
            "Dice": 0,
            "IoU": 0
        }
        
        for groundtruth, prediction in zip(groundtruth_list, prediction_list):

            groundtruth = groundtruth.view(-1)
            prediction = prediction.view(-1)
            
            tp = torch.sum(groundtruth * prediction) # True Positives
            fp = torch.sum(prediction * (1 - groundtruth)) # False Positives
            fn = torch.sum((1 - prediction) * groundtruth) # False Negatives
            tn = torch.sum((1 - prediction) * (1 - groundtruth)) # True Negatives
            
            # print(f"tp: {tp}\n fp: {fp}\n fn: {fn}\n tn: {tn}")
            
            precision = (tp + self.eps) / (tp + fp + self.eps)
            recall = (tp + self.eps) / (tp + fn + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            accuracy = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
            dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
            iou = (tp + self.eps) / (tp + fp + fn + self.eps)
            
            self.metrics["Precision"] += precision.item()
            self.metrics["Recall"] += recall.item()
            self.metrics["F1"] += f1.item()
            self.metrics["Accuracy"] += accuracy.item()
            self.metrics["Dice"] += dice.item()
            self.metrics["IoU"] += iou.item()
            
        # Average the metrics
        num_samples = len(groundtruth_list)
        for key in self.metrics:
            self.metrics[key] /= num_samples
        
        return self.metrics