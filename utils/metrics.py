import numpy as np
import torch
import torch.nn as nn


class BinaryMetrics():
    
    def __init__(self, eps=1e-5):
        self.eps = eps
        
    def calculate_metrics(self, groundtruth_list, prediction_list):
        assert len(groundtruth_list) == len(prediction_list), "The number of groundtruths and predicitions does not match."
        
        metrics = {
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
            
            precision = (tp + self.eps) / (tp + fp + self.eps)
            recall = (tp + self.eps) / (tp + fn + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            accuracy = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
            dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
            iou = (tp + self.eps) / (tp + fp + fn + self.eps)
            
            metrics["Precision"] += precision.item()
            metrics["Recall"] += recall.item()
            metrics["F1"] += f1.item()
            metrics["Accuracy"] += accuracy.item()
            metrics["Dice"] += dice.item()
            metrics["IoU"] += iou.item()
            
        # Average the metrics
        num_samples = len(groundtruth_list)
        for key in metrics:
            metrics[key] /= num_samples
        
        return metrics