import os
import sys
import tqdm
import csv
import matplotlib.pyplot as plt
import yaml
from collections import defaultdict
import imageio
from sklearn.metrics import precision_recall_curve, average_precision_score
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold

from data.dataset import SegmentationDataset
from utils.metrics import BinaryMetrics
from utils.trainer import Trainer

class KFoldTrainer(Trainer):
    
    def __init__(self, *args, k_folds=5, **kwargs):
        super.__init__(*args, **kwargs)
        self.k_folds = k_folds
        self.all_fold_results = []
        self.current_fold = 0
        
    def _get_kfold_dataloaders(self, train_indices, val_indices):
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_dataset = self.dataset  # Use the combined dataset
        val_dataset = self.dataset    # Use the combined dataset

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, sampler=val_sampler)

        if self.save_output:
            self.image_evolution_idx = np.random.randint(0, len(val_indices)-1)

        return train_loader, val_loader
    
    def train(self):
        
        kf = KFold(n_splits=self.k_folds, shuffle=True)
        
        