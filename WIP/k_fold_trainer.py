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

class KfoldTrainer(Trainer):
    
    def __init__(self, *args, k_folds=5, **kwargs):
        super.__init__(*args, **kwargs)
        self.k_folds = k_folds
        self.all_fold_results = []
        self.current_fold = 0
        
    def _get_kfold_dataloaders(self, fold, train_indices, val_indices):
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_dataset = SegmentationDataset(
            image_dir=os.path.join(self.data_dir, "train/images"),
            mask_dir=os.path.join(self.data_dir, "train/masks"), 
            image_transform=self.train_transform,
            mask_transform=self.train_transform,
            normalize=self.normalize,
            negative=self.negative,
            verbose=False
        )