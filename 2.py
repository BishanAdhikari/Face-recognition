import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.persons = os.listdir(root_dir)

        for person in self.persons:
            images = os.listdir(os.path.join(root_dir, person))
            if len(images) < 2:
                continue  # Ignore folders with less than 2 images
            self.data.append(images)
            self.labels.append(person)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        anchor_label = self.labels[index]
        anchor_dir = os.path.join(self.root_dir, anchor_label)
        anchor_img = random.choice(os.listdir(anchor_dir))
        anchor_path = os.path.join(anchor_dir, anchor_img)

        positive_img = random.choice(os.listdir(anchor_dir))
        positive_path = os.path.join(anchor_dir, positive_img)

        negative_label = random.choice([x for x in self.labels if x != anchor_label])
        negative_dir = os.path.join(self.root_dir, negative_label)
        negative_img = random.choice(os.listdir(negative_dir))
        negative_path = os.path.join(negative_dir, negative_img)

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(SiameseNetwork, self).__init__()
        if backbone == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Identity()  # Remove classification head
            self.embedding_size = 2048  # ResNet50 output feature size
        elif backbone == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier = nn.Identity()
            self.embedding_size = 1280  # MobileNet output feature size
        else:
            raise ValueError("Backbone not supported")

        self.fc = nn.Linear(self.embedding_size, 512)  # Reduce feature size
        self.relu = nn.ReLU()
