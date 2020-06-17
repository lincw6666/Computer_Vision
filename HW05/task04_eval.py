# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd

batch_size = 1
device = 'cuda:0'

if __name__ == '__main__':

    """
    Create a new network to load the trained model.
    """

    # Load Model.
    net = torchvision.models.resnet18(pretrained=False)
    num_features = net.fc.in_features
    num_classes = 15
    net.fc = torch.nn.Linear(num_features, num_classes)
    net.load_state_dict(torch.load('./checkpoints/net_100.pth'))
    net.eval()
    net.to(device)

    """
    Evaluation
    """
    data_transform = transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.ImageFolder(root='hw5_data/test',
                                        transform=data_transform)
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader_test:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (
        len(data_loader_test), 100 * correct / total))
