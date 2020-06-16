# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd

batch_size = 128

if __name__ == '__main__':
    """
    Load the data. Seperate it into training and validation set.
    """

    # Define the transform apply on the dataset.
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop((228, 228)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # Load the dataset.
    train_dataset = datasets.ImageFolder(root='hw5_data/train',
                                    transform=data_transform)
    data_classes = train_dataset.classes

    # Split the dataset into training and testing set.
    torch.manual_seed(1)

    # Define training and validation data loaders.
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    """
    Build a CNN model.
    """

    device = torch.device('cuda') if torch.cuda.is_available()\
                                else torch.device('cpu')

    # Get the pretrained model.
    model = torchvision.models.resnet18(pretrained=True)
    # Modify the ouput layer to fit our task.
    num_features = model.fc.in_features
    # Our dataset has 13 classes.
    num_classes = 15
    model.fc = torch.nn.Linear(num_features, num_classes)
    # move model to the right device
    model.to(device)

    # Construct a criterion and an optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # And a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=50,
    #                                             gamma=0.1)

    # let's train it for @num_epochs epochs
    num_epochs = 200
    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        # update the learning rate
        # lr_scheduler.step()
        # Save checkpoint
        if epoch > 0 and epoch%10 == 0:
            torch.save(model.state_dict(), f'./checkpoints/net_{epoch}.pth')

    print('Finished Training')

    """
    Save the parameters of the trained model.
    """

    # Save result.
    model_params_result_path = './checkpoints/hw1_net.pth'
    torch.save(model.state_dict(), model_params_result_path)

    """
    Create a new network to load the trained model.
    """

    # Load Model.
    net = torchvision.models.resnet18(pretrained=False)
    net.fc = torch.nn.Linear(num_features, num_classes)
    net.load_state_dict(torch.load(model_params_result_path))
    net.eval()
    net.to(device)

    """
    Evaluation
    """
    test_dataset = datasets.ImageFolder(root='hw5_data/test',
                                        transform=transforms.Compose([
                                            transforms.Resize((228, 228)),
                                            transforms.ToTensor(),
                                        ]))
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
