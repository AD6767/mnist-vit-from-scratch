import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader

from config import BATCH_SIZE, LEARNING_RATE, EPOCHS
from models.vit import VisionTransformer


def main():
    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # transformation of PIL data into tensor format
    transformation_operation = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformation_operation)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation_operation)
    # using dataloader to prepare data for neural network
    train_data = dataloader.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data = dataloader.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = VisionTransformer().to(device=device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # train loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_epoch = 0
        correct_epoch = 0
        print(f"\nEpoch {epoch + 1}")

        for batch_idx, (images, labels) in enumerate(train_data):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # gradient descent
            optimizer.step() # update weights
            total_loss += loss.item()

            preds = outputs.argmax(dim=1) # max probablity of the class. We will have NUM_CLASSES prob for each image, we get the max.
            correct = (preds == labels).sum().item() # get count of all correct predictions.
            accuracy = (correct / labels.size(0)) * 100.0

            correct_epoch += correct
            total_epoch += labels.size(0)

            if batch_idx % 100 == 0:
                print(f" Batch {batch_idx+1:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%")

        epoch_acc = (correct_epoch / total_epoch) * 100.0
        print(f"==> Epoch {epoch + 1} Summary: Total Loss = {total_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
    # val accuracy compute
    compute_accuracy(model, val_data, device)
    # plot 10 test images with predictions
    visualize(model, val_data, device)


def compute_accuracy(model, val_data, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_data:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = 100.0 * correct / total
    print(f"\n==> Val Accuracy: {test_acc:.2f}%")

def visualize(model, val_data, device):
    model.eval()
    images, labels = next(iter(val_data))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        output = model(images)
        preds = output.argmax(dim=1)

    # Plot 10 test images with predictions
    plt.figure(figsize=(12, 4))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[i].squeeze().cpu(), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    


if __name__ == '__main__':
    main()
