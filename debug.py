import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader

from config import BATCH_SIZE
from models.vit import PatchEmbedding



def debug():
    # transformation of PIL data into tensor format
    transformation_operation = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformation_operation)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation_operation)
    # using dataloader to prepare data for neural network
    train_data = dataloader.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data = dataloader.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # print("length of train data = ", len(train_data)) # length of train data =  938
    # print("length of val data = ", len(val_data)) # length of val data =  157

    images, labels = next(iter(train_data))
    print("This is shape of input image tensor: ", images.shape) # Images shape:  torch.Size([64, 1, 28, 28])
    # print("Labels shape: ", labels.shape) # Labels shape:  torch.Size([64])
    patch_embed = PatchEmbedding()
    embedded_img = patch_embed.patch_embed(images)
    print("This is shape of input image tensor after conv2d", embedded_img.shape) # torch.Size([64, 20, 4, 4]) 
    # but we want (64, 16, 20) last number should be embed_dim
    print(embedded_img.flatten(2).shape) # torch.Size([64, 20, 16]) preserve 3 dimensions
    print(embedded_img.flatten(2).transpose(1, 2).shape) # torch.Size([64, 16, 20]) -- 0th dim untouched. Swap 1st and 2nd dim.
    # torch.Size([64, 16, 20]) == (batch_size, total no. of patches per img, embedding dim each img)


if __name__ == '__main__':
    debug()