import torch
import torch.nn as nn
from torch.multiprocessing import freeze_support


"""
Feed Forward NueralNet with 2 hidden layers - Network name called NueroLine
Contains 2 hidden layers 

Bag of words as input , 
One layer connected , number of patterns as input , then two hidden layers, then output size the number of classes, then apply softmax to get probability of each classes

"""


class NueroLine(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NueroLine, self).__init__()

        # hidden size only can change
        self.l1 = nn.Linear(input_size, hidden_size)  # first layer
        self.l2 = nn.Linear(hidden_size, hidden_size)  # second layer
        self.l3 = nn.Linear(hidden_size, num_classes)  # third layer

        # RELU activation function
        self.relu = nn.ReLU()

    # implement forward pass
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax , because apply cross entropy loss
        return out
