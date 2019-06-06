import torch
from torch import nn
from torch import optim
from torchvision import transforms, models
import torch.nn.functional as F
import datetime
import os
from Utilities import preprocess_utility


class Network(nn.Module):
    def __init__(self, input_length, output_length, hidden_layers, drop_prob=0):
        ''' To Create a Neural network with configurable number of hidden layers,input and      output lengths.

            input_length: size of the input(int)
            output_length: size of the output(int)
            hidden_layers: A list Containing the nodes in each hidden layer, starting with input layer
            drop_prob: Dropout probability, value between 0-1 (float)
        '''
        super().__init__()
        # Add the first layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_length,
                                                      hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([
            nn.Linear(h_input, h_output) for h_input, h_output in layer_sizes])

        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_length)

        # Include dropout
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # Forward through each hidden layer with ReLU and dropout
        for layer in self.hidden_layers:
            x = F.relu(layer(x))  # Apply activation
            x = self.dropout(x)  # Apply dropout

        # Pass through output layer
        x = self.output(x)

        return F.log_softmax(x, dim=1)