import torch
import torch.nn as nn

class Actor(nn.Module):

    def __init__(self,in_features,out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features,16)
        self.layer2 = nn.Linear(16,16)
        #making a network with two heads, one for mean and one for std
        self.mean_layer = nn.Linear(16,out_features)
        self.std_layer = nn.Linear(16,out_features)

    def forward(self,x):
        x = self.layer1(x)
        x = nn.functional.tanh(x)
        x = self.layer2(x)
        x = nn.functional.tanh(x)

        mean = self.mean_layer(x)

        log_std = self.std_layer(x)
        std = torch.exp(log_std)

        return mean,std