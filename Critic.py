import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features,16)
        self.layer2 = nn.Linear(16,16)
        self.layer3 = nn.Linear(16,1)

    def forward(self,x):
        x = self.layer1(x)
        x = nn.functional.tanh(x)
        x = self.layer2(x)
        x = nn.functional.tanh(x)
        return self.layer3(x)
