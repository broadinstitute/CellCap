"""Classifier with gradient reversal"""

import torch
from .gradient_reversal import GradientReverseModule
from ..utils import init_weights


class AdvNet(torch.nn.Module):
    def __init__(self, in_feature=20, hidden_size=20, out_dim=2):
        super(AdvNet, self).__init__()
        self.ad_layer1 = torch.nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = torch.nn.Linear(hidden_size, out_dim)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm1d(hidden_size)
        self.norm2 = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(init_weights)
        self.grl = GradientReverseModule()

    def forward(self, x, reverse=True, if_activation=True):
        if reverse:
            x = self.grl(x)
        x = self.ad_layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        if if_activation:
            y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
