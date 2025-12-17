import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

#learn f(x) such that x is the wt and f(x) is the CTCFKO - then i have to clean and label in a different way.
#or x is the CTCFKO and f(x) is the DKO
# class LastLayerNN(nn.Module):
#     def __init__(self):
#         super(LastLayerNN, self).__init__()
#         self.net = nn.Sequential(nn.Linear(83, 2),
#             nn.GELU(),
#             nn.Softmax(dim=-1),
#             )
#     def forward(self, x1, x2):
#         return self.net(x1-x2)

import torch
import torch.nn as nn
import numpy as np

class TripletNet(nn.Module):
    def __init__(self, mask=False):
        super(TripletNet, self).__init__()
        if mask:
            mask = np.tril(np.ones(256), k=-3) + np.triu(np.ones(256), k=3)
            self.mask = nn.Parameter(torch.from_numpy(np.array(mask)).to(torch.int32), requires_grad=False)

    def mask_data(self, x):
        if hasattr(self, "mask"):
            x = torch.mul(self.mask, x)
        return x

    def forward_one(self, x):
        raise NotImplementedError

    def forward(self, anchor, positive, negative):
        # Apply mask to all three inputs
        anchor = self.mask_data(anchor)
        positive = self.mask_data(positive)
        negative = self.mask_data(negative)

        # Get embeddings for all three inputs
        anchor_out = self.forward_one(anchor)
        positive_out = self.forward_one(positive)
        negative_out = self.forward_one(negative)
        return anchor_out, positive_out, negative_out

# refer to the weights of SLeNet
class TripletLeNet(TripletNet):
    def __init__(self, *args, **kwargs):
        super(TripletLeNet, self).__init__(*args, **kwargs)
        
        # Feature extraction layers (CNN part)
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 6, 5, 1),
            nn.BatchNorm2d(6),      
            nn.ReLU(),              
            nn.MaxPool2d(2, stride=2),
            
            # Layer 2
            nn.Conv2d(6, 16, 5, 1),
            nn.BatchNorm2d(16),     
            nn.ReLU(),              
            nn.MaxPool2d(2, stride=2),
        )
        
        # Embedding layers (FC part)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            
            nn.Linear(16 * 61 * 61, 120),
            nn.BatchNorm1d(120),    
            nn.GELU(),
            
            nn.Linear(120, 83),
            nn.BatchNorm1d(83),     
            nn.GELU(),
        )

    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1) 
        x = self.linear(x)
        return x

    def compute_distances(self, anchor_out, positive_out, negative_out):
        # Calculate Euclidean distances between anchor-positive and anchor-negative
        pos_dist = torch.norm(anchor_out - positive_out, dim=1, p=2)
        neg_dist = torch.norm(anchor_out - negative_out, dim=1, p=2)
        return pos_dist, neg_dist

# resnet
class TripletResNet(nn.Module):
    def __init__(self, mask=False, embedding_dim=128, backbone="resnet18"):
        super(TripletResNet, self).__init__()

        # optional masking
        if mask:
            mask = np.tril(np.ones(256), k=-3) + np.triu(np.ones(256), k=3)
            self.mask = nn.Parameter(torch.from_numpy(np.array(mask)).to(torch.int32),requires_grad=False)

        # load ResNet backbone
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=False)
        elif backbone == "resnet34":
            base_model = models.resnet34(pretrained=False)
        else:
            raise ValueError(f"Backbone {backbone} not supported")

        # modify first conv layer to accept 1-channel Hi-C input (instead of 3-channel RGB)
        base_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # remove classifier head, keep feature extractor
        modules = list(base_model.children())[:-1]  # remove fc layer
        self.feature_extractor = nn.Sequential(*modules)

        # add projection head to get embeddings of size embedding_dim
        self.embedding_layer = nn.Linear(base_model.fc.in_features, embedding_dim)

    def mask_data(self, x):
        if hasattr(self, "mask"):
            x = torch.mul(self.mask, x)
        return x

    def forward_one(self, x):
        x = self.feature_extractor(x)   # [batch, 512, 1, 1] for resnet18
        x = torch.flatten(x, 1)
        x = self.embedding_layer(x)     # [batch, embedding_dim]
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalize embeddings
        return x

    def forward(self, anchor, positive, negative):
        anchor = self.mask_data(anchor)
        positive = self.mask_data(positive)
        negative = self.mask_data(negative)

        anchor_out = self.forward_one(anchor)
        positive_out = self.forward_one(positive)
        negative_out = self.forward_one(negative)

        return anchor_out, positive_out, negative_out




# ------below keep the same as the original code-----

class SiameseNet(nn.Module):
    def __init__(self, mask=False):
        super(SiameseNet, self).__init__()
        if mask:
            mask = np.tril(np.ones(256), k=-3)+np.triu(np.ones(256), k=3)
            self.mask = nn.Parameter(torch.from_numpy(np.array(mask)).to(torch.int32),requires_grad=False)

    def mask_data(self, x):
        if hasattr(self, "mask"): x=torch.mul(self.mask, x)
        return x
    def forward_one(self, x):
        raise NotImplementedError
    def forward(self, x1, x2):
        x1, x2 = self.mask_data(x1), self.mask_data(x2)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

class SLeNet(SiameseNet):
    def __init__(self, *args, **kwargs):
        super(SLeNet, self).__init__(*args, **kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, 1),
            nn.MaxPool2d(2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(16*61*61, 120),
            nn.GELU(),
            nn.Linear(120, 83),
            nn.GELU(),
            )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

class SAlexNet(SiameseNet):
    def __init__(self, *args, **kwargs):
        super(SAlexNet, self).__init__(*args, **kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.GELU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.GELU(),
            nn.Linear(in_features=4096, out_features=83),
        )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

class SZFNet(SiameseNet):
    def __init__(self, *args, **kwargs):
        super(SZFNet, self).__init__(*args, **kwargs)
        self.channels = 1
        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()
    def get_conv_net(self):
        layers = []
        # in_channels = self.channels, out_channels = 96
        # kernel_size = 7x7, stride = 2
        layer = nn.Conv2d(
            self.channels, 96, kernel_size=7, stride=2, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))
        # in_channels = 96, out_channels = 256
        # kernel_size = 5x5, stride = 2
        layer = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))
        # in_channels = 256, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        # in_channels = 384, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        # in_channels = 384, out_channels = 256
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        return nn.Sequential(*layers)
    def get_fc_net(self):
        layers = []
        # in_channels = 9216 -> output of self.conv_net
        # out_channels = 4096
        layer = nn.Linear(256*7*7, 4096)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())
        # in_channels = 4096
        # out_channels = self.class_count
        layer = nn.Linear(4096, 83)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())
        return nn.Sequential(*layers)
    def forward_one(self, x):
        y = self.conv_net(x)
        y = y.view(-1, 7*7*256)
        y = self.fc_net(y)
        return y