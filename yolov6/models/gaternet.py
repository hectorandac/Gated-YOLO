import torch
import torch.nn as nn
import torchvision.models as models
from yolov6.layers.common import CounterA
from yolov6.tools.models.darknet import Darknet53
from yolov6.tools.models.darknet import DarkResidualBlock

class DimensionalityReduction(nn.Module):
    def __init__(self, input_channels, bottleneck_size):
        super(DimensionalityReduction, self).__init__()
        self.conv1x1 = nn.Conv2d(input_channels, bottleneck_size, kernel_size=1)  # Bottleneck with 1x1 convolution
        self.bn = nn.BatchNorm2d(bottleneck_size)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)
        return x

class GaterNetwork(nn.Module):
    def __init__(self, feature_extractor_arch, num_features, num_filters, sections, bottleneck_size, gtg_threshold=-0.5, feature_extractors=None):
        super().__init__()
        
        # Feature extractor (E)
        if feature_extractors != None:
            self.feature_extractors = nn.ModuleList(feature_extractors)
            self.feature_extractor = None
        else:
            self.feature_extractor, self.fe_last_layer = feature_extractor_arch(pretrained=False)
            self.feature_extractors = None


        self.gtg_threshold = gtg_threshold
        
        # Fully-connected layers with bottleneck (D) and Adaptive Pooling
        self.fc1 = nn.Linear(num_features, bottleneck_size)
        self.bn1 = nn.BatchNorm1d(bottleneck_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(bottleneck_size, num_filters)

        self.sections = sections
        self.num_filter = num_filters
        self.enable_fixed_gates = False

    def forward(self, x, training=False, epsilon=None):
        CounterA.reset()

        f_out = None
        
        if self.enable_fixed_gates:
            return self.fixed_gates
        
        # Feature extraction
        if self.feature_extractors == None:
            f = self.feature_extractor(x)
            f_out = self.fe_last_layer(f.view(-1, 1024))
            f = torch.flatten(f, 1)
        else:
            combined_features = []
        
            for input_tensor, feature_extractor in zip(x, self.feature_extractors):
                # Feature extraction
                f = feature_extractor(input_tensor)
                f = torch.flatten(f, 1)
                combined_features.append(f)
            
            # Concatenate all extracted features
            f = torch.cat(combined_features, dim=1)
        
        # Bottleneck mapping
        f0 = self.fc1(f)
        f0 = self.bn1(f0)
        f0 = self.relu1(f0)
        
        # Real-valued vector before binarization
        g0 = self.fc2(f0)

        closed_gates_percentage_beta = 0
        
        if training:
            ## SEM HASH
            # During training, add noise and use both g_alpha (real-valued) and g_beta (binary)
            noise = torch.randn_like(g0) * epsilon
            g0_noisy = g0 + noise
            g_alpha = torch.clamp(1.2 * torch.sigmoid(g0_noisy) - 0.1, 0, 1)
            g_beta = (g0_noisy > self.gtg_threshold).float()
            g = g_beta + g_alpha - g_alpha.detach()
            
            closed_gates_percentage_beta = (g_beta == 0).float().mean().item() * 100

        else:
            # During inference, always use the binary gates
            g = (g0 > self.gtg_threshold).float()

        section_gates_list = []
        start_idx = 0
        for end_idx in self.sections:
            section_gates_list.append([g[:, start_idx:end_idx].unsqueeze(-1).unsqueeze(-1), None])
            start_idx = end_idx
        
        return section_gates_list, closed_gates_percentage_beta, f_out
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    @staticmethod
    def create_feature_extractor_resnet101(pretrained=True):
        model = models.resnet101(pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        last_layer = nn.Sequential(*list(model.children())[-1:])
        return feature_extractor, last_layer
    
    @staticmethod
    def create_feature_extractor_resnet18(pretrained=True):
        model = models.resnet18(pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        last_layer = nn.Sequential(*list(model.children())[-1:])
        return feature_extractor, last_layer
    
    @staticmethod
    def create_feature_extractor_darknet53(pretrained=False):
        model = Darknet53(DarkResidualBlock, num_classes=5)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        last_layer = nn.Sequential(*list(model.children())[-1:])
        return feature_extractor, last_layer
    
    @staticmethod
    def dimensionality_reduction(input_channels, bottleneck_size):
        return DimensionalityReduction(input_channels, bottleneck_size)
