import torch
import torch.nn as nn
import torchvision.models as models
from yolov6.layers.common import CounterA

class GaterNetwork(nn.Module):
    def __init__(self, feature_extractor_arch, num_features, num_filters, sections, bottleneck_size, gtg_threshold = 0.0):
        super().__init__()
        # Feature extractor (E)
        self.feature_extractor = feature_extractor_arch(pretrained=False)
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
        
        if self.enable_fixed_gates:
            return self.fixed_gates
        
        # Feature extraction
        f = self.feature_extractor(x)
        f = torch.flatten(f, 1)
        
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
        
        return section_gates_list, closed_gates_percentage_beta
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    @staticmethod
    def create_feature_extractor_resnet101(pretrained=True):
        # For example, use ResNet18 as the base model
        model = models.resnet101(pretrained=pretrained)
        # Remove the final fully connected layer (classifier)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        return feature_extractor
    
    @staticmethod
    def create_feature_extractor_resnet18(pretrained=True):
        # For example, use ResNet18 as the base model
        model = models.resnet18(pretrained=pretrained)
        # Remove the final fully connected layer (classifier)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        return feature_extractor
