import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from yolov6.layers.common import CounterA

class GaterNetwork(nn.Module):
    def __init__(self, feature_extractor_arch, num_features, num_filters, sections, bottleneck_size):
        super().__init__()
        # Feature extractor (E)
        self.feature_extractor = feature_extractor_arch(pretrained=False)
        
        # Fully-connected layers with bottleneck (D) and Adaptive Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((20, 20))
        self.fc1 = nn.Linear(num_features, bottleneck_size)
        self.fc2 = nn.Linear(bottleneck_size, num_filters)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.sections = sections

        self.enable_fixed_gates = False 

    def forward(self, x, training=False, epsilon=None):
        if self.enable_fixed_gates:
            return self.fixed_gates
        
        CounterA.reset()
        # Feature extraction
        f = self.feature_extractor(x)
        f = self.adaptive_pool(f)
        f = torch.flatten(f, 1)
        
        # Bottleneck mapping
        f0 = self.fc1(f)
        f0 = self.batch_norm(f0)
        f0 = F.relu(f0)
        
        # Real-valued vector before binarization
        g0 = self.fc2(f0)
        
        if training:
            ## SEM HASH
            # During training, add noise and use both g_alpha (real-valued) and g_beta (binary)
            noise = torch.randn_like(g0) * epsilon
            g0_noisy = g0 + noise
            g_alpha = torch.clamp(1.2 * torch.sigmoid(g0_noisy) - 0.1, 0, 1)
            g_beta = (g0_noisy > 0).float()
            g = g_beta if torch.rand(1).item() < 0.5 else g_alpha
        else:
            # During inference, always use the binary gates
            g = (g0 > 0).float()

        section_gates_list = []
        start_idx = 0
        for end_idx in self.sections:
            section_gates_list.append([g[:, start_idx:end_idx].unsqueeze(-1).unsqueeze(-1), None])
            start_idx = end_idx
        
        return section_gates_list
    
    @staticmethod
    def create_feature_extractor_resnet18(pretrained=True):
        # For example, use ResNet18 as the base model
        model = models.resnet18(pretrained=pretrained)
        # Remove the final fully connected layer (classifier)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        return feature_extractor
