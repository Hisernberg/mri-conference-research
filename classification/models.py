"""Models adapted from the supplied classification notebook; see audit for provenance.
The no-multiscale control keeps all branches and parameters but uses dilation 1.
ResNet2D18 is the original narrow GroupNorm implementation, not torchvision ResNet.
"""
import torch
from torch import nn

class CONFIG:
    IN_CHANNELS=1
    BASE_CHANNELS=16
    DROPOUT=0.4

def norm_layer(channels):
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)

def act_layer():
    return nn.ReLU(inplace=False)

class ChannelAttention2D(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), act_layer(), nn.Conv2d(hidden, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class SpatialAttention2D(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM2D(nn.Module):

    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention2D(channels, reduction)
        self.sa = SpatialAttention2D(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class MultiScaleResBlock2D(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, use_attention=True, multiscale=True):
        super().__init__()
        if out_ch < 3:
            raise ValueError('out_ch must be at least 3 for the three-branch block')
        branch_ch = out_ch // 3
        rem = out_ch - branch_ch * 3
        self.b1 = nn.Conv2d(in_ch, branch_ch, 3, stride=stride, padding=1, dilation=1, bias=False)
        self.b2 = nn.Conv2d(in_ch, branch_ch, 3, stride=stride, padding=2 if multiscale else 1, dilation=2 if multiscale else 1, bias=False)
        self.b3 = nn.Conv2d(in_ch, branch_ch + rem, 3, stride=stride, padding=3 if multiscale else 1, dilation=3 if multiscale else 1, bias=False)
        self.bn1 = norm_layer(out_ch)
        self.act = act_layer()
        self.fuse = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn2 = norm_layer(out_ch)
        self.use_attention = use_attention
        self.attn = CBAM2D(out_ch) if use_attention else None
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), norm_layer(out_ch))

    def forward(self, x):
        identity = x
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.act(self.bn1(out))
        out = self.bn2(self.fuse(out))
        if self.attn is not None:
            out = self.attn(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(out + identity)

def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class MSCANet2D(nn.Module):

    def __init__(self, num_classes, in_channels=None, base=None, use_attention=True, dropout=None, multiscale=True):
        super().__init__()
        in_channels = CONFIG.IN_CHANNELS if in_channels is None else in_channels
        base = CONFIG.BASE_CHANNELS if base is None else base
        dropout = CONFIG.DROPOUT if dropout is None else dropout
        self.stem = nn.Sequential(nn.Conv2d(in_channels, base, 3, stride=1, padding=1, bias=False), norm_layer(base), act_layer(), nn.MaxPool2d(2))
        chs = [base, base * 2, base * 4, base * 8]
        self.stage1 = MultiScaleResBlock2D(chs[0], chs[1], stride=2, use_attention=use_attention, multiscale=multiscale)
        self.stage2 = MultiScaleResBlock2D(chs[1], chs[2], stride=2, use_attention=use_attention, multiscale=multiscale)
        self.stage3 = MultiScaleResBlock2D(chs[2], chs[3], stride=2, use_attention=use_attention, multiscale=multiscale)
        self.stage4 = MultiScaleResBlock2D(chs[3], chs[3], stride=2, use_attention=use_attention, multiscale=multiscale)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(chs[3], 128), act_layer(), nn.Dropout(dropout / 2), nn.Linear(128, num_classes))
        self.target_layer_name = 'stage4'
        init_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.classifier(self.gap(x))

class Plain2DCNN(nn.Module):

    def __init__(self, num_classes, in_channels=None, base=None, dropout=None):
        super().__init__()
        in_channels = CONFIG.IN_CHANNELS if in_channels is None else in_channels
        base = CONFIG.BASE_CHANNELS if base is None else base
        dropout = CONFIG.DROPOUT if dropout is None else dropout

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm_layer(cout), act_layer(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(in_channels, base), block(base, base * 2), block(base * 2, base * 4), block(base * 4, base * 8), block(base * 8, base * 8))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(base * 8, 128), act_layer(), nn.Linear(128, num_classes))
        self.target_layer_name = 'features'
        init_weights(self)

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = norm_layer(out_ch)
        self.act = act_layer()
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), norm_layer(out_ch))

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(out + identity)

class ResNet2D18(nn.Module):

    def __init__(self, num_classes, in_channels=None, base=None):
        super().__init__()
        in_channels = CONFIG.IN_CHANNELS if in_channels is None else in_channels
        base = CONFIG.BASE_CHANNELS if base is None else base
        self.stem = nn.Sequential(nn.Conv2d(in_channels, base, 7, stride=2, padding=3, bias=False), norm_layer(base), act_layer(), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(BasicBlock2D(base, base), BasicBlock2D(base, base))
        self.layer2 = nn.Sequential(BasicBlock2D(base, base * 2, stride=2), BasicBlock2D(base * 2, base * 2))
        self.layer3 = nn.Sequential(BasicBlock2D(base * 2, base * 4, stride=2), BasicBlock2D(base * 4, base * 4))
        self.layer4 = nn.Sequential(BasicBlock2D(base * 4, base * 8, stride=2), BasicBlock2D(base * 8, base * 8))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(base * 8, num_classes))
        self.target_layer_name = 'layer4'
        init_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.gap(x))

def build_model(name, num_classes):
    if name == 'MSCANet2D':
        return MSCANet2D(num_classes, use_attention=True)
    if name == 'MSCANet2D_NoAttention':
        return MSCANet2D(num_classes, use_attention=False)
    if name == 'Plain2DCNN':
        return Plain2DCNN(num_classes)
    if name == 'ResNet2D18':
        return ResNet2D18(num_classes)
    raise ValueError(f'Unknown model name: {name}')
