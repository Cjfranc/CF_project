import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import models
from torchvision.datasets import VOCDetection

class CenterNet(nn.Module):
  def __init__(self, C):
    '''
    CenterNet model
    with class C
    '''
    super(CenterNet, self).__init__()
    
    # Pre-trained backbone
    self.resnet18 = models.resnet18(pretrained=True)
    
    self.f0 = nn.Sequential(*list(self.resnet18.children())[0:4])
    
    self.f1 = nn.Sequential(*list(self.resnet18.layer1))
    self.f2 = nn.Sequential(*list(self.resnet18.layer2))
    self.f3 = nn.Sequential(*list(self.resnet18.layer3))
    self.f4 = nn.Sequential(*list(self.resnet18.layer4))
    
    # Up sampling layers
    self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
    self.bn1 = torch.nn.BatchNorm2d(256)
    self.up_conv1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)

    self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
    self.bn2 = torch.nn.BatchNorm2d(128)
    self.up_conv2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)

    self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
    self.bn3 = torch.nn.BatchNorm2d(64)
    self.up_conv3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
    
    
    
    # Head layers
    self.hm_conv1 = nn.Conv2d(64, 256, 3, padding=1)
    self.hm_conv2 = nn.Conv2d(256, C, 1)
    
    self.s_conv1 = nn.Conv2d(64, 256, 3, padding=1)
    self.s_conv2 = nn.Conv2d(256, 2, 1)

        
  def forward(self, x):
    #transfer learning layers
    x0 = self.f0(x) # (..., 64, H/4, W/4)
    
    x1 = self.f1(x0) # (..., 64, H/4, W/4)
    x2 = self.f2(x1) # (..., 128, H/8, W/8)
    x3 = self.f3(x2) # (..., 256, H/16, W/16)
    x4 = self.f4(x3) # (..., 512, H/32, W/32)
    
    # Upsample
    up1 = self.conv1(x4) # (..., 256, H/16, W/16)
    up1 = F.relu(self.bn1(up1))
    up1 = F.relu(self.up_conv1(up1))

    up2 = self.conv2(up1) # (..., 128, H/8, W/8)
    up2 = F.relu(self.bn2(up2))
    up2 = F.relu(self.up_conv2(up2))
    
    up3 = self.conv3(up2) # (..., 64, H/4, W/4)
    up3 = F.relu(self.bn3(up3))
    up3 = F.relu(self.up_conv3(up3))
    

    # Keypoint obtained with heatmap
    hm = F.relu(self.hm_conv1(up3)) # (..., 256, H/4, W/4)
    hm = torch.sigmoid(self.hm_conv2(hm)) # (..., C, H/4, W/4)
    
    # Size
    size = F.relu(self.s_conv1(up3)) # (..., 256, H/4, W/4)
    size = self.s_conv2(size) # (..., 2, H/4, W/4)
    
    return hm, size