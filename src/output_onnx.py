
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import time


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        # 定义网络层
        self.conv1 = nn.Conv2d(3, 10, 3)  #12 -> 10 -> maxp -> 5
        self.conv2 = nn.Conv2d(10, 16, 3) #5 -> 3
        self.conv3 = nn.Conv2d(16, 32, 3) #3 -> 1

        self.face_det = nn.Conv2d(32, 2, 1) #1 -> 1
        self.bbox = nn.Conv2d(32, 4, 1) #1 -> 1
        self.landmark = nn.Conv2d(32, 10, 1) #1 -> 1

    def forward(self, x):
        # 定义前向传播
        x = F.relu(self.conv1(x)) #10
        x = F.max_pool2d(x, 2) #5
        x = F.relu(self.conv2(x)) #3
        x = F.relu(self.conv3(x)) #1

        facedet = self.face_det(x)
        bbox = self.bbox(x)
        landmark = self.landmark(x)

        return facedet, bbox, landmark


net1 = torch.load(r"C:\Users\lucyc\Desktop\model\Pnet\Col_HardNg_2\Pnet_epoch_170.pth")

p_net = PNet()
p_net.load_state_dict(net1.state_dict())

torch_input = torch.randn(1, 3, 12, 12)
torch.onnx.export(p_net, torch_input, "TTTT.onnx", verbose=True)
