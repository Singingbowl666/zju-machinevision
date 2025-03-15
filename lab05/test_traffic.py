import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, ResNet
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import MultiStepLR # 引入学习率调度器
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torchvision.datasets import ImageFolder



class ResNetCustom(ResNet):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        # 第一个卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    return ResNetCustom(BasicBlock, [2, 2, 2, 2], **kwargs)

# 假设模型类为 ResNetCustom
# 加载整个模型

if __name__ == '__main__':
    # 定义测试集的数据变换（保持与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize 到合适的大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_set = ImageFolder(root='./Traffic_sign/test_dataset', transform=transform)


    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256,
                                            shuffle=False, num_workers=4)



    model = resnet18(num_classes=len(test_set.classes))
    model.load_state_dict(torch.load('traffic_model_params.pth'))

    # 创建评估引擎
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            'accuracy': Accuracy(),
        }
    )

    # 运行评估
    state = evaluator.run(test_loader)

    # 打印测试结果
    accuracy = state.metrics['accuracy']
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

