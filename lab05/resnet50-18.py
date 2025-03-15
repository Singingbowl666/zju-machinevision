import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, ResNet

# Custom ResNet model for CIFAR-100
class ResNetCustom(ResNet):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        # First convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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

def resnet18(num_classes=100):
    return ResNetCustom(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# Knowledge Distillation Loss Functions
def soft_target_loss(student_logits, teacher_logits, temperature=3.0):
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    student_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
    return torch.nn.functional.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def logits_loss(student_logits, teacher_logits):
    return torch.nn.functional.mse_loss(student_logits, teacher_logits)

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(default_config_files=['config.yml'])
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    parser.add_argument('--root', required=False, type=str, default='./data', help='data root path')
    parser.add_argument('--workers', required=False, type=int, default=4, help='number of workers for data loader')
    parser.add_argument('--bsize', required=False, type=int, default=256, help='batch size')
    parser.add_argument('--epochs', required=False, type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.1, help='learning rate')
    parser.add_argument('--device', required=False, default=0, type=int, help='CUDA device id for GPU training')

    options = parser.parse_args()

    root = options.root
    bsize = options.bsize
    workers = options.workers
    epochs = options.epochs
    lr = options.lr
    device = 'cpu' if options.device is None else torch.device(f'cuda:{options.device}')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize, shuffle=True, num_workers=workers)

    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize, shuffle=False, num_workers=workers)

    # Initialize teacher model
    teacher_model = torchvision.models.resnet50(pretrained=True).to(device)
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 100).to(device)
    teacher_model.eval()

    # Loss types to compare
    loss_types = ["soft_target", "logits"]
    results = {}

    # 引入交叉熵损失函数
    criterion_supervised = nn.CrossEntropyLoss()

    # 调整权重系数
    alpha = 0.5  # 监督损失权重
    beta = 0.25   # soft target损失权重
    gamma = 0.25  # logits损失权重

    for loss_type in loss_types:
        print(f"\nTraining with {loss_type} loss function.")
        student_model = resnet18(num_classes=100).to(device)
        optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(epochs):
            student_model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                student_logits = student_model(images)
                
                # 获取教师模型输出
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                
                # 计算监督损失
                L_supervised = criterion_supervised(student_logits, labels)

                # 计算 soft target 和 logits 损失
                if loss_type == "soft_target":
                    L_distill = soft_target_loss(student_logits, teacher_logits)
                    L_logits = torch.tensor(0).to(device)  # 不使用logits损失
                elif loss_type == "logits":
                    L_distill = torch.tensor(0).to(device)  # 不使用soft target损失
                    L_logits = logits_loss(student_logits, teacher_logits)
                else:
                    raise ValueError("Unsupported loss type")

                # 综合损失函数
                L_total = alpha * L_supervised + beta * L_distill + gamma * L_logits

                # 反向传播和优化
                optimizer.zero_grad()
                L_total.backward()
                optimizer.step()
                total_loss += L_total.item()

            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)

            # 评估
            student_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = student_model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            epoch_accuracies.append(accuracy)

            print(f'Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        results[loss_type] = {"losses": epoch_losses, "accuracies": epoch_accuracies}
