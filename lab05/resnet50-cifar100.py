import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import Bottleneck, ResNet
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint


results = []


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


def resnet50(**kwargs):
    return ResNetCustom(Bottleneck, [3, 4, 6, 3], **kwargs)


def logger(engine, model, evaluator, loader, pbar):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    pbar.log_message(
        "Test Results - Avg accuracy: {:.2f}".format(avg_accuracy)
    )
    results.append(avg_accuracy)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=False,
                        is_config_file=True, help='config file')
    parser.add_argument('--root', required=False, type=str, default='./data',
                        help='data root path')
    parser.add_argument('--workers', required=False, type=int, default=4,
                        help='number of workers for data loader')
    parser.add_argument('--bsize', required=False, type=int, default=256,
                        help='batch size')
    parser.add_argument('--epochs', required=False, type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--device', required=False, default=0, type=int,
                        help='CUDA device id for GPU training')
    options = parser.parse_args()

    root = options.root
    bsize = options.bsize
    workers = options.workers
    epochs = options.epochs
    lr = options.lr
    print(lr)
    device = 'cpu' if options.device is None \
        else torch.device('cuda:{}'.format(options.device))

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR100(root=root, train=True,
                                             download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize,
                                               shuffle=True, num_workers=workers)

    test_set = torchvision.datasets.CIFAR100(root=root, train=False,
                                            download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize,
                                              shuffle=False, num_workers=workers)


    # define model
    model = resnet50(num_classes=100)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    # define optimizer with Adam
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    # define optimizer with Adagrad
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.0001)


    # create ignite engines
    trainer = create_supervised_trainer(model=model,
                                        optimizer=optimizer,
                                        loss_fn=criterion,
                                        device=device)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy()},
                                            device=device)

    # ignite handlers
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger, model, evaluator, test_loader, pbar)

    checkpoint_handler = ModelCheckpoint('models', 'model', save_interval=1, n_saved=5)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})

    # start training
    t0 = time.time()
    trainer.run(train_loader, max_epochs=epochs)
    t1 = time.time()
    print('Model Accuracy:', results[-1])
    print('Total time:', t1 - t0)
