# ResNet10 on CIFAR-100

pytorch版本最好是1.10

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run the example on the CPU:

```bash
python resnet-cifar100.py -c config.yml
```

Run the example on the GPU (device 0):

```bash
python resnet-cifar100.py -c config.yml --device 0
```