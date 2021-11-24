# Hyper-parameter.
LR = 1e-3
BATCH_SIZE = 32
EPOCH = 20
MOMENTUM = .9
WEIGHT_DECAY = 1e-4

# image tags in Cifar10.
CLASSES = (
    'airplane', 'autombile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck')

class_dict = dict(zip(range(10), CLASSES))
