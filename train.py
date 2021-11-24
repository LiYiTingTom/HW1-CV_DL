import torch
from torch import nn, optim

from tensorboardX import SummaryWriter

import utils
from model import VGG16
from config import *

# Setup device.
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# Get data loader.
train_loader, test_loader = utils.get_data_loader()


# using VGG16.
net = VGG16().to(device)


# Loss Func & Optimizer.
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR,
                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


# Record writer.
writer = SummaryWriter(log_dir='./log')


# Start Training.
for epoch in range(EPOCH):
    print(f'{"="*100}\nEpoch: {epoch+1}')
    net.train()
    total_loss = 0.0
    accuracy = 0.0
    total = 0.0

    for i, data in enumerate(train_loader):
        length = len(train_loader)
        inputs, labels = data
        # Move to device.
        inputs, labels = inputs.to(device), labels.to(device)

        # Gradient Descent.
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # Recording.
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        accuracy += predicted.eq(labels.data).cpu().sum()

        print(f'[epoch: {epoch+1:3d} | iter: {i+1 + epoch*length:4d} | Loss: {total_loss/(i+1):.3f} | Accu: {100*accuracy / total:.3f}%', end='\r')

    writer.add_scalar('loss/train', total_loss/(i+1), epoch)
    writer.add_scalar('accu/train', 100*accuracy / total, epoch)

    with torch.no_grad():
        accuracy = 0
        total = 0
        for data in test_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum()

    acc = 100. * accuracy / total
    print(f'Test Accu: {acc:.3f}%')

    print('saving model ...')
    torch.save(net.state_dict(), f'./model/net_{epoch+1}.pth')


writer.close()

