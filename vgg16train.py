import os
import torch, torchvision
import torch.nn as nn
import vgg16

# Model
MODEL_NAME = 'vgg16'

# Batch size
N_BATCH = 16

# Training epochs
N_EPOCH = 20

# Learning rate
LEARNING_RATE = 0.005

# Learning rate decay
LEARNING_RATE_DECAY = 0.50

# Learning rate decay period
LEARNING_RATE_DECAY_PERIOD = 20

def train(model,
          dataloader,
          n_epoch,
          optimizer,
          learning_rate_decay,
          learning_rate_decay_period,
          device):
    '''
    Trains the model using optimizer and specified learning rate schedule

    Arg(s):
        model : torch.nn.Module
            neural network or logistic regression
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        n_epoch : int
            number of epochs to train
        optimizer : torch.optim
            https://pytorch.org/docs/stable/optim.html
            optimizer to use for updating weights
        learning_rate_decay : float
            rate of learning rate decay
        learning_rate_decay_period : int
            period to reduce learning rate based on decay e.g. every 2 epoch
        device : str
            device to run on
    Returns:
        torch.nn.Module : trained network
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    # Move model to device
    model = model.to(device)

    # Cross entropy loss
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # Decrease learning rate as necessary
        if epoch and epoch % learning_rate_decay_period == 0:
            pass

        for batch, (images, labels) in enumerate(dataloader):

            # Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            optimizer.zero_grad()

            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()

        mean_loss = total_loss / len(dataloader)

        # Log average loss over the epoch
        print('Epoch={}/{}  Loss: {:.3f}'.format(epoch + 1, n_epoch, mean_loss))

    return model

# Convert data to torch tensor
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

checkpoint_path = './checkpoint-{}.pth'.format(MODEL_NAME)

'''
Set up dataloading
'''
# Download and setup CIFAR10 training set using preconfigured torchvision.datasets.CIFAR10
cifar10_train = torchvision.datasets.CIFAR10(
    root=os.path.join('data', 'CIFAR10'),
    train=True,
    download=True,
    transform=transforms)

dataloader_train = torch.utils.data.DataLoader(
    cifar10_train,
    batch_size=N_BATCH,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# Possible classes in CIFAR10
class_names = [
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

n_class = len(class_names)

'''
Set up model and optimizer
'''
n_input_feature = 3 * 32 * 32

if MODEL_NAME == 'vgg16':
    model = vgg16.VGG16(n_input_feature, n_class)
else:
    raise('Unsupported model name: {}'.format(MODEL_NAME))

# https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

'''
Train model and store weights
'''
model.train()

model = train(
    model,
    dataloader_train,
    N_EPOCH,
    optimizer,
    learning_rate_decay=LEARNING_RATE_DECAY,
    learning_rate_decay_period=LEARNING_RATE_DECAY_PERIOD,
    device='cuda')

torch.save({'state_dict' : model.state_dict()}, checkpoint_path)
