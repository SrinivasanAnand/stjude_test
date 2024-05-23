import os
import torch, torchvision
import torch.nn as nn
import vgg16

# TODO: Choose hyper-parameters

# Model - either neural network or logistic regression
MODEL_NAME = 'vgg16'

# Batch size - number of images within a training batch of one training iteration
N_BATCH = 16

# Training epoch - number of passes through the full training dataset
N_EPOCH = 20

# Learning rate - step size to update parameters
LEARNING_RATE = 0.005

# Learning rate decay - scaling factor to decrease learning rate at the end of each decay period
LEARNING_RATE_DECAY = 0.50

# Learning rate decay period - number of epochs before reducing/decaying learning rate
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

    # TODO: Move model to device
    model = model.to(device)

    # TODO: Define cross entropy loss
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # TODO: Decrease learning rate when learning rate decay period is met
        # Directly modify param_groups in optimizer to set new learning rate
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch
        if epoch and epoch % learning_rate_decay_period == 0:
            pass

        for batch, (images, labels) in enumerate(dataloader):

            # TODO: Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # TODO: Vectorize images from (N, H, W, C) to (N, d)
            #n_dim = np.prod(images.shape[1:])
            #images = images.view(-1, n_dim)

            # TODO: Forward through the model
            outputs = model(images)

            # TODO: Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()

            # TODO: Compute loss function
            loss = loss_func(outputs, labels)

            # TODO: Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # TODO: Accumulate total loss for the epoch
            total_loss = total_loss + loss.item()

        mean_loss = total_loss / len(dataloader)

        # Log average loss over the epoch
        print('Epoch={}/{}  Loss: {:.3f}'.format(epoch + 1, n_epoch, mean_loss))

    return model

# Create transformations convert data to torch tensor
# https://pytorch.org/docs/stable/torchvision/transforms.html
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Set path to save checkpoint
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

# TODO: Setup a dataloader (iterator) to fetch from the training set using
# torch.utils.data.DataLoader and set shuffle=True, drop_last=True, num_workers=2
# Set your batch size to the hyperparameter N_BATCH
dataloader_train = torch.utils.data.DataLoader(
    cifar10_train,
    batch_size=N_BATCH,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# Define the possible classes in CIFAR10
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

# CIFAR10 has 10 classes
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
# TODO: Set model to training mode
model.train()

# TODO: Train model with device='cuda'
model = train(
    model,
    dataloader_train,
    N_EPOCH,
    optimizer,
    learning_rate_decay=LEARNING_RATE_DECAY,
    learning_rate_decay_period=LEARNING_RATE_DECAY_PERIOD,
    device='cuda')

# TODO: Save weights into checkpoint path
torch.save({'state_dict' : model.state_dict()}, checkpoint_path)
