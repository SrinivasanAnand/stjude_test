import os
import numpy as np
import torch, torchvision
import torch.nn as nn
import vgg16

# Model
MODEL_NAME = 'vgg16'

# Batch size
N_BATCH = 25

# Training epochs
N_EPOCH = 20

# Learning rate
LEARNING_RATE = 0.005

# Learning rate decay
LEARNING_RATE_DECAY = 0.50

# Learning rate decay period
LEARNING_RATE_DECAY_PERIOD = 20

def evaluate(model, dataloader, class_names, device):
    '''
    Evaluates the network on a dataset

    Arg(s):
        model : torch.nn.Module
            neural network or logistic regression
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        class_names : list[str]
            list of class names to be used in plot
        device : str
            device to run on
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    # Move model to device
    model = model.to(device)

    n_correct = 0
    n_sample = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            outputs = torch.argmax(outputs, dim=1)

            n_sample = n_sample + N_BATCH

            for i in range(25):
                if outputs[i] == labels[i]:
                    n_correct = n_correct + 1

    mean_accuracy = n_correct / n_sample * 100

    print('Mean accuracy over {} images: {:.3f}%'.format(n_sample, mean_accuracy))

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Set path to save checkpoint
checkpoint_path = './checkpoint-{}.pth'.format(MODEL_NAME)

def evaluate(model, dataloader, class_names, device):
    '''
    Evaluates the network on a dataset

    Arg(s):
        model : torch.nn.Module
            neural network or logistic regression
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        class_names : list[str]
            list of class names to be used in plot
        device : str
            device to run on
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    # TODO: Move model to device
    model = model.to(device)

    n_correct = 0
    n_sample = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            # Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            outputs = torch.argmax(outputs, dim=1)

            n_sample = n_sample + N_BATCH

            for i in range(25):
                if outputs[i] == labels[i]:
                    n_correct = n_correct + 1

    mean_accuracy = n_correct / n_sample * 100

    print('Mean accuracy over {} images: {:.3f}%'.format(n_sample, mean_accuracy))


'''
Set up dataloading
'''
cifar10_test = torchvision.datasets.CIFAR10(
    root=os.path.join('data', 'CIFAR10'),
    train=False,
    download=True,
    transform=transforms)

dataloader_test = torch.utils.data.DataLoader(
    cifar10_test,
    batch_size=N_BATCH,
    shuffle=False,
    drop_last=False,
    num_workers=2
)

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
Set up model
'''
n_input_feature = 3 * 32 * 32

if MODEL_NAME == 'vgg16':
    model = vgg16.VGG16(n_input_feature, n_class)
else:
    raise('Unsupported model name: {}'.format(MODEL_NAME))

'''
Restore weights and evaluate model
'''
# Load model from checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

evaluate(model, dataloader_test, class_names, 'cuda')