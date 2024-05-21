import torch.nn as nn

class VGG16(nn.Module):
    '''
    Neural network class of fully connected layers

    Arg(s):
        n_input_feature : int
            number of input features
        n_output : int
            number of output classes
    '''

    def __init__(self, n_input_feature, n_output):
        super(VGG16, self).__init__()

        # Create your 6-layer neural network using fully connected layers with ReLU activations
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
        # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

        # n_input_feature = image H x W
        max_pool_layers = 5
        n_input_feature_reduced = 32 // (2 ** max_pool_layers)

        #ReLU
        self.relu = nn.ReLU()

        #Layer 1
        self.conv_layer_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        #Layer 2
        self.conv_layer_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Layer 3
        self.conv_layer_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        #Layer 4
        self.conv_layer_4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Layer 5
        self.conv_layer_5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        #Layer 6
        self.conv_layer_6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        #Layer 7
        self.conv_layer_7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.max_pool_7 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Layer 8
        self.conv_layer_8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        #Layer 9
        self.conv_layer_9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        #Layer 10
        self.conv_layer_10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.max_pool_10 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Layer 11
        self.conv_layer_11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        #Layer 12
        self.conv_layer_12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512)

        #Layer 13
        self.conv_layer_13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.max_pool_13 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Layer FC
        self.dropout_1 = nn.Dropout(0.5)
        self.linear_1 = nn.Linear(n_input_feature_reduced*512, 4096)

        #Layer FC 1
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(4096, 4096)

        #Layer FC 2
        self.linear_3 = nn.Linear(4096, n_output)

    def forward(self, x):
        '''
        Forward pass through the neural network

        Arg(s):
            x : torch.Tensor[float32]
                tensor of N x d
        Returns:
            torch.Tensor[float32]
                tensor of n_output predicted class
        '''

        # TODO: Implement forward function

        #Layer 1
        x = self.conv_layer_1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #Layer 2
        x = self.conv_layer_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool_2(x)

        #Layer 3
        x = self.conv_layer_3(x)
        x = self.bn3(x)
        x = self.relu(x)

        #Layer 4
        x = self.conv_layer_4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.max_pool_4(x)

        #Layer 5
        x = self.conv_layer_5(x)
        x = self.bn5(x)
        x = self.relu(x)

        #Layer 6
        x = self.conv_layer_6(x)
        x = self.bn6(x)
        x = self.relu(x)

        #Layer 7
        x = self.conv_layer_7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.max_pool_7(x)

        #Layer 8
        x = self.conv_layer_8(x)
        x = self.bn8(x)
        x = self.relu(x)

        #Layer 9
        x = self.conv_layer_9(x)
        x = self.bn9(x)
        x = self.relu(x)

        #Layer 10
        x = self.conv_layer_10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.max_pool_10(x)

        #Layer 11
        x = self.conv_layer_11(x)
        x = self.bn11(x)
        x = self.relu(x)

        #Layer 12
        x = self.conv_layer_12(x)
        x = self.bn12(x)
        x = self.relu(x)

        #Layer 13
        x = self.conv_layer_13(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = self.max_pool_13(x)

        #FC Layers
        x = x.view(x.size(0), -1)

        #FC Layer 1
        x = self.dropout_1(x)
        x = self.linear_1(x)
        x = self.relu(x)

        #FC Layer 2
        x = self.dropout_2(x)
        x = self.linear_2(x)
        x = self.relu(x)

        #FC Layer 3
        output_logits = self.linear_3(x)

        return output_logits
