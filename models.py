import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    """Gaussian noise regulation.
    https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4
    
    # torch.distributions.normal.Normal(loc=0, scale=0.3).sample(sample_shape=torch.Size([1, 1]))
    """

    def __init__(self, mean=0, std=1, is_relative_detach=True, device=None):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std
        self.noise = torch.tensor(0)
        if device:
            self.noise = self.noise.to(device)

    def forward(self, x):        
        if self.training and self.std != 0:
            sampled_noise = self.noise.repeat(*x.size()).float().normal_(mean=self.mean, std=self.std)
            x = x + sampled_noise.type(x.dtype)
        return x 

class Model_NN(nn.Module):
    def __init__(self, input_shape, output_shape, leaky=False):
        super(Model_NN, self).__init__()
        self.gn1 = GaussianNoise(std=0.3)
        self.gn2 = GaussianNoise(std=0.5)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=leaky)
        else:
            self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(input_shape, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 250)
        self.fc5 = nn.Linear(250, 250)
        self.out = nn.Linear(250, output_shape)
        
    def forward(self, x, mid=False):
        x = self.gn1(x)
        x = self.relu(self.fc1(x))
        x = self.gn2(x)
        x = self.relu(self.fc2(x))
        x = self.gn2(x)
        x = self.relu(self.fc3(x))
        x = self.gn2(x)
        x = self.relu(self.fc4(x))
        x = self.gn2(x)
        x = self.relu(self.fc5(x))
        if mid:
            return x
        
        y = self.out(x)
        return y

class Generator(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Generator, self).__init__()
        self.sp = nn.Softplus()
        self.tanh = nn.Tanh()
        
        self.fc1 = nn.Linear(input_shape, 500)
        self.bn = nn.BatchNorm1d(500, eps=2e-5, momentum=0.9)
        self.fc2 = nn.Linear(500, 500)
        self.out = nn.Linear(500, output_shape)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sp(x)
        x = self.bn(x)
        x = self.fc2(x)
        x = self.sp(x)
        y = self.out(x)
        y = self.tanh(y)
        return y

class Discriminator(Model_NN):
    def __init__(self, input_shape, output_shape, feature_matching=False, leaky=False):
        super(Discriminator, self).__init__(input_shape, output_shape, leaky=leaky)
        self.feature_matching = feature_matching
        if feature_matching:
            self.mid = self._forward_mid
            
    def _forward_mid(self, x):
        return super(Discriminator, self).forward(x, mid=True)
