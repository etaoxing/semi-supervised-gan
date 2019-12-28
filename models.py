import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as weight_norm_

class GaussianNoise(nn.Module):
    """Gaussian noise regulation.
    https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4
    
    # only active during training
    # torch.distributions.normal.Normal(loc=0, scale=0.3).sample(sample_shape=torch.Size([1, 1]))
    """

    def __init__(self, mean=0, std=1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.noise = torch.tensor(0)

    def forward(self, x):
        if self.training and self.std != 0:
            sampled_noise = self.noise.repeat(*x.size()).float().normal_(mean=self.mean, std=self.std)
            x = x + sampled_noise.to(x.device)
        return x 

class Net(nn.Module):
    def __init__(self, input_shape, output_shape, leaky=False, weight_norm=False):
        super().__init__()
        self.gn1 = GaussianNoise(std=0.3)
        self.gn2 = GaussianNoise(std=0.5)
        if leaky:
            self.act = nn.LeakyReLU(negative_slope=leaky)
        else:
            self.act = nn.ReLU()

        self.fc1 = nn.Linear(input_shape, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 250)
        self.fc5 = nn.Linear(250, 250)
        self.out = nn.Linear(250, output_shape)
        
        self.weight_norm = weight_norm
        if weight_norm:
            self.fc1 = weight_norm_(self.fc1)
            self.fc2 = weight_norm_(self.fc2)
            self.fc3 = weight_norm_(self.fc3)
            self.fc4 = weight_norm_(self.fc4)
            self.fc5 = weight_norm_(self.fc5)
            self.out = weight_norm_(self.out)

    def forward(self, x, mid=False):
        x = self.gn1(x)
        x = self.act(self.fc1(x))
        x = self.gn2(x)
        x = self.act(self.fc2(x))
        x = self.gn2(x)
        x = self.act(self.fc3(x))
        x = self.gn2(x)
        x = self.act(self.fc4(x))
        x = self.gn2(x)
        x = self.act(self.fc5(x))
        if mid:
            return x
        
        x = self.out(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_shape, output_shape, 
                 final_act='tanh',
                 weight_norm=False,
                 bn_params=dict(eps=1e-6, momentum=0.9, affine=True),
                ):
        super().__init__()
        self.sp = nn.Softplus()

        if final_act == 'tanh':
            self.final_act = nn.Tanh()
        elif final_act == 'softplus':
            self.final_act = nn.Softplus()
        elif final_act is None:
            self.final_act = None
        else:
            raise ValueError

        self.fc1 = nn.Linear(input_shape, 500)
        self.bn1 = nn.BatchNorm1d(500, **bn_params)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500, **bn_params)
        self.out = nn.Linear(500, output_shape)

        self.weight_norm = weight_norm
        if weight_norm:
            self.out = weight_norm_(self.out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.sp(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sp(x)
        x = self.out(x)

        if self.final_act is not None:
            x = self.final_act(x)
        return x

class Discriminator(Net):
    def __init__(self, input_shape, output_shape, feature_matching=False, **kwargs):
        super().__init__(input_shape, output_shape, **kwargs)
        self.feature_matching = feature_matching
        if feature_matching:
            self.mid = self._forward_mid
            
    def _forward_mid(self, x):
        return super().forward(x, mid=True)
