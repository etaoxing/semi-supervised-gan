import torch
import torchvision
from torch.utils import data
import torch.optim as optim

import pandas as pd
import numpy as np

from datasets import SemiSupervisedMNIST
from models import Discriminator, Generator
from losses import DiscriminatorLoss, GeneratorLoss
from metrics import AverageAccuracy, FakeAccuracy, Loss, ClassAccuracy, RunTime
from history import History
from trainers import GAN_Trainer

def run_gan():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    np.random.seed(876)
    pd.np.random.seed(876)
    torch.manual_seed(876)
    if use_cuda:
        torch.cuda.manual_seed_all(876)

    noise_size = 100
    distribution = torch.distributions.normal.Normal(0, 1)     
    label_encoding = {n: n for n in range(9)}
    label_encoding['fake'] = len(label_encoding)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: x.flatten().float())
    ])         
    train_dataset = SemiSupervisedMNIST(num_labeled=30000,
                                        noise_size=noise_size,
                                        distribution=distribution,
                                        label_encoding=label_encoding,
                                        root='./mnist_data', train=True, transform=transforms, download=False)
    test_dataset = torchvision.datasets.MNIST(root='./mnist_data', train=False, transform=transforms, download=False)
    test_dataset.label_encoding = label_encoding       
    datasets = {'train': train_dataset,
                'test': test_dataset}
    phases = list(datasets.keys())

    dataloader_params = {'train': {'batch_size': 64, 'shuffle': True, 'num_workers': 8, 'pin_memory': use_cuda},
                        'test':  {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': use_cuda}
                        }
    dataloaders = {l: data.DataLoader(d, **dataloader_params[l]) for l, d in datasets.items()}

    nets = {
        'D': Discriminator(datasets['train'].shape[0], datasets['train'].shape[1], feature_matching=True, leaky=0.2),
        'G': Generator(noise_size, datasets['train'].shape[0])
    }

    optimizers = {
        'D': optim.Adam(nets['D'].parameters(), lr=0.0006, betas=(0.5, 0.999)),
        'G': optim.Adam(nets['G'].parameters(), lr=0.0006, betas=(0.5, 0.999)),
    }
        
    criterions = {
        'D': DiscriminatorLoss(return_all=True),
        'G': GeneratorLoss()
    }

    metrics = {
        'D': [AverageAccuracy(),
              FakeAccuracy(output_transform=lambda x: (x[0], x[1].long())),
              Loss(criterions['D'], name='loss_D'),
              Loss(criterions['D'], name='loss_labeled'),
              Loss(criterions['D'], name='loss_unlabeled'),
              ClassAccuracy(len(label_encoding)),
              RunTime()],
        'G': [FakeAccuracy(output_transform=lambda x: (x[0], x[1].long())),
              Loss(criterions['G'], name='loss_G'),
              RunTime()],
    }

    viz_params = {
        'D': {
            'to_viz': False,
            'bands': False
        },
        'G': {
            'to_viz': False,
            'bands': False
        }
    }

    history = {
        'D': History(metrics=metrics['D'], viz_params=viz_params['D'], phases=list(datasets.keys()), verbose=1),
        'G': History(metrics=metrics['G'], viz_params=viz_params['G'], phases=list(datasets.keys()), verbose=1),
    }

    t = GAN_Trainer(model=nets,
                    dataloaders=dataloaders, 
                    optimizer=optimizers,
                    criterion=criterions, 
                    history=history,
                    device=device)

    # t.load('checkpoints/checkpoint_50.pt')

    t.run(max_epoch=0)

    print('discriminator history:')
    print(history['D'].to_df())
    print('-' * 100)
    print('generator history:')
    print(history['G'].to_df())

    t.save()

if __name__ == '__main__':
    run_gan()