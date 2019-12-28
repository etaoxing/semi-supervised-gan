import torch
import torchvision
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import argparse

from datasets import make_mnist_ssdatasets, make_mreo_ssdatasets
from models import Discriminator, Generator
from losses import DiscriminatorLoss, GeneratorLoss
from metrics import AverageAccuracy, FakeAccuracy, Loss, ClassAccuracy, RunTime
from history import History
from trainers import GAN_Trainer
from utils import set_seed

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def make_trainer(args, to_viz=False, verbose=1):
    if args.noise_dist == 'normal':
        noise_dist = torch.distributions.Normal(0, 1)
    elif args.noise_dist == 'uniform':
        noise_dist = torch.distributions.Uniform(0, 1)
    else:
        raise ValueError

    if args.dataset == 'mnist':
        train_dataset, test_dataset, label_encoding = make_mnist_ssdatasets(args.perc_labeled, args.eq_union, args.noise_size, noise_dist)
    elif args.dataset == 'mreo':
        train_dataset, test_dataset, label_encoding = make_mreo_ssdatasets(args.perc_labeled, args.eq_union, args.noise_size, noise_dist)
    else:
        raise ValueError

    datasets = {'train': train_dataset,
                'test': test_dataset}
    phases = ['train', 'test']
    print('dataset sizes:', ', '.join(k + ' ' + str(len(v)) for k, v in datasets.items()))
    print('perc labeled: {}, num labeled: {}, num unlabeled: {}, labeled_i: {}'.format(\
        train_dataset.perc_labeled, train_dataset.num_labeled, train_dataset.num_unlabeled,\
        len(train_dataset.labeled_i)))
    print(label_encoding)

    dataloader_params = {'train': {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': use_cuda},
                         'test':  {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'pin_memory': use_cuda}
                        }
    dataloaders = {l: DataLoader(d, **dataloader_params[l]) for l, d in datasets.items()}

    data_shape = datasets['train'].size()
    in_shape, out_shape = data_shape[0], data_shape[1]

    if args.dataset == 'mnist':
        gen_bn_params = dict(eps=1e-6, momentum=0.5, affine=False)
    elif args.dataset == 'mreo':
        gen_bn_params = dict(eps=2e-5, momentum=0.9, affine=True)
    else:
        raise ValueError

    nets = {
        'D': Discriminator(in_shape, out_shape, feature_matching=args.feature_matching, weight_norm=args.weight_norm),
        'G': Generator(args.noise_size, in_shape, weight_norm=args.weight_norm, final_act=args.gen_final_act)
    }
    optimizers = {
        'D': torch.optim.Adam(nets['D'].parameters(), lr=args.lr, betas=(0.5, 0.999)),
        'G': torch.optim.Adam(nets['G'].parameters(), lr=args.lr, betas=(0.5, 0.999)),
    }
    criterions = {
        'D': DiscriminatorLoss(return_all=True),
        'G': GeneratorLoss()
    }

    metrics = {'D': [AverageAccuracy(),
                     FakeAccuracy(output_transform=lambda x: (x[0], x[1].long())),
                     Loss(criterions['D'], name='loss_D'),
                     Loss(criterions['D'], name='loss_labeled'),
                     Loss(criterions['D'], name='loss_unlabeled'),
                     ClassAccuracy(len(label_encoding)),
                     RunTime()
                    ],
               'G': [FakeAccuracy(output_transform=lambda x: (x[0], x[1].long())),
                     Loss(criterions['G'], name='loss_G'),
                     RunTime()
                    ],
    }

    viz_params = {
        'D': {
            'to_viz': to_viz,
            'bands': False
        },
        'G': {
            'to_viz': False,
            'bands': False
        }
    }
    history = {
        'D': History(metrics=metrics['D'], viz_params=viz_params['D'], phases=list(datasets.keys()), verbose=verbose),
        'G': History(metrics=metrics['G'], viz_params=viz_params['G'], phases=list(datasets.keys()), verbose=0),
    }

    t = GAN_Trainer(model=nets,
                    dataloaders=dataloaders, 
                    optimizer=optimizers,
                    criterion=criterions, 
                    history=history,
                    device=device)
    return t

    # net = Net(datasets['train'].shape[0], datasets['train'].shape[1])
    # optimizer = optim.Adam(net.parameters(), lr=0.0006, betas=(0.5, 0.999))
    # metrics = [AverageAccuracy(), 
    #         ClassAccuracy(len(label_encoding)), 
    #         Loss(nn.CrossEntropyLoss(), name='CrossEntropy'), 
    #         Loss(nn.MSELoss(), name='MSE', output_transform=lambda y_pred, y: (y_pred, to_onehot(y, len(label_encoding)).float())),
    #         RunTime() 
    #         ]
    # viz_params = {
    #     'to_viz': True,
    #     'bands': False
    # }
    # history = History(metrics=metrics, viz_params=viz_params, phases=list(datasets.keys()))
    # t = Trainer(model=net,
    #             dataloaders=dataloaders, 
    #             optimizer=optimizer, 
    #             criterion=nn.CrossEntropyLoss(), 
    #             history=history)
    # return t

def ssgan_exp(args):
    print(args)
    set_seed(args.seed, use_cuda=use_cuda)
    t = make_trainer(args)

    s = 'ckpt_ssgan_{}_perclabeled{}_noisesize{}_noise{}_lr{}_featmatch{}_weightnorm{}_gfa{}_equnion{}_seed{}'.format(args.dataset, \
        args.perc_labeled, args.noise_size, args.noise_dist, args.lr,
        int(args.feature_matching), int(args.weight_norm), args.gen_final_act,
        int(args.eq_union), args.seed)
    s = s.replace('.', ',')
    print('exp', s)

    if args.load_ckpt: t.load(args.load_ckpt)
    t.run(max_epoch=args.epochs)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
        print('discriminator history:')
        print(t.history['D'].to_df())
        print('-' * 50)
        print('generator history:')
        print(t.history['G'].to_df())

    t.save(name=s)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lc', '--load_ckpt', default=None, type=str)
    parser.add_argument('--dataset', required=True, choices=['mnist', 'mreo'])
    parser.add_argument('--lr', default=0.006, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--perc_labeled', default=1.0, type=float)
    parser.add_argument('--noise_size', default=100, type=int)
    parser.add_argument('--noise_dist', default='uniform', choices=['normal', 'uniform'], help='noise input for generator')
    parser.add_argument('--gen_final_act', default=None, choices=['tanh', 'softplus'])
    parser.add_argument('--seed', default=1000, type=int)

    parser.add_argument('--no_feature_matching', dest='feature_matching', action='store_false')
    parser.set_defaults(feature_matching=True)

    parser.add_argument('--no_eq_union', dest='eq_union', action='store_false')
    parser.set_defaults(eq_union=True)


    parser.add_argument('--use_weight_norm', dest='weight_norm', action='store_true')
    parser.set_defaults(weight_norm=False)

    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    ssgan_exp(args)