import numpy as np
import torch
import torchvision
from torch.utils import data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils import equalize_union, equalize

import os, pickle, collections
import pandas as pd
import librosa

class SemiSupervisedMixin:
    """
    Properties for dataset split into two subsets of labeled and unlabeled data
    Requires perc_labeled, eq_union, and num_samples to be set
    """

    def make_split(self, stratify=None):
        assert self.perc_labeled <= 1 and self.perc_labeled >= 0

        self.labeled_i = np.array(range(self.num_samples))
        if self.perc_labeled < 1:
            self.labeled_i, self.unlabeled_i = train_test_split(self.labeled_i, test_size=self.num_unlabeled, stratify=stratify)

            if self.eq_union:
                self.labeled_i, self.unlabeled_i = equalize_union(self.labeled_i, self.unlabeled_i)
            else:
                self.labeled_i, self.unlabeled_i = equalize(self.labeled_i, self.unlabeled_i)

    @property
    def has_unlabeled(self):
        return self.perc_labeled < 1

    @property
    def perc_unlabeled(self):
        return 1 - self.perc_labeled

    @property
    def num_labeled(self):
        return int(self.num_samples * self.perc_labeled)

    @property
    def num_unlabeled(self):
        return self.num_samples - self.num_labeled

class MaterialDataset(data.Dataset):
    def __init__(self, modalities, label_encoding, df=None):
        self.df = df
        self.modalities = modalities
        self.label_encoding = label_encoding
    
    def size(self):
        X_size = sum([len(self.df.iloc[0][modality]) for modality in self.modalities])
        y_size = len(self.label_encoding)
        return (X_size, y_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        X = torch.cat([torch.tensor(row[m], dtype=torch.float32) for m in self.modalities])
        y = row['material']
        y_encoded = torch.tensor(self.label_encoding[y])
        
        return X, y_encoded

class SemiSupervisedMaterialDataset(MaterialDataset, SemiSupervisedMixin):
    def __init__(self, *args, df=None, noise_size=100, noise_dist=None, perc_labeled=1.0, eq_union=True):
        super().__init__(*args, df=df)
        self.noise_size = noise_size
        self.noise_dist = noise_dist

        self.perc_labeled = perc_labeled
        self.num_samples = len(df)
        self.eq_union = eq_union
        self.make_split(stratify=self.df['material'])
   
    def __len__(self):
        return len(self.labeled_i)

    def size(self):
        n = 5 if self.has_unlabeled else 4
        return tuple(list(super().size()) + [n]) 
            
    def __getitem__(self, index):
        i_l = self.labeled_i[index]
        row_l = self.df.iloc[i_l]
        X_l = torch.cat([torch.tensor(row_l[m], dtype=torch.float32) for m in self.modalities])
        y_l = row_l['material']
        y_l_encoded = torch.tensor(self.label_encoding[y_l])

        X_ul = -1
        if self.has_unlabeled:
            i_ul = self.unlabeled_i[index]
            row_ul = self.df.iloc[i_l]
            X_ul = torch.cat([torch.tensor(row_ul[m], dtype=torch.float32) for m in self.modalities])

        noise1 = self.noise_dist.sample(sample_shape=(self.noise_size,))
        noise2 = self.noise_dist.sample(sample_shape=(self.noise_size,))
        return {'X_labeled': X_l, 'y_labeled': y_l_encoded, 'X_unlabeled': X_ul, 'noise1': noise1, 'noise2': noise2}

class SemiSupervisedMNIST(torchvision.datasets.MNIST, SemiSupervisedMixin):
    def __init__(self, noise_size=100, noise_dist=None, label_encoding=None, perc_labeled=1.0, eq_union=True, **kwargs):
        super().__init__(**kwargs)

        self.noise_size = noise_size
        self.noise_dist = noise_dist
        self.label_encoding = label_encoding

        self.perc_labeled = perc_labeled
        self.eq_union = eq_union
        self.num_samples = super().__len__()
        self.make_split(stratify=self.targets.numpy())

    def __len__(self):
        return len(self.labeled_i)

    def size(self):
        return (28*28, 10)

    def __getitem__(self, index):
        i_l = self.labeled_i[index]
        X_l, y_l_encoded = super().__getitem__(i_l)
        X_ul = -1
        if self.has_unlabeled:
            i_ul = self.unlabeled_i[index]
            X_ul, _ = super().__getitem__(i_ul)

        noise1 = self.noise_dist.sample(sample_shape=(self.noise_size,))
        noise2 = self.noise_dist.sample(sample_shape=(self.noise_size,))
        return {'X_labeled': X_l, 'y_labeled': y_l_encoded, 'X_unlabeled': X_ul, 'noise1': noise1, 'noise2': noise2}

def make_mnist_ssdatasets(perc_labeled, eq_union, noise_size, noise_dist, data_dir='mnist_data/'):
    label_encoding = {n: n for n in range(9)}
    label_encoding['fake'] = len(label_encoding)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: x.flatten().float())
    ])
    train_dataset = SemiSupervisedMNIST(perc_labeled=perc_labeled,
                                        eq_union=eq_union,
                                        noise_size=noise_size,
                                        noise_dist=noise_dist,
                                        label_encoding=label_encoding,
                                        root=data_dir, train=True, transform=transforms, download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transforms, download=True)
    test_dataset.label_encoding = label_encoding
    return train_dataset, test_dataset, label_encoding

def load_mreo_data(data_dir, modalities):
    data_files = os.listdir(data_dir)

    raw = {}
    materials = []
    for f in data_files:
        path = os.path.join(data_dir, f)
        material = f.split('_')[2]
        materials.append(material)
        with open(path, 'rb') as pkl_file:
            c = pickle.load(pkl_file, encoding='latin1')
            raw[material] = c

    d_material = []
    d_obj = []
    d_obj_sample_num = []
    d_obj_raw = []
    for material in materials:
        c = raw[material]
        for obj in c:
            for obj_sample_num in range(len(c[obj]['temperature'])):
                d_material.append(material)
                d_obj.append(obj)
                d_obj_sample_num.append(obj_sample_num)

                r = {m: c[obj][m][obj_sample_num] for m in modalities}
                d_obj_raw.append(r)

    return d_material, d_obj, d_obj_sample_num, d_obj_raw 

def make_mreo_ssdatasets(perc_labeled,
                         eq_union,
                         noise_size,
                         noise_dist,
                         data_dir='mreo_data/',
                         test_size=1200, # 0.167
                         mel=True,
                         modalities=['temperature', 'force0', 'force1', 'contact'],
                         cached=True,
                        ):
    cached_path = os.path.join(data_dir, 'cached_mreo_mel{}_{}.pkl'.format(int(mel), ','.join(modalities)))
    if os.path.exists(cached_path) and cached:
        df = pickle.load(open(cached_path, 'rb'))
    else:
        d_material, d_obj, d_obj_sample_num, d_obj_raw = load_mreo_data(data_dir, modalities)
        d_data = collections.defaultdict(list)
        for material, obj, obj_sample_num, obj_raw in zip(d_material, d_obj, d_obj_sample_num, d_obj_raw):
            for modality in modalities:
                modality_data = obj_raw[modality]
                if modality is 'contact' and mel:
                    S = librosa.feature.melspectrogram(np.array(modality_data), sr=48000, n_mels=128)
                    # Convert to log scale (dB)
                    log_S = librosa.amplitude_to_db(S, ref=np.max)
                    d_data[modality].append(log_S.flatten())
                else:
                    d_data[modality].append(modality_data)

        d = dict(material=d_material, obj=d_obj, obj_sample_num=d_obj_sample_num)
        for modality in modalities:
            d[modality] = d_data[modality]

        df = pd.DataFrame(data=d)
        if cached: pickle.dump(df, open(cached_path, 'wb'))

    data_i = list(range(len(df)))
    train_i, test_i = train_test_split(data_i, test_size=test_size, stratify=df['material'].iloc[data_i])

    # Scale data to zero mean and unit variance
    for modality in modalities:
        df_m = df[modality].copy()
        scaler = preprocessing.StandardScaler()
        train_norm = scaler.fit_transform(np.stack(df_m.iloc[train_i].values))
        test_norm = scaler.transform(np.stack(df_m.iloc[test_i].values))

        df_m.iloc[train_i] = train_norm.tolist()
        df_m.iloc[test_i] = test_norm.tolist()
        df[modality] = df_m

    # label_encoding = {m: i for i, m in enumerate(list(df['material'].unique()))}
    # datasets_i = {'train': train_i, 'test': test_i}
    # datasets = {l: MaterialDataset(modalities, label_encoding, df=df.iloc[i].reset_index().rename(columns={'index': 'sample_id'})) for l, i in datasets_i.items()}
    # return datasets['train'], datasets['test'], label_encoding

    label_encoding = {m: i for i, m in enumerate(list(df['material'].unique()))}
    label_encoding['fake'] = len(label_encoding)
    train_dataset = SemiSupervisedMaterialDataset(modalities, 
                                                  label_encoding,
                                                  df=df.iloc[train_i].reset_index().rename(columns={'index': 'sample_id'}), 
                                                  perc_labeled=perc_labeled,
                                                  eq_union=eq_union,
                                                  noise_size=noise_size,
                                                  noise_dist=noise_dist)
    test_dataset = MaterialDataset(modalities, label_encoding, df.iloc[test_i].reset_index().rename(columns={'index': 'sample_id'}))
    return train_dataset, test_dataset, label_encoding