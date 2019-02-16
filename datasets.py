import numpy as np
import torch
import torchvision
from torch.utils import data
from sklearn.model_selection import train_test_split
from utils import equalize

class MaterialDataset(data.Dataset):
    def __init__(self, modalities, label_encoding, df=None):
        self.df = df
        self.modalities = modalities
        self.label_encoding = label_encoding
        self.shape = self.size()
    
    def size(self):
        X_size = sum([len(df.iloc[0][modality]) for modality in self.modalities])
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

class SemiSupervisedMaterialDataset(MaterialDataset):
    """Holds an equal number of collated data comprised of labeled, unlabeled, and noise
    """
    def __init__(self, *args, df=None, percent_unlabeled=None, noise_size=100, distribution=None):
        assert (percent_unlabeled < 1), 'Must have some labeled data, need percent_unlabeled < 1'
        assert distribution, 'Must provide a distribution to sample noise from'
        self.percent_unlabeled = percent_unlabeled
        self.noise_size = noise_size
        self.labeled_i = np.array(range(len(df)))
        self.distribution = distribution
            
        super(SemiSupervisedMaterialDataset, self).__init__(*args, df=df)
        if percent_unlabeled:
            self.split_data()
   
    def split_data(self):        
        self.labeled_i, self.unlabeled_i = equalize(*train_test_split(self.labeled_i, test_size=self.percent_unlabeled, stratify=self.df['material'].iloc[self.labeled_i]))
                
    def size(self):
        n = 5 if self.percent_unlabeled > 0 else 4
        return tuple(list((super(SemiSupervisedMaterialDataset, self).size())) + [n]) 
            
    def __len__(self):
        return len(self.labeled_i)
            
    def __getitem__(self, index):
        i_l = self.labeled_i[index]        
        row_l = self.df.iloc[i_l]
        X_l = torch.cat([torch.tensor(row_l[m], dtype=torch.float32) for m in self.modalities])
        y_l = row_l['material']
        y_l_encoded = torch.tensor(self.label_encoding[y_l])
        
        X_ul = -1
        if self.percent_unlabeled:
            i_ul = self.unlabeled_i[index]
            row_ul = self.df.iloc[i_l]
            X_ul = torch.cat([torch.tensor(row_ul[m], dtype=torch.float32) for m in self.modalities])
            
        noise1 = self.distribution.sample(sample_shape=(self.noise_size,))
        noise2 = self.distribution.sample(sample_shape=(self.noise_size,))
        return {'X_labeled': X_l, 'y_labeled': y_l_encoded, 'X_unlabeled': X_ul, 'noise1': noise1, 'noise2': noise2}

class SemiSupervisedMNIST(torchvision.datasets.MNIST):
    def __init__(self, num_labeled=0, noise_size=100, distribution=None, label_encoding=None, **kwargs):
        super(SemiSupervisedMNIST, self).__init__(**kwargs)
                
        if self.train:
            self.labels = self.train_labels
        else:
            self.labels = self.test_labels
            
        self.noise_size = noise_size
        self.distribution = distribution
        self.shape = self.size()
        self.label_encoding = label_encoding
        
        self.num_samples = super(SemiSupervisedMNIST, self).__len__()
        self.num_unlabeled = self.num_samples - num_labeled
        self.labeled_i = np.array(range(self.num_samples))
        if self.num_unlabeled:
            self.split_data()
            
    def split_data(self):
        self.labeled_i, self.unlabeled_i = equalize(*train_test_split(self.labeled_i, test_size=self.num_unlabeled, stratify=self.labels.numpy()))

    def __len__(self):
        return len(self.labeled_i)
        
    def size(self):
        return (28*28, 10)
        
    def __getitem__(self, index):
        i_l = self.labeled_i[index]
        X_l, y_l_encoded = super(SemiSupervisedMNIST, self).__getitem__(i_l)
        X_ul = -1
        if self.num_unlabeled:
            i_ul = self.unlabeled_i[index]
            X_ul, _ = super(SemiSupervisedMNIST, self).__getitem__(i_ul)
            
        noise1 = self.distribution.sample(sample_shape=(self.noise_size,))
        noise2 = self.distribution.sample(sample_shape=(self.noise_size,))
        return {'X_labeled': X_l, 'y_labeled': y_l_encoded, 'X_unlabeled': X_ul, 'noise1': noise1, 'noise2': noise2}
