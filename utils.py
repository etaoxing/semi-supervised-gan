import torch
import numpy as np

# equalize the number of samples in two sets by taking random permutations of the data we have
def equalize(a, b):
    diff = abs(len(a) - len(b))
    if diff != 0:
        to_equalize = a if len(a) < len(b) else b
        whole_perm = []
        
        num_perm = int(diff / len(to_equalize))
        if num_perm:
            whole_perm = np.concatenate([np.random.permutation(to_equalize) for _ in range(num_perm)])
            
        remain_choice = np.random.choice(to_equalize, size=diff - len(whole_perm))
        to_equalize = np.concatenate([to_equalize, whole_perm, remain_choice]).astype(int)
            
        if len(a) < len(b):
            a = to_equalize
        else: 
            b = to_equalize
            
    return a, b

def to_onehot(indices, num_classes):
    """Convert a tensor of indices to a tensor of one-hot indicators."""
    onehot = torch.zeros(indices.shape[0], num_classes, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
