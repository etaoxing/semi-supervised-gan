import torch
import torch.nn as nn
import torch.nn.functional as F

# Supervised Loss
class LabeledLoss(nn.Module):
    def forward(self, y_pred_labeled, y_labeled, **kwargs):        
        pdist = y_pred_labeled.gather(1, y_labeled.view(-1, 1)).squeeze(1)
        labeled = torch.logsumexp(y_pred_labeled, dim=1)
        loss = -torch.mean(pdist) + torch.mean(labeled)
        return loss
    
# Unsupervised Loss
class UnlabeledLoss(nn.Module):    
    def forward(self, *args, y_pred_unlabeled=None, y_pred_fake=None, **kwargs):
        unlabeled = torch.logsumexp(y_pred_unlabeled, dim=1)
        fake = torch.logsumexp(y_pred_fake, dim=1)
        loss = -0.5 * torch.mean(unlabeled) + 0.5 * torch.mean(F.softplus(unlabeled)) + 0.5 * torch.mean(F.softplus(fake))
        return loss
    
# Semisupervised Loss
class DiscriminatorLoss(nn.Module):
    def __init__(self, unlabeled_weight=1.0, return_all=False):
        super(DiscriminatorLoss, self).__init__()
        self.ll = LabeledLoss()
        self.lu = UnlabeledLoss()
        self.unlabeled_weight = unlabeled_weight
        self.return_all = return_all
    
    def forward(self, y_pred_labeled, y_labeled, y_pred_unlabeled=None, y_pred_fake=None, **kwargs):       
        loss_labeled = self.ll(y_pred_labeled, y_labeled)
        loss_unlabeled = self.lu(y_pred_unlabeled=y_pred_unlabeled, y_pred_fake=y_pred_fake)
        loss = loss_labeled + self.unlabeled_weight * loss_unlabeled
        if self.return_all:
            return loss, loss_labeled, loss_unlabeled
        else:
            return loss
    
class GeneratorLoss(nn.Module):
    def forward(self, *args, y_pred_unlabeled=None, y_pred_fake=None, **kwargs):
        mom_real = torch.mean(y_pred_unlabeled, dim=0)
        mom_fake = torch.mean(y_pred_fake, dim=0)
        diff = mom_fake - mom_real
        loss = torch.mean(diff * diff)
        return loss
