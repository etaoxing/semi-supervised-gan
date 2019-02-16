from abc import ABCMeta, abstractmethod
import time
import torch
import numpy as np

from utils import to_onehot

class Metric:
    __metaclass__ = ABCMeta
    
    def __init__(self, name=None, is_scalar=True, output_transform=lambda x: x, **kwargs):
        self.reset()
        self.is_scalar = is_scalar
        self.output_transform = output_transform
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__
    
    def reset(self):
        """Reset computed value. Need to reset other values used to compute in sub-Metrics.
        """
        self.computed = None
    
    @abstractmethod
    def update(self, output, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        """Update self.computed with value
        """
        pass

class EpochRecord(Metric):
    """An EpochRecord is used to compute a set of Metrics for one epoch
    """
    def __init__(self, metrics, **kwargs):
        self.metrics = metrics
        super(EpochRecord, self).__init__(**kwargs)
        
    def reset(self):
        super(EpochRecord, self).reset()
        for metric in self.metrics: 
            metric.reset()
    
    def update(self, *output, **kwargs):
        for metric in self.metrics:
            metric.update(output, **kwargs)
            
    def compute(self):
        for metric in self.metrics:
            metric.compute()
        self.computed = {metric.name: metric.computed for metric in self.metrics}
    
class ClassAccuracy(Metric):
    def __init__(self, num_classes, **kwargs):
        super(ClassAccuracy, self).__init__(is_scalar=False, **kwargs)
        self.num_classes = num_classes
        
    def reset(self):
        super(ClassAccuracy, self).reset()
        self.correct = 0
        self.n = 0
        
    def update(self, output, **kwargs):
        y_pred, y = self.output_transform(output)
        
        dim = 1 if y_pred.dim() > 1 else 0
        _, predicted = torch.max(y_pred, dim=dim)
        predicted = to_onehot(predicted, self.num_classes)
        y = to_onehot(y, self.num_classes)
        
        correct = torch.eq(predicted, y)
        correct[1 - y.byte()] = 0
        correct = torch.sum(correct, dim=0)
                
        self.correct += correct
        self.n += torch.sum(y, dim=0)
                
    def compute(self):
        self.computed = torch.div(self.correct.float(), self.n.float()).numpy()
        # self.computed[torch.isnan(self.computed)] = 0
    
class AverageAccuracy(Metric):
    def __init__(self, **kwargs):
        super(AverageAccuracy, self).__init__(**kwargs)
    
    def reset(self):
        super(AverageAccuracy, self).reset()
        self.correct = 0
        self.n = 0
    
    def update(self, output, **kwargs):
        y_pred, y = self.output_transform(output)
        
        dim = 1 if y_pred.dim() > 1 else 0
        _, predicted = torch.max(y_pred, dim=dim)
        correct = torch.eq(predicted, y)
        correct = correct.view(-1)        
        self.correct += torch.sum(correct).item()
        self.n += correct.shape[0]
        
    def compute(self):
        if self.n == 0:
            self.computed = np.nan
        else:
            self.computed = self.correct / self.n
    
class Loss(Metric): 
    def __init__(self, loss_fn, batch_size=lambda x: x.shape[0], **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.loss_fn = loss_fn
        self.batch_size = batch_size
    
    def reset(self):
        super(Loss, self).reset()
        self.sum = 0
        self.n = 0
    
    def update(self, output, loss=None, **kwargs):
        y_pred, y = self.output_transform(output)
        
        if type(loss) is dict and self.name in loss.keys():
            loss = loss[self.name]
        else:
            loss = None
        
        if loss is None:
            loss = self.loss_fn(y_pred, y, **kwargs)
            
        N = self.batch_size(y)
        self.sum += loss.item() * N
        self.n += N
    
    def compute(self):
        if self.n == 0:
            self.computed = np.nan
        else:
            self.computed = self.sum / self.n
        
class RunTime(Metric):
    def __init__self(self, **kwargs):
        super(RunTime, self).__init__(**kwargs)
        
    def reset(self):
        super(RunTime, self).reset()
        self.start = None
        self.end = None
        
    def update(self, *args, **kwargs):
        if self.start is None:
            self.start = time.time()
        self.end = time.time()
        
    def compute(self):
        self.computed = self.end - self.start
        
class FakeAccuracy(AverageAccuracy):
    def __init__(self, **kwargs):
        super(FakeAccuracy, self).__init__(**kwargs)
        
    def update(self, *args, y_pred_fake=None, fake_class=None, **kwargs):
        if y_pred_fake is None or fake_class is None:
            return torch.Tensor([0])
        y_fake = y_pred_fake.new_ones(y_pred_fake.shape[0]) * fake_class
        super(FakeAccuracy, self).update((y_pred_fake, y_fake), **kwargs)