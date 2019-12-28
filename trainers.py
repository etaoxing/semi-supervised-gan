import torch
import os
from tqdm.autonotebook import tqdm
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(self, **kwargs):
        self.state = {}
        self.state['cur_epoch'] = 0
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.move_model()

    def move_model(self):
        self.model = self.model.to(self.device)

    def load(self, file):
        checkpoint = torch.load(file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history.load_state_dict(checkpoint['history'])
        self.state = checkpoint['state']
        
    def save(self, path='checkpoints/', name='checkpoint'):
        if not os.path.exists(path): os.makedirs(path)
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'state': self.state,
                    'history': self.history.state_dict()
                    }, os.path.join(path, 'e{}_{}.pth'.format(self.state['cur_epoch'], name)))
        
    def tqdm_pb(self, it, phase):
        it = tqdm(it, desc='phase {}, epoch {}:'.format(phase, self.state['cur_epoch']), leave=False)
        return it

    def run(self, max_epoch=0):
        for _ in range(self.state['cur_epoch'], max_epoch):
            self.state['cur_epoch'] += 1
            for phase in self.dataloaders.keys():
                self.run_pass(phase, grad=phase == 'train', record=self.history.record)
                self.history.save_record(phase, self.state['cur_epoch'])
                
    def run_pass(self, phase, grad=False, record=None, disp_tqdm=True):
        torch.set_grad_enabled(grad)
        if grad:
            self.model.train()
        else:
            self.model.eval()

        if record: record.reset()

        it = self.dataloaders[phase]
        if disp_tqdm: it = self.tqdm_pb(it, phase)

        for X, y in it:
            if grad: self.optimizer.zero_grad()

            X = X.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(X)
            if grad:
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

            if record: record.update(y_pred, y)

        if record: record.compute()
        return record

class GAN_Trainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def move_model(self):
        for m in ['D', 'G']:
            self.model[m] = self.model[m].to(self.device)

    def load(self, file):
        checkpoint = torch.load(file, map_location=self.device)
        for m in ['D', 'G']:
            self.model[m].load_state_dict(checkpoint[m]['model'])
            self.model[m] = self.model[m].to(self.device)
            self.optimizer[m].load_state_dict(checkpoint[m]['optimizer'])
            self.history[m].load_state_dict(checkpoint[m]['history'])

        self.state = checkpoint['state']
        
    def save(self, path='checkpoints/', name='checkpoint'):
        if not os.path.exists(path): os.makedirs(path)
        state_dict = dict(state=self.state)
        for m in ['D', 'G']:
            state_dict[m] = {}
            state_dict[m]['model'] = self.model[m].state_dict()
            state_dict[m]['optimizer'] = self.optimizer[m].state_dict()
            state_dict[m]['history'] = self.history[m].state_dict()
        
        torch.save(state_dict, os.path.join(path, 'e{}_{}.pth'.format(self.state['cur_epoch'], name)))

    def run(self, max_epoch=0):
        for _ in range(self.state['cur_epoch'], max_epoch):
            self.state['cur_epoch'] += 1
            for phase in self.dataloaders.keys():
                record = {m: history.record for m, history in self.history.items()}
                if phase == 'train':
                    self.train_pass(phase, record=record)
                else:
                    self.eval_pass(phase, m='D', record=record)
                
                for m, history in self.history.items():
                    history.save_record(phase, self.state['cur_epoch'])

    def eval_pass(self, phase, m='D', record=None, disp_tqdm=True):
        torch.set_grad_enabled(False)
        self.model[m].eval()

        if record: record[m].reset()
            
        it = self.dataloaders[phase]
        if disp_tqdm: it = self.tqdm_pb(it, phase)

        for X, y in it:
            X = X.to(self.device)
            y = y.to(self.device)
            
            y_pred = self.model[m](X)

            if record:
                placeholder = torch.Tensor([0])
                record[m].update(y_pred, y,
                                 loss=dict(loss_D=placeholder,
                                           loss_labeled=placeholder, 
                                           loss_unlabeled=placeholder)) 

        if record: record[m].compute()
        return record

    def train_pass(self, phase, record=None, disp_tqdm=True, clip_grad=35):
        torch.set_grad_enabled(True)

        if record:
            record['D'].reset()
            record['G'].reset()

        it = self.dataloaders[phase]
        if disp_tqdm: it = self.tqdm_pb(it, phase)

        for data in it:
            for k, v in data.items():
                data[k] = v.to(self.device)

            X_labeled, y_labeled, X_unlabeled, noise1, noise2 = data.values()

            # ----
            # train discriminator
            # ----            
            self.model['D'].train()
            self.model['G'].eval()
            self.optimizer['D'].zero_grad()

            y_pred_labeled = self.model['D'](X_labeled)
            y_pred_unlabeled = self.model['D'](X_unlabeled)
            y_pred_fake = self.model['D'](self.model['G'](noise1))
            
            loss_D, loss_labeled, loss_unlabeled = self.criterion['D'](y_pred_labeled, y_labeled, y_pred_unlabeled=y_pred_unlabeled, y_pred_fake=y_pred_fake)
            loss_D.backward()
            if clip_grad: clip_grad_norm_(self.model['D'].parameters(), clip_grad)
            self.optimizer['D'].step()

            if record:
                record['D'].update(y_pred_labeled, 
                                   y_labeled,
                                   y_pred_unlabeled=y_pred_unlabeled,
                                   y_pred_fake=y_pred_fake,
                                   fake_class=self.dataloaders[phase].dataset.label_encoding['fake'],
                                   loss=dict(loss_D=loss_D,
                                             loss_labeled=loss_labeled, 
                                             loss_unlabeled=loss_unlabeled))
                
            # ----
            # train generator
            # ---- 
            self.model['D'].eval()
            self.model['G'].train()
            self.optimizer['G'].zero_grad()
            
            X_fake = self.model['G'](noise2)
            
            net_D = self.model['D']
            if self.model['D'].feature_matching:
                net_D = net_D.mid

            y_pred_unlabeled = net_D(X_unlabeled)
            y_pred_fake = net_D(X_fake)
            
            loss_G = self.criterion['G'](y_pred_unlabeled=y_pred_unlabeled, y_pred_fake=y_pred_fake)
            loss_G.backward()
            if clip_grad: clip_grad_norm_(self.model['G'].parameters(), clip_grad)
            self.optimizer['G'].step()

            if record:
                record['G'].update(None, 
                                   torch.cat([y_pred_fake, y_pred_unlabeled]),
                                   y_pred_unlabeled=y_pred_unlabeled, 
                                   y_pred_fake=y_pred_fake,
                                   fake_class=self.dataloaders[phase].dataset.label_encoding['fake'],
                                   loss=dict(loss_G=loss_G))

        if record:
            record['D'].compute()
            record['G'].compute()

        return record
