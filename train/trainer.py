import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

class Trainer:
    def __init__(self,
                 model: Module,
                 criterion: Module,
                 epochs: int,
                 optimizer: Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 lr_scheduler = None,
                 device = 'cpu'):
        self.model = model
        self.criterion = criterion
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.train_losses = []
        self.val_losses = []
        self.lr = []

    def run(self):
        for i in trange(self.epochs, desc='Progress'):
            self.train()
            if self.val_loader:
                self.validate()
        
        return self.train_losses, self.val_losses

    def train(self):
        self.model.train()
        losses = []

        with tqdm(total=len(self.train_loader), desc='Train', leave=False) as pbar:
            for input, target in self.train_loader:
                input = input.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                predict = self.model(input)
                loss = self.criterion(predict, target)
                loss_value = loss.item()
                losses.append(loss_value)
                loss.backward()
                self.optimizer.step()

                pbar.set_description(f'Train Loss : {loss_value:.4f}')
                pbar.update()
            
            pbar.close()

        self.train_losses.append(losses.mean())

    def validate(self):
        self.model.eval()
        losses = []

        with tqdm(total=len(self.val_loader), desc='Validation', leave=False) as pbar, torch.no_grad():
            for input, target in self.train_loader:
                input = input.to(self.device)
                target = target.to(self.device)
                predict = self.model(input)
                loss = self.criterion(predict, target)
                loss_value = loss.item()
                losses.append(loss.item())

                pbar.set_description(f'Validation Loss : {loss_value:.4f} ')
                pbar.update()
            
            pbar.close()

        self.val_losses.append(losses.mean())