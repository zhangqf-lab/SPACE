#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:09:45 2022

@author: liyuzhe
"""
import os
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from torch_geometric import seed_everything
import matplotlib.pyplot as plt

def adjust_learning_rate(init_lr, optimizer, iteration,seperation):
    lr = max(init_lr * (0.9 ** (iteration//seperation)), 0.0001)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, checkpoint_file=''):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 10
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        # loss=loss.cpu().detach().numpy()
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter > self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss



def train_SPACE(model, train_data, outdir, epoch=5000,lr=0.005, a=0.5,loss_type='MSE',patience=50, GPU=0,seed=42, verbose=False):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    seed_everything(seed)
    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(GPU)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        device='cpu'
    
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.

    model.to(device)
    model.train()
    early_stopping = EarlyStopping(patience=patience, checkpoint_file=os.path.join(outdir,'model.pt'))
    
    y_loss = {}  # loss history
    y_loss['feature_loss'] = []
    y_loss['graph_loss'] = []
    y_loss['loss']=[]
    x_epoch = []
    fig = plt.figure()

    for epoch in tqdm(range(0, epoch+1)):
        epoch_loss = 0.0
        loss1 = 0.0
        loss2 = 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        epoch_lr=adjust_learning_rate(lr, optimizer, epoch, seperation=50)
            
        x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device) 
        optimizer.zero_grad()
            
        z, _ = model.encode(x, edge_index) 
        graph_loss = model.graph_loss(z, train_data.pos_edge_label_index) * a  
        loss = graph_loss  
            
        reconstructed_features = model.decoder_x(z)
        if loss_type=='BCE': 
            feature_loss = torch.nn.functional.binary_cross_entropy(reconstructed_features, x) * 10
        elif loss_type=='MSE':
            feature_loss = torch.nn.functional.mse_loss(reconstructed_features, x) * 10
            
        loss += feature_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        loss1 += graph_loss.item()
        loss2 += feature_loss.item()
       
            
        # if epoch%50==0:
        #     if verbose:
        #         print('Epoch {:03d} -- Total epoch loss: {:.4f} -- Feature decoder epoch loss: {:.4f} -- Graph decoder epoch loss: {:.4f} -- epoch lr: {:.4f}'.format(epoch, epoch_loss, feature_loss, epoch_loss-feature_loss, epoch_lr))
        #     else:
        #         print('====> Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss)) 
        # early_stopping(epoch_loss, model)
        # if early_stopping.early_stop:
        #     print('EarlyStopping: run {} iteration'.format(epoch))
        #     break
        if verbose:
            if epoch%50==0:
                print('====> Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss)) 
        early_stopping(epoch_loss, model)
        # plot loss
        y_loss['feature_loss'].append(loss2)
        y_loss['graph_loss'].append(loss1)
        y_loss['loss'].append(epoch_loss)
        x_epoch.append(epoch)
        plt.plot(x_epoch, y_loss['loss'], 'go-', label='loss',linewidth=1, markersize=2)
        plt.plot(x_epoch, y_loss['graph_loss'], 'ro-', label='graph_loss',linewidth=1, markersize=2)
        plt.plot(x_epoch, y_loss['feature_loss'], 'bo-', label='feature_loss',linewidth=1, markersize=2)
        if len(x_epoch)==1:
            plt.legend()
        
        if early_stopping.early_stop:
            print('====> Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss)) 
            print('EarlyStopping: run {} iteration'.format(epoch))
            break
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(os.path.join(outdir, 'train_loss.pdf'))
    
    return device
