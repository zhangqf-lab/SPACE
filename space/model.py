#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 20:33:19 2022

@author: liyuzhe
"""
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, Sigmoid, LeakyReLU
from torch_geometric.nn import GAE, InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)

EPS = 1e-15


class SPACE_Graph(GAE):
    def __init__(self, encoder, decoder,loss_type='MSE'):
        super(SPACE_Graph, self).__init__(encoder, decoder)
        
        self.decoder = InnerProductDecoder()
        
        if loss_type == 'BCE':
            self.decoder_x = Sequential(Linear(in_features=self.encoder.latent_dim, out_features=128),
                                      BatchNorm1d(128),
                                      LeakyReLU(),
                                      Dropout(0.1),
                                      Linear(in_features=128, out_features=self.encoder.in_channels),
                                      Sigmoid())
        else:
            self.decoder_x = Sequential(Linear(in_features=self.encoder.latent_dim, out_features=128),
                                      BatchNorm1d(128),
                                      LeakyReLU(),
                                      Dropout(0.1),
                                      Linear(in_features=128, out_features=self.encoder.in_channels),
                                      ReLU())

    
    def encode(self, *args, **kwargs):
        
        z, attn_w = self.encoder(*args, **kwargs)
        
        return z, attn_w


    def graph_loss(self, z, pos_edge_index, neg_edge_index=None):

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
    
     
    def load_model(self, path):

        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
