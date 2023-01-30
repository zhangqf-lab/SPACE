#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:58:54 2022

@author: liyuzhe
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv



class GAT_Encoder(nn.Module):
    def __init__(self, in_channels,num_heads, hidden_dims, dropout, concat):
        super(GAT_Encoder, self).__init__()
        
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.latent_dim = 10
        self.conv = GATv2Conv

        self.hidden_layer1 = self.conv(
            in_channels=in_channels, out_channels=hidden_dims[0],
            heads=self.num_heads['first'],
            dropout=dropout[0],
            concat=concat['first'])
        in_dim_hidden = hidden_dims[0] * self.num_heads['first'] if concat['first'] else hidden_dims[0]

        self.hidden_layer2 = self.conv(
            in_channels=in_dim_hidden, out_channels=hidden_dims[1],
            heads=self.num_heads['second'],
            dropout=dropout[1],
            concat=concat['second'])
        in_dim_final = hidden_dims[-1] * self.num_heads['second'] if concat['second'] else hidden_dims[-1]
        

        self.conv_z = self.conv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                heads=self.num_heads['mean'], concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        hidden_out1, attn_w_1 = self.hidden_layer1(x, edge_index, return_attention_weights=True)
        hidden_out1 = F.leaky_relu(hidden_out1)
        hidden_out2, attn_w_2 = self.hidden_layer2(hidden_out1, edge_index, return_attention_weights=True)
        hidden_out2 = F.leaky_relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        last_out = hidden_out2
        z, attn_w_z = self.conv_z(last_out, edge_index, return_attention_weights=True)
       
        return z , (attn_w_1, attn_w_2, attn_w_z)
    

class InnerProduct(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self,act=torch.relu):
        super(InnerProduct, self).__init__()
        self.act = act

    def forward(self, z):
        recon_a = self.act(torch.mm(z, z.t()))
        return recon_a 