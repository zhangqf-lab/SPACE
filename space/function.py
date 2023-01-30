#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:21:44 2022

@author: liyuzhe
"""
import os
import random
import numpy as np
import scanpy as sc
import scipy as sci
import networkx as nx
from sklearn.preprocessing import MaxAbsScaler

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data

from .layer import GAT_Encoder
from .model import SPACE_Graph
from .train import train_SPACE
from .utils import graph_construction



def SPACE(adata,k=20,alpha=0.5,seed=42,GPU=0,epoch=5000,lr=0.005,patience=50,outdir='./',loss_type='MSE',save_attn=False,verbose=False,heads=6):
    """
    Single-Cell integrative Analysis via Latent feature Extraction
    
    Parameters
    ----------
    adata
        An AnnData Object with spot/cell x gene matrix stored.
    k
        The number cells consider when constructing Adjacency Matrix. Default: 20.
    alpha
        The relative ratio of reconstruction loss of Adjacency Matrix. Default: 0.5 (0.5 is recommanded for domain detection, 0.05 is recommanded for cell type detection).
    lr
        Learning rate. Default: 5e-3.
    patience
        Patience in early stopping. Default: 50.
    epoch
        Max epochs for training. Default: 5000.
    loss_type
        Loss Function of feature matrix reconstruction loss. Default: MSE (BCE was recommanded for Visium data's domain finding). 
    GPU
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: '.'.
    save_attn
        Save attention weights, True or False. Default: False.
    verbose
        Verbosity, True or False. Default: False.
    heads
        Number of attention heads. Default: 6.
    
    
    Returns
    -------
    adata with the low-dimensional representation of the data stored at adata.obsm['latent'], and calculated neighbors and Umap. 
    The output folder contains:
    adata.h5ad
        The AnnData Object with the low-dimensional representation of the data stored at adata.obsm['latent'].
    checkpoint
        model.pt contains the variables of the model.
    """
    
    np.random.seed(seed) 
    torch.manual_seed(seed)
    random.seed(seed)
    
    os.makedirs(outdir, exist_ok=True)
        
    print('Construct Graph')
    graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0],k=k)
    adj = graph_dict.toarray()
    print('Average links: {:.2f}'.format(np.sum(adj>0)/adj.shape[0]))

    G = nx.from_numpy_array(adj).to_undirected() 
    edge_index = (torch_geometric.utils.convert.from_networkx(G)).edge_index
    
    if sci.sparse.issparse(adata.X):
        X_hvg = adata.X.toarray()
    else:
        X_hvg = adata.X.copy()
        
    if loss_type == 'BCE':
        scaler = MaxAbsScaler() # MinMaxScaler() 
        scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))
    elif loss_type == 'MSE':
        scaled_x = torch.from_numpy(X_hvg) 
    
    #prepare training data
    data_obj = Data(edge_index=edge_index, x=scaled_x) 
    data_obj.num_nodes = X_hvg.shape[0] 
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None

    transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                  add_negative_train_samples=False, split_labels=True)
    train_data, _ , _ = transform(data_obj) 
    num_features = data_obj.num_features
    
    print('Load SPACE model')
    encoder = GAT_Encoder(
        in_channels=num_features,
        num_heads={'first':heads,'second':heads,'mean':heads,'std':heads},
        hidden_dims=[128,128],
        dropout=[0.3,0.3,0.3,0.3],
        concat={'first': True, 'second': True})
    model = SPACE_Graph(encoder= encoder,decoder=None,loss_type=loss_type)
    print(model) 
    print('Train SPACE model')
    
    device=train_SPACE(model, train_data, epoch=epoch, lr=lr, patience=patience, GPU=GPU, seed=seed, a=alpha,loss_type=loss_type, outdir= outdir, verbose=verbose)
    
    # Load model
    pretrained_dict = torch.load(os.path.join(outdir,'model.pt'), map_location=device)                            
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    model = model.eval()

    node_embeddings = []
    x, edge_index = data_obj.x.to(torch.float).to(device), data_obj.edge_index.to(torch.long).to(device)
    z_nodes, attn_w = model.encode(x, edge_index)
    node_embeddings.append(z_nodes.cpu().detach().numpy()) 
    node_embeddings = np.array(node_embeddings) 
    node_embeddings = node_embeddings.squeeze()

    adata.obsm['latent']=node_embeddings 
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10,use_rep='latent',random_state=seed,key_added='SPACE') 
    sc.tl.umap(adata,random_state=seed, neighbors_key='SPACE')
    
    # Save attention weights
    if save_attn:
        for i in range(2):
            edge_index, attention_weights = attn_w[i]
            edge_index, attention_weights = edge_index.detach().cpu(), attention_weights.detach().cpu()
            
            layer_filename = f'GAT_Layer_{i+1}_edge_index.pt'
            edges_filepath = os.path.join(outdir, layer_filename)
            layer_filename = f'GAT_Layer_{i+1}_attention_weights.pt'
            attn_w_filepath = os.path.join(outdir, layer_filename)
            
            torch.save(edge_index, edges_filepath)
            torch.save(attention_weights, attn_w_filepath)

    #Output
    adata.write(os.path.join(outdir,'adata.h5ad'))   
    return adata
    
